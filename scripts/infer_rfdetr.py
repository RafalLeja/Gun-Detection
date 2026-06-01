from __future__ import annotations

import random
from pathlib import Path

import click
import fiddle as fdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.datasets.gunmen_dataset import GunmenYoloDataset
from src.utils.config import parse_fiddle_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_image_paths(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_source_image(source: Path | None, dataset_root: Path | None) -> Path:
    if source is not None:
        if source.is_dir():
            image_paths = collect_image_paths(source)
            if not image_paths:
                raise FileNotFoundError(f"No images found in directory: {source}")
            return random.choice(image_paths)
        return source

    dataset_folder = dataset_root or GunmenYoloDataset._default_dataset_root()
    image_paths = collect_image_paths(dataset_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in dataset folder: {dataset_folder}")
    return random.choice(image_paths)


def annotate_image(
    image: Image.Image,
    detections,
    id2label: dict[int, str],
) -> np.ndarray:
    import supervision as sv

    labels = [
        f"{id2label.get(int(class_id), f'class_{int(class_id)}')}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    annotated = image.copy()
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    return np.asarray(annotated)


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "ckpt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    default=None,
    help="Image file or directory. If omitted, a random dataset image is used.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("outputs/rfdetr_inference.png"),
    show_default=True,
    help="Where to save the plotted detections.",
)
@click.option("--threshold", type=float, default=0.3, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option(
    "--show", is_flag=True, default=False, help="Display the plot in a window."
)
def main(
    config_path: Path,
    ckpt_path: Path,
    source: Path | None,
    output: Path,
    threshold: float,
    device: str,
    show: bool,
) -> None:
    import supervision as sv

    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)

    model = built_cfg.model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    device_obj = resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    image_processor = model.image_processor

    dataset_root = getattr(built_cfg.data_module, "dataset_root", None)
    if dataset_root is not None:
        dataset_root = Path(dataset_root)

    source_path = resolve_source_image(source, dataset_root)
    image = Image.open(source_path).convert("RGB")

    inputs = image_processor(images=image, return_tensors="pt").to(device_obj)
    with torch.no_grad():
        outputs = model.detector(**inputs)

    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([image.size[::-1]]),
        threshold=threshold,
    )[0]

    id2label = dict(model.detector.config.id2label)
    detections = sv.Detections.from_transformers(
        transformers_results=results, id2label=id2label
    )

    annotated_image = annotate_image(image, detections, id2label)

    title = f"{source_path.name} | detections: {len(detections)}"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(annotated_image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[*] Source: {source_path}")
    print(f"[*] Saved plotted detections to: {output}")
    print(f"[*] Detections kept above threshold {threshold}: {len(detections)}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(Image.open(output))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
