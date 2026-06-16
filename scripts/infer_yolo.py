from __future__ import annotations

import random
from pathlib import Path

import click
import fiddle as fdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

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


def resolve_source_image(source: Path | None, dataset_root: Path | None, pick_index: int | None = None) -> Path:
    if source is not None:
        if source.is_dir():
            image_paths = collect_image_paths(source)
            if not image_paths:
                raise FileNotFoundError(f"No images found in directory: {source}")
            if pick_index is not None:
                if not (0 <= pick_index < len(image_paths)):
                    raise IndexError(f"pick_index {pick_index} is out of bounds for {len(image_paths)} images in {source}")
                return image_paths[pick_index]
            return random.choice(image_paths)
        return source

    dataset_folder = dataset_root or GunmenYoloDataset._default_dataset_root()
    image_paths = collect_image_paths(dataset_folder)
    if not image_paths:
        raise FileNotFoundError(f"No images found in dataset folder: {dataset_folder}")
    return random.choice(image_paths)


def preprocess_image(
    image: Image.Image, image_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    image_array = np.asarray(image)
    resized_array = LetterBox(new_shape=(image_size, image_size), auto=False)(
        image=image_array
    )
    tensor = torch.from_numpy(np.ascontiguousarray(resized_array)).permute(2, 0, 1)
    tensor = tensor.float().div(255.0).unsqueeze(0)
    return tensor, resized_array.shape[:2]


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


def make_unique(path: Path) -> Path:
    """Return a non-existing Path by appending _N before the suffix if needed."""
    path = Path(path)
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    candidate = path
    i = 1
    while candidate.exists():
        candidate = parent / f"{stem}_{i}{suffix}"
        i += 1
    return candidate


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
    default=Path("outputs/yolo_inference.png"),
    show_default=True,
    help="Where to save the plotted detections.",
)
@click.option("--threshold", type=float, default=0.25, show_default=True)
@click.option("--iou-threshold", type=float, default=0.45, show_default=True)
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
    iou_threshold: float,
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

    dataset_root = getattr(built_cfg.data_module, "dataset_root", None)
    if dataset_root is not None:
        dataset_root = Path(dataset_root)

    source_path = resolve_source_image(source, dataset_root, pick_index=397)
    image = Image.open(source_path).convert("RGB")
    source_shape = (image.height, image.width)

    image_size = getattr(built_cfg.data_module, "image_size", 640)
    tensor, inference_shape = preprocess_image(image, image_size)
    tensor = tensor.to(device_obj)

    with torch.no_grad():
        predictions = model(tensor)[0]
        detections = (
            non_max_suppression(
                predictions,
                conf_thres=threshold,
                iou_thres=iou_threshold,
                nc=getattr(model, "nc", 2),
            )[0]
            .detach()
            .cpu()
        )

    class_names = GunmenYoloDataset(dataset_root=dataset_root, strict=False).class_names
    id2label = {index: name for index, name in enumerate(class_names)}

    # Scale boxes from the letterboxed inference space back to the original image.
    if detections.numel():
        boxes = scale_boxes(inference_shape, detections[:, :4].clone(), source_shape)
        sv_detections = sv.Detections(
            xyxy=boxes.numpy(),
            confidence=detections[:, 4].numpy(),
            class_id=detections[:, 5].to(torch.int64).numpy(),
        )
        sv_detections = sv_detections.with_nms(threshold=iou_threshold)
    else:
        sv_detections = sv.Detections.empty()

    annotated_image = annotate_image(image, sv_detections, id2label)

    title = f"{source_path.name} | detections: {len(sv_detections)}"
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(annotated_image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    # Ensure output directory exists and choose unique filenames for before/after.
    output.parent.mkdir(parents=True, exist_ok=True)
    before_target = output.with_name(f"{output.stem}_before{output.suffix}")
    after_target = output.with_name(f"{output.stem}_after{output.suffix}")
    before_path = make_unique(before_target)
    after_path = make_unique(after_target)

    image.save(before_path)
    fig.savefig(after_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[*] Source: {source_path}")
    print(f"[*] Saved original image to: {before_path}")
    print(f"[*] Saved plotted detections to: {after_path}")
    print(f"[*] Detections kept above threshold {threshold}: {len(sv_detections)}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
