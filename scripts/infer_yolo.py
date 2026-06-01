from __future__ import annotations

import random
from pathlib import Path

import click
import fiddle as fdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
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
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


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


def preprocess_image(image: Image.Image, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image_array = np.asarray(image)
    resized_array = LetterBox(new_shape=(image_size, image_size), auto=False)(image=image_array)
    tensor = torch.from_numpy(np.ascontiguousarray(resized_array)).permute(2, 0, 1)
    tensor = tensor.float().div(255.0).unsqueeze(0)
    return tensor, resized_array.shape[:2]


def plot_detections(
    image: Image.Image,
    detections: torch.Tensor,
    class_names: list[str],
    source_shape: tuple[int, int],
    inference_shape: tuple[int, int],
    output_path: Path,
    title: str,
) -> None:
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    if detections.numel():
        boxes = detections[:, :4].clone()
        boxes = scale_boxes(inference_shape, boxes, source_shape)
        boxes = boxes.round().to(torch.int64)

        for box, det in zip(boxes.tolist(), detections.tolist()):
            x1, y1, x2, y2 = box
            confidence = det[4]
            class_id = int(det[5])
            label = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"

            color = "#d7263d" if class_id == 1 else "#1d4ed8"
            text_y = max(0, y1 - 14)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 4, text_y), f"{label}: {confidence:.2f}", fill=color)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(annotated_image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ckpt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
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
@click.option("--conf-threshold", type=float, default=0.25, show_default=True)
@click.option("--iou-threshold", type=float, default=0.45, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--show", is_flag=True, default=False, help="Display the plot in a window.")
def main(
    config_path: Path,
    ckpt_path: Path,
    source: Path | None,
    output: Path,
    conf_threshold: float,
    iou_threshold: float,
    device: str,
    show: bool,
) -> None:
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

    source_path = resolve_source_image(source, dataset_root)
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
                conf_thres=conf_threshold,
                iou_thres=iou_threshold,
                nc=getattr(model, "nc", 2),
            )[0]
            .detach()
            .cpu()
        )

    class_names = GunmenYoloDataset(dataset_root=dataset_root, strict=False).class_names
    title = f"{source_path.name} | detections: {len(detections)}"
    plot_detections(
        image=image,
        detections=detections,
        class_names=class_names,
        source_shape=source_shape,
        inference_shape=inference_shape,
        output_path=output,
        title=title,
    )

    print(f"[*] Source: {source_path}")
    print(f"[*] Saved plotted detections to: {output}")
    print(f"[*] Detections kept after NMS: {len(detections)}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(Image.open(output))
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
