#!/usr/bin/env python3
"""Real-time webcam inference with RF-DETR via OpenCV.

Usage:
    uv run webcam_rfdetr.py src/config/rfdetr_detection.py notebooks/artifacts/rfdetr.ckpt \
        --iou-threshold 0.3 --threshold 0.5
"""

from __future__ import annotations

from pathlib import Path

import click
import cv2
import fiddle as fdl
import numpy as np
import torch
from PIL import Image

from src.utils.config import parse_fiddle_config

COLORS = [
    (0, 200, 255),
    (255, 80,  80),
    (80,  255, 80),
    (255, 200,  0),
    (200,  80, 255),
]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    id2label: dict[int, str],
) -> np.ndarray:
    out = frame.copy()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[int(cls_id) % len(COLORS)]
        label_text = f"{id2label.get(int(cls_id), str(int(cls_id)))}: {score:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        tag_y = max(y1 - 4, th + baseline)
        cv2.rectangle(
            out,
            (x1, tag_y - th - baseline),
            (x1 + tw + 4, tag_y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            out, label_text, (x1 + 2, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
        )
    return out


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("ckpt_path",   type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--threshold",     type=float, default=0.5,    show_default=True)
@click.option("--iou-threshold", type=float, default=0.3,    show_default=True)
@click.option("--device",        type=str,   default="auto", show_default=True)
@click.option("--camera-id",     type=int,   default=0,      show_default=True)
@click.option("--width",         type=int,   default=1280,   show_default=True)
@click.option("--height",        type=int,   default=720,    show_default=True)
def main(
    config_path: Path,
    ckpt_path: Path,
    threshold: float,
    iou_threshold: float,
    device: str,
    camera_id: int,
    width: int,
    height: int,
) -> None:
    """Stream live webcam feed through RF-DETR and display annotated frames."""
    import supervision as sv

    # ── Load model ────────────────────────────────────────────────────────────
    print("[*] Loading config…")
    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)

    model = built_cfg.model
    print(f"[*] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    device_obj = resolve_device(device)
    print(f"[*] Using device: {device_obj}")
    model = model.to(device_obj).eval()

    image_processor = model.image_processor
    id2label: dict[int, str] = dict(model.detector.config.id2label)
    print(f"[*] Classes: {id2label}")

    # ── Open webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("[*] Webcam open. Press Q / ESC to quit, S to save snapshot, +/- to adjust threshold.")

    fps_counter = 0
    fps_display = 0.0
    tick = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # ── Inference ─────────────────────────────────────────────────────────
        inputs = image_processor(images=pil_image, return_tensors="pt").to(device_obj)
        with torch.no_grad():
            outputs = model.detector(**inputs)

        h, w = frame.shape[:2]
        results = image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]]),
            threshold=threshold,
        )[0]

        detections = sv.Detections.from_transformers(
            transformers_results=results, id2label=id2label
        ).with_nms(threshold=iou_threshold)

        # ── Draw ──────────────────────────────────────────────────────────────
        annotated = (
            draw_detections(frame, detections.xyxy, detections.confidence, detections.class_id, id2label)
            if len(detections) > 0
            else frame.copy()
        )

        # FPS
        fps_counter += 1
        if fps_counter >= 10:
            elapsed    = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
            fps_display = fps_counter / elapsed
            fps_counter = 0
            tick = cv2.getTickCount()

        cv2.putText(annotated,
            f"FPS: {fps_display:.1f}  |  det: {len(detections)}  |  thr: {threshold:.2f}  iou: {iou_threshold:.2f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated,
            f"device: {device_obj}  |  Q/ESC=quit  S=snapshot  +/-=threshold",
            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("RF-DETR  |  Gunmen Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord("+"):
            threshold = min(threshold + 0.05, 0.99)
            print(f"[+] threshold -> {threshold:.2f}")
        elif key == ord("-"):
            threshold = max(threshold - 0.05, 0.05)
            print(f"[-] threshold -> {threshold:.2f}")
        elif key in (ord("s"), ord("S")):
            save_path = f"webcam_capture_{cv2.getTickCount()}.jpg"
            cv2.imwrite(save_path, annotated)
            print(f"[s] Snapshot saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Done.")


if __name__ == "__main__":
    main()
