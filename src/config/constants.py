from pathlib import Path

WANDB_ENTITY = "mkarapka-uniwroc"
WANDB_PROJECT = "gun-detection"


class Constants:
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data"

    classes = ["person", "gun"]

    yolo_26_medium = "yolo26m.pt"
    rf_detr_medium = "Roboflow/rf-detr-medium"

    manual_seed = 42
