import torch

from src.datasets.gunmen_yolo_datamodule import collate_gunmen_yolo_batch


def test_collate_gunmen_yolo_batch_handles_empty_targets():
    batch = [
        (
            torch.zeros((3, 8, 8), dtype=torch.float32),
            torch.tensor([[1.0, 0.5, 0.5, 0.25, 0.25]], dtype=torch.float32),
        ),
        (
            torch.zeros((3, 8, 8), dtype=torch.float32),
            torch.zeros((0, 5), dtype=torch.float32),
        ),
    ]

    collated = collate_gunmen_yolo_batch(batch)

    assert collated["img"].shape == (2, 3, 8, 8)
    assert collated["cls"].shape == (1, 1)
    assert collated["bboxes"].shape == (1, 4)
    assert collated["batch_idx"].tolist() == [[0]]
