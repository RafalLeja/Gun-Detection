from __future__ import annotations

from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from src.datasets.gunmen_dataset import GunmenYoloDataset


def collate_gunmen_yolo_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    class_targets: list[torch.Tensor] = []
    bbox_targets: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []

    for batch_index, target in enumerate(targets):
        if target.numel() == 0:
            continue

        class_targets.append(target[:, :1].to(dtype=torch.long))
        bbox_targets.append(target[:, 1:5].to(dtype=torch.float32))
        batch_indices.append(torch.full((target.shape[0], 1), batch_index, dtype=torch.long))

    if class_targets:
        cls = torch.cat(class_targets, dim=0)
        bboxes = torch.cat(bbox_targets, dim=0)
        batch_idx = torch.cat(batch_indices, dim=0)
    else:
        cls = torch.zeros((0, 1), dtype=torch.long)
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0, 1), dtype=torch.long)

    return {
        "img": images,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
    }


class GunmenYoloDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        batch_size: int = 8,
        image_size: int = 640,
        num_workers: int = 4,
        val_split: float = 0.2,
        strict: bool = False,
        transforms: Callable | None = None,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.strict = strict
        self.transforms = transforms

        self.class_names: list[str] = []
        self.num_classes: int = 0
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str | None = None) -> None:
        transform = self.transforms
        if transform is None:
            transform = T.Compose(
                [
                    T.Resize((self.image_size, self.image_size)),
                    T.ToTensor(),
                ]
            )

        dataset = GunmenYoloDataset(
            dataset_root=self.dataset_root,
            image_transform=transform,
            strict=self.strict,
        )

        if len(dataset) == 0:
            raise ValueError("Gunmen YOLO dataset is empty.")

        self.class_names = dataset.class_names
        self.num_classes = len(self.class_names)

        val_size = max(1, int(len(dataset) * self.val_split))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Validation split is too large for the available dataset size.")

        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_gunmen_yolo_batch,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_gunmen_yolo_batch,
        )
