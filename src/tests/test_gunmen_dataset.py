from __future__ import annotations

from pathlib import Path
import random
import unittest

import torch
from torch.utils.data import DataLoader

from src.datasets import GunmenYoloDataset, validate_gunmen_dataset_integrity


class TestGunmenYoloDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = GunmenYoloDataset(strict=False)

    def test_dataset_not_empty(self) -> None:
        self.assertGreater(len(self.dataset), 0)

    def test_sample_structure(self) -> None:
        image, targets = self.dataset[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(targets.dtype, torch.float32)
        self.assertEqual(targets.ndim, 2)
        self.assertEqual(targets.shape[1], 5)

    def test_random_targets_valid(self) -> None:
        sample_count = min(20, len(self.dataset))
        indices = random.sample(range(len(self.dataset)), k=sample_count)

        for idx in indices:
            _, targets = self.dataset[idx]
            self.assertEqual(targets.ndim, 2)
            self.assertEqual(targets.shape[1], 5)

            if targets.numel() == 0:
                continue

            class_ids = set(targets[:, 0].tolist())
            self.assertTrue(class_ids.issubset({0.0, 1.0}))

            box_values = targets[:, 1:]
            self.assertTrue(torch.all((box_values >= 0.0) & (box_values <= 1.0)))

    def test_path_accessors(self) -> None:
        image_path = self.dataset.get_sample_path(0)
        label_path = self.dataset.get_raw_label_path(0)
        self.assertIsInstance(image_path, Path)
        self.assertIsInstance(label_path, Path)
        self.assertTrue(image_path.exists())
        self.assertTrue(label_path.exists())

    def test_integrity_utility(self) -> None:
        report = validate_gunmen_dataset_integrity()
        self.assertIn("paired_samples", report)
        self.assertIn("class_frequencies", report)
        self.assertGreater(report["paired_samples"], 0)
        self.assertGreaterEqual(report["missing_image_count"], 0)
        self.assertGreaterEqual(report["missing_label_count"], 0)
        self.assertEqual(report["missing_image_count"], len(report["missing_images"]))
        self.assertEqual(report["missing_label_count"], len(report["missing_labels"]))

    def test_dataloader_collate(self) -> None:
        def yolo_collate_fn(batch):
            images, targets = zip(*batch)
            return list(images), list(targets)

        loader = DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=yolo_collate_fn,
        )

        images, targets = next(iter(loader))
        self.assertEqual(len(images), 4)
        self.assertEqual(len(targets), 4)
        self.assertTrue(all(isinstance(t, torch.Tensor) for t in targets))


if __name__ == "__main__":
    unittest.main()
