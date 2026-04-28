from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_CLASS_ID_MAPPING = {15: 0, 16: 1}
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class _PairedSample:
    image_path: Path
    label_path: Path


class GunmenYoloDataset(Dataset):
    """PyTorch Dataset for Gunmen images with YOLO-style labels.

    Returns tuples in the format:
    - image: transformed image (or PIL Image if no transform)
    - targets: torch.FloatTensor with shape [N, 5] in format
      [class, x_center, y_center, width, height]

    Default remapping converts source IDs {15, 16} -> {0, 1}.
    """

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
        image_transform: Callable[[Image.Image], Any] | None = None,
        target_transform: Callable[[torch.Tensor], Any] | None = None,
        strict: bool = True,
        class_id_mapping: dict[int, int] | None = None,
    ) -> None:
        self.dataset_root = (
            Path(dataset_root) if dataset_root else self._default_dataset_root()
        )
        self.dataset_root = self.dataset_root.expanduser().resolve()
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.strict = strict
        self.class_id_mapping = dict(class_id_mapping or DEFAULT_CLASS_ID_MAPPING)

        if not self.dataset_root.exists() or not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_root}")

        self._samples, self._missing_images, self._missing_labels = self._build_index(
            self.dataset_root,
            self.image_extensions,
        )

        if self.strict and (self._missing_images or self._missing_labels):
            raise ValueError(
                "Dataset index contains missing image/label pairs. "
                f"Missing images: {len(self._missing_images)}, "
                f"missing labels: {len(self._missing_labels)}"
            )

        self._class_names = self._load_class_names()

    @staticmethod
    def _default_dataset_root() -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "data"
            / "sources"
            / "Gunmen Dataset"
            / "All"
        )

    @property
    def class_names(self) -> list[str]:
        """Class names aligned to remapped IDs, when available."""
        return list(self._class_names)

    def get_sample_path(self, index: int) -> Path:
        return self._samples[index].image_path

    def get_raw_label_path(self, index: int) -> Path:
        return self._samples[index].label_path

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor]:
        sample = self._samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        targets = self._parse_yolo_label_file(sample.label_path)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return image, targets

    def _build_index(
        self,
        dataset_root: Path,
        image_extensions: tuple[str, ...],
    ) -> tuple[list[_PairedSample], list[Path], list[Path]]:
        image_map: dict[str, Path] = {}
        label_map: dict[str, Path] = {}

        for path in dataset_root.iterdir():
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            stem = path.stem

            if suffix in image_extensions:
                image_map[stem] = path
            elif suffix == ".txt" and path.name != "classes.txt":
                label_map[stem] = path

        stems = sorted(set(image_map) | set(label_map))
        pairs: list[_PairedSample] = []
        missing_images: list[Path] = []
        missing_labels: list[Path] = []

        for stem in stems:
            image_path = image_map.get(stem)
            label_path = label_map.get(stem)

            if image_path and label_path:
                pairs.append(
                    _PairedSample(image_path=image_path, label_path=label_path)
                )
            elif image_path and not label_path:
                missing_labels.append(image_path)
            elif label_path and not image_path:
                missing_images.append(label_path)

        return pairs, missing_images, missing_labels

    def _parse_yolo_label_file(self, label_path: Path) -> torch.Tensor:
        rows: list[list[float]] = []

        with label_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    message = (
                        f"Invalid YOLO row in {label_path} at line {line_number}: "
                        f"expected 5 values, got {len(parts)}"
                    )
                    if self.strict:
                        raise ValueError(message)
                    warnings.warn(message, stacklevel=2)
                    continue

                try:
                    source_class = int(float(parts[0]))
                    x_center, y_center, width, height = (
                        float(value) for value in parts[1:]
                    )
                except ValueError as exc:
                    message = f"Non-numeric YOLO row in {label_path} at line {line_number}: {line}"
                    if self.strict:
                        raise ValueError(message) from exc
                    warnings.warn(message, stacklevel=2)
                    continue

                if source_class not in self.class_id_mapping:
                    message = f"Unknown class ID {source_class} in {label_path} at line {line_number}"
                    if self.strict:
                        raise ValueError(message)
                    warnings.warn(message, stacklevel=2)
                    continue

                remapped_class = float(self.class_id_mapping[source_class])
                rows.append([remapped_class, x_center, y_center, width, height])

        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def _load_class_names(self) -> list[str]:
        candidate_paths = [
            self.dataset_root / "classes.txt",
            self.dataset_root.parent / "classes.txt",
        ]

        source_lines: list[str] = []
        for path in candidate_paths:
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    source_lines = [line.strip() for line in handle]
                break

        if not source_lines:
            return ["human", "gun"]

        cleaned_lines = []
        for raw in source_lines:
            cleaned = raw.strip().strip('"').strip("'")
            if not cleaned:
                continue
            cleaned_lines.append(cleaned)

        if not cleaned_lines:
            return ["human", "gun"]

        reverse_map = {dst: src for src, dst in self.class_id_mapping.items()}
        max_mapped_id = max(reverse_map) if reverse_map else -1
        names: list[str] = []

        for remapped_id in range(max_mapped_id + 1):
            source_id = reverse_map.get(remapped_id)
            if source_id is None:
                names.append(f"class_{remapped_id}")
                continue

            if 0 <= source_id < len(cleaned_lines):
                names.append(cleaned_lines[source_id])
            else:
                names.append(f"class_{remapped_id}")

        return names


def validate_gunmen_dataset_integrity(
    dataset_root: str | Path | None = None,
    image_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
    class_id_mapping: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Validate Gunmen dataset pair integrity and class frequencies.

    The validation is non-strict: malformed or unknown rows are skipped and
    counted under `ignored_label_rows`.
    """

    root = (
        Path(dataset_root)
        if dataset_root
        else GunmenYoloDataset._default_dataset_root()
    )
    root = root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    mapping = dict(class_id_mapping or DEFAULT_CLASS_ID_MAPPING)
    extensions = tuple(ext.lower() for ext in image_extensions)

    dataset = GunmenYoloDataset(
        dataset_root=root,
        image_extensions=extensions,
        strict=False,
        class_id_mapping=mapping,
    )

    frequencies: Counter[int] = Counter()
    ignored_rows = 0

    for sample in dataset._samples:
        with sample.label_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    ignored_rows += 1
                    continue

                try:
                    source_class = int(float(parts[0]))
                except ValueError:
                    ignored_rows += 1
                    continue

                remapped_class = mapping.get(source_class)
                if remapped_class is None:
                    ignored_rows += 1
                    continue

                frequencies[remapped_class] += 1

    return {
        "dataset_root": str(root),
        "paired_samples": len(dataset),
        "missing_image_count": len(dataset._missing_images),
        "missing_label_count": len(dataset._missing_labels),
        "missing_images": [str(path) for path in dataset._missing_images],
        "missing_labels": [str(path) for path in dataset._missing_labels],
        "class_frequencies": dict(sorted(frequencies.items())),
        "ignored_label_rows": ignored_rows,
    }
