import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import lightning as L
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from src.datasets.gunmen_dataset import GunmenYoloDataset


def bbox_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Oblicza Intersection over Union (IoU) dla dwóch bounding boxów [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if b1_area + b2_area - inter_area == 0:
        return 0.0
    return inter_area / (b1_area + b2_area - inter_area)


class GunmenCropDataset(Dataset):
    """
    Dataset wycinający małe kwadraty (crops) z pełnych zdjęć na podstawie etykiet YOLO.
    Zwraca pary: (Wycinek Obrazka Tensor [C, H, W], Etykieta [1.0 dla broni, 0.0 dla tła]).
    """
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        crop_size: int = 128,
        negatives_per_positive: int = 1,
        transform: Optional[Callable] = None,
    ):
        # Inicjalizujemy stary dataset bez transformacji wizualnych - posłuży tylko
        # do łatwego znalezienia ścieżek i odczytania współrzędnych YOLO.
        self.base_dataset = GunmenYoloDataset(
            dataset_root=dataset_root,
            image_transform=None,
            target_transform=None,
            strict=False
        )
        self.crop_size = crop_size
        self.transform = transform or T.Compose([
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
        ])

        # Lista zapamiętująca wszystkie wycinki w formacie:
        # (ścieżka_do_zdjęcia, (lewy_X, górny_Y, prawy_X, dolny_Y), klasa_0_lub_1)
        self.crops: List[Tuple[Path, Tuple[float, float, float, float], int]] = []

        print("Przygotowywanie bazy wycinków (positives & negatives)...")
        self._prepare_crops(negatives_per_positive)
        print(f"Gotowe! Wygenerowano łącznie {len(self.crops)} wycinków.")

    def _prepare_crops(self, negatives_per_positive: int) -> None:
        """Skanuje cały zbiór, zapisuje współrzędne broni i losuje tło."""
        for i in range(len(self.base_dataset)):
            img_path = self.base_dataset.get_sample_path(i)
            label_path = self.base_dataset.get_raw_label_path(i)

            # Wymiary obrazka potrzebne do denormalizacji bboxów z YOLO
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            targets = self.base_dataset._parse_yolo_label_file(label_path)

            pos_boxes = []
            for t in targets:
                # YOLO: Class, Center_X, Center_Y, Width, Height
                cls_id, cx, cy, w, h = t.tolist()

                # Dodajemy 1 do ID klasy, żeby zwolnić 0 dla Tła.
                # Wtedy: 1 = Człowiek, 2 = Broń
                label = int(cls_id) + 1

                # Denormalizacja do pikseli
                px_cx, px_cy = cx * img_w, cy * img_h
                px_w, px_h = w * img_w, h * img_h

                # Zmiana na lewy, górny, prawy, dolny róg
                left = max(0, px_cx - px_w / 2)
                top = max(0, px_cy - px_h / 2)
                right = min(img_w, px_cx + px_w / 2)
                bottom = min(img_h, px_cy + px_h / 2)

                # Dodajemy 15% marginesu naokoło (padding), żeby sieć widziała dłonie/kontekst
                pad_x = (right - left) * 0.15
                pad_y = (bottom - top) * 0.15

                left = max(0, left - pad_x)
                top = max(0, top - pad_y)
                right = min(img_w, right + pad_x)
                bottom = min(img_h, bottom + pad_y)

                box = (left, top, right, bottom)
                pos_boxes.append(box)

                # Zapisujemy nasz pozytywny przykład (Label 1 [Human] lub 2 [Gun])
                self.crops.append((img_path, box, label))

            # Generowanie Negatywów (tła) dla każdego zdjęcia, na którym są obiekty
            num_negatives = len(pos_boxes) * negatives_per_positive
            neg_attempts = 0
            neg_found = 0

            # Użyjemy tła o podobnej średniej wielkości co broń na tym zdjęciu
            if pos_boxes:
                avg_w = sum(b[2]-b[0] for b in pos_boxes) / len(pos_boxes)
                avg_h = sum(b[3]-b[1] for b in pos_boxes) / len(pos_boxes)
            else:
                avg_w, avg_h = self.crop_size, self.crop_size

            while neg_found < num_negatives and neg_attempts < 50:
                neg_attempts += 1
                # Wylosuj miejsce dla "ślepego" okna
                n_left = random.uniform(0, max(1, img_w - avg_w))
                n_top = random.uniform(0, max(1, img_h - avg_h))
                n_right = min(img_w, n_left + avg_w)
                n_bottom = min(img_h, n_top + avg_h)

                n_box = (n_left, n_top, n_right, n_bottom)

                # Upewnijmy się, że okno z tłem NIE nachodzi na broń (IoU musi być małe)
                iou_ok = True
                for pb in pos_boxes:
                    if bbox_iou(n_box, pb) > 0.1:  # Jeśli nakłada się w > 10%, odrzucamy
                        iou_ok = False
                        break

                if iou_ok:
                    self.crops.append((img_path, n_box, 0)) # Label 0 = Tło
                    neg_found += 1

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, box, label = self.crops[index]

        # Wczytanie obrazu
        image = Image.open(img_path).convert("RGB")
        # MAGIC: Wycięcie interesującego nas kwadratu (broni lub tła)!
        image = image.crop(box)

        if self.transform:
            image = self.transform(image)

        # Zwracamy słownik z etykietą jako liczba całkowita (long)
        # Słownik dlatego, że model ClassificationModel.py oczekuje obiektu attributes[self.attribute]
        target = torch.tensor(label, dtype=torch.long)

        return image, target


class GunmenCropDataModule(L.LightningDataModule):
    """
    Moduł zarządzający paczkami (batchami) dla PyTorch Lightning.
    Dzieli dane na zbiór treningowy i walidacyjny i przygotowuje DataLoadery.
    """
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        batch_size: int = 32,
        crop_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        # Transformacje: zmiana rozmiaru wyciecia na kwadrat 128x128 i format Tensora
        transform = T.Compose([
            T.Resize((self.crop_size, self.crop_size)),
            # TUTAJ MOZESZ DODAC Inne np. T.RandomHorizontalFlip()
            T.ToTensor(),
        ])

        # Tworzymy nasz ogromny zbiór wszystkich łatek (crops)
        full_dataset = GunmenCropDataset(
            dataset_root=self.dataset_root,
            crop_size=self.crop_size,
            transform=transform
        )

        # Dzielenie losowe na trening (80%) i walidacje (20%)
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        self.train_ds, self.val_ds = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
