import random
from pathlib import Path

import click
import cv2
import fiddle as fdl
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.config.constants import Constants
from src.utils.config import parse_fiddle_config
from scripts.infer_sliding_window import (
    resolve_device,
    resolve_source_image,
    IMAGE_EXTENSIONS,
    BOX_COLORS,
)


def create_fcn_from_model(model):
    """
    Konwertuje wytrenowany ClassificationModel bazujący na ResNet18
    na architekture Fully Convolutional Network (FCN).
    Zamiast końcowego wypluwania jednego wektora z warstwy liniowej,
    wypluwa siatkę aktywacji zachowując rozkład przestrzenny obrazu.
    """
    resnet = model.backbone.resnet

    class FCN(nn.Module):
        def __init__(self, original_model, resnet):
            super().__init__()
            # Wyciągamy warstwy przestrzenne ResNeta pomijając AdaptiveAvgPool
            self.resnet_features = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,  # Stride 2
                resnet.layer1,  # Stride 1 (suma: x2)
                resnet.layer2,  # Stride 2 (suma: x4)
                resnet.layer3,  # Stride 2 (suma: x8)
                resnet.layer4,  # Stride 2 (suma: x16) -> Wait, ResNet layer4 total stride = 32
            )

            # Konwersja warstw liniowych na sploty konwolucyjne 1x1
            in_feat = resnet.fc.in_features
            out_feat = resnet.fc.out_features

            self.conv_fc = nn.Conv2d(in_feat, out_feat, kernel_size=1)
            # .view transformuje macierz wag (Out, In) do formatu dla conv2d (Out, In, H, W)
            self.conv_fc.weight.data = resnet.fc.weight.data.view(
                out_feat, in_feat, 1, 1
            )
            self.conv_fc.bias.data = resnet.fc.bias.data

            self.use_head = original_model.use_head
            if self.use_head:
                in_head = original_model.head.in_features
                out_head = original_model.head.out_features
                self.conv_head = nn.Conv2d(in_head, out_head, kernel_size=1)
                self.conv_head.weight.data = original_model.head.weight.data.view(
                    out_head, in_head, 1, 1
                )
                self.conv_head.bias.data = original_model.head.bias.data

        def forward(self, x):
            x = self.resnet_features(x)
            x = self.conv_fc(x)
            if self.use_head:
                x = self.conv_head(x)
            return x

    return FCN(model, resnet)


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("ckpt_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--source",
    default=None,
    help="Obraz wejściowy lub folder. Losowy z datasetu jeśli pusty.",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Próg ufności dla aktywacji heatmapy.",
)
@click.option(
    "--output",
    default="outputs/fcn_inference.jpg",
    show_default=True,
    help="Gdzie zapisać wynik.",
)
def main(config_path, ckpt_path, source, threshold, output):
    print(f"[*] Budowanie modelu z pliku konfiguracyjnego: {config_path}")
    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)
    model = built_cfg.model

    print(f"[*] Wczytywanie wag z checkpointu: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = resolve_device()

    print("[*] Konwersja modelu do klasy FCN (Fully Convolutional/Heatmap)...")
    fcn_model = create_fcn_from_model(model).to(device)
    fcn_model.eval()

    # Z Constant bierzemy wszystkie nazwy klas - zakładamy że 0=tło, 1..N = klasy
    class_names = list(Constants.classes)
    foreground_classes = list(range(1, len(class_names) + 1))

    source_path = resolve_source_image(source)
    print(f"[*] Wybrano zdjęcie: {source_path.name}")
    original_image = Image.open(source_path).convert("RGB")
    width, height = original_image.size
    print(f"[*] Oryginalny rozmiar obrazu: {width}x{height}")

    # Skalujemy trochę obraz na wejście fcn, by działał szybciej/nie wywalał RAMu, max do 1024.
    max_dim = 1024
    scale = min(max_dim / width, max_dim / height, 1.0)
    new_w, new_h = int(width * scale), int(height * scale)

    # Rozmiar musi być wielokrotnością 32! (Tyle warstw stridów ma ResNet).
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32

    transform = T.Compose(
        [
            T.Resize((new_h, new_w)),
            T.ToTensor(),
        ]
    )

    # Tworzymy tensor wejściowy, unsqueeze dodaje [BatchSize = 1]
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    print("[*] Pushowanie przez sieć splotową bez O(N^2) pętli For Loop...")
    with torch.no_grad():
        logits = fcn_model(input_tensor)  # Wynik np: [1, 17, 32, 24]
        probs = torch.softmax(logits, dim=1)[0]  # Wynik mapy: [17, 32, 24]

    probs_np = probs.cpu().numpy()

    draw = ImageDraw.Draw(original_image)

    # ResNet w layer4 robi generalny downsampling 32x.
    stride = 32
    # Przeliczniki by móc narysować na oryginalnym zdjęciu
    scale_x_back = width / new_w
    scale_y_back = height / new_h

    det_idx = 0
    print("[*] Wyciąganie boksów po analizie heatmapy...")
    for model_class in foreground_classes:
        if model_class >= probs_np.shape[0]:
            print(f"[!] Warning: model wypuścił mniej klas niż oczekiwano.")
            continue

        heatmap = probs_np[model_class]

        # Binaryzacja progowa
        binary_mask = (heatmap >= threshold).astype(np.uint8) * 255

        if np.max(binary_mask) == 0:
            continue

        # Zamiast sliding window, sprawdzamy połączone wyspy pikseli ("Connected Components")
        # to pozwoli boksowi dostosować się do kształtu i rozłożenia wyspy
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # Ignorujemy tło (0)
        for i in range(1, num_labels):
            x_grid, y_grid, w_grid, h_grid, area = stats[i]

            # Mapujemy bounding boxy z gridu warstwy 4 na wejściowy obraz Resneta (mnożąc przez stride)
            # Aplikujemy trochę sztucznego padding, żeby box ładniej łapał sylwetkę (-0.5 grid offsetu)
            box_x1 = max(0, (x_grid - 0.5) * stride)
            box_y1 = max(0, (y_grid - 0.5) * stride)
            box_x2 = min(new_w, (x_grid + w_grid + 0.5) * stride)
            box_y2 = min(new_h, (y_grid + h_grid + 0.5) * stride)

            # Ostateczne wymiary dla original_image.size
            orig_x1 = box_x1 * scale_x_back
            orig_y1 = box_y1 * scale_y_back
            orig_x2 = box_x2 * scale_x_back
            orig_y2 = box_y2 * scale_y_back

            # Rysowanie
            color = BOX_COLORS[model_class % len(BOX_COLORS)]
            # class_names ma 0 jako pierwsza wartosc, modele mają klasy offsetnięte backgroundem
            label_text = (
                class_names[model_class - 1]
                if (model_class - 1) < len(class_names)
                else f"K{model_class}"
            )

            # Score dla statystyki (np avg confidence on region)
            score = np.mean(heatmap[labels == i])

            draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline=color, width=4)
            draw.text(
                (orig_x1 + 5, max(0, orig_y1 - 15)),
                f"{label_text}: {score:.2f}",
                fill=color,
            )

            det_idx += 1

    if det_idx == 0:
        print("[*] Żadna klasa nie przekroczyła threshold na heatmapie :(")
    else:
        print(f"[*] Znaleziono obiekty (clustry na heatmapie): {det_idx}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_image.save(output_path)
    print(f"[+] Zapisano rezultat obrazu do: {output_path}")


if __name__ == "__main__":
    main()
