import random
from pathlib import Path

import click
import fiddle as fdl
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import torchvision.transforms as T
from PIL import Image, ImageDraw

from src.config.constants import Constants
from src.utils.config import parse_fiddle_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BOX_COLORS = ["blue", "red", "green", "orange", "purple", "cyan"]


def parse_float_list(raw: str) -> list[float]:
    return [float(part) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_source_image(source: Path | None) -> Path:
    dataset_dir = Path("sources/Gunmen Dataset/All").resolve()

    if source is not None:
        source = Path(source)
        if source.is_dir():
            dataset_dir = source
        else:
            return source

    images = [
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"Nie znaleziono zdjęć w {dataset_dir}")
    return random.choice(images)


def generate_windows(
    width: int,
    height: int,
    window_sizes: list[int],
    aspect_ratios: list[float],
    stride_ratio: float,
) -> list[tuple[int, int, int, int]]:
    """Tworzy okna o różnych rozmiarach i proporcjach (szer./wys.).

    Dla bazowego rozmiaru ``s`` i proporcji ``a`` okno ma wysokość ``s`` i
    szerokość ``s * a``, dzięki czemu nie są to już tylko kwadraty.
    """
    windows: list[tuple[int, int, int, int]] = []

    for size in window_sizes:
        for aspect_ratio in aspect_ratios:
            win_h = min(size, height)
            win_w = min(int(round(size * aspect_ratio)), width)
            if win_w < 8 or win_h < 8:
                continue

            step_x = max(1, int(win_w * stride_ratio))
            step_y = max(1, int(win_h * stride_ratio))

            xs = list(range(0, max(1, width - win_w + 1), step_x))
            ys = list(range(0, max(1, height - win_h + 1), step_y))

            # Dorzuć okna dokładnie przy prawej/dolnej krawędzi, żeby nie gubić brzegów.
            if xs and xs[-1] != width - win_w:
                xs.append(max(0, width - win_w))
            if ys and ys[-1] != height - win_h:
                ys.append(max(0, height - win_h))

            for y in ys:
                for x in xs:
                    windows.append((x, y, x + win_w, y + win_h))

    # Usuń duplikaty (różne konfiguracje mogą wygenerować to samo okno).
    return list(dict.fromkeys(windows))


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("ckpt_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    default=None,
    help="Plik obrazu lub folder. Jeśli pominięty, losowy obraz z datasetu.",
)
@click.option(
    "--window-sizes",
    type=str,
    default="96,128,192",
    show_default=True,
    help="Lista bazowych rozmiarów okien w pikselach (po przecinku).",
)
@click.option(
    "--aspect-ratios",
    type=str,
    default="1.0,0.6,1.7",
    show_default=True,
    help="Lista proporcji szer./wys. (1.0=kwadrat, <1 pionowe, >1 poziome).",
)
@click.option(
    "--stride-ratio",
    type=float,
    default=0.25,
    show_default=True,
    help="Krok okna jako ułamek jego rozmiaru (skaluje się z oknem).",
)
@click.option(
    "--threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Próg pewności, powyżej którego rysujemy okno.",
)
@click.option(
    "--iou_threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Próg NMS (im mniejszy, tym agresywniej skleja nakładające się okna).",
)
@click.option(
    "--batch-size",
    type=int,
    default=64,
    show_default=True,
    help="Ile wycinków przepuszczać przez sieć naraz.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("outputs/sliding_window_inference.jpg"),
    show_default=True,
    help="Gdzie zapisać wynik.",
)
def main(
    config_path,
    ckpt_path,
    source,
    window_sizes,
    aspect_ratios,
    stride_ratio,
    threshold,
    iou_threshold,
    batch_size,
    output,
):
    print(f"[*] Budowanie modelu z pliku konfiguracyjnego: {config_path}")
    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)
    model = built_cfg.model

    print(f"[*] Wczytywanie wag z checkpointu: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = resolve_device()
    model = model.to(device)

    window_size_list = parse_int_list(window_sizes)
    aspect_ratio_list = parse_float_list(aspect_ratios)

    # Model klasy: 0 = tło, 1..N = klasy z Constants.classes.
    class_names = list(Constants.classes)
    foreground_classes = list(range(1, len(class_names) + 1))

    def class_name(model_class: int) -> str:
        index = model_class - 1
        if 0 <= index < len(class_names):
            return class_names[index]
        return f"class_{model_class}"

    source_path = resolve_source_image(source)
    print(f"[*] Wybrano zdjęcie: {source_path.name}")
    original_image = Image.open(source_path).convert("RGB")
    width, height = original_image.size
    print(f"[*] Rozmiar obrazu (szer. x wys.): {width}x{height}")

    crop_size = built_cfg.data_module.crop_size
    transform = T.Compose(
        [
            T.Resize((crop_size, crop_size)),
            T.ToTensor(),
        ]
    )

    windows = generate_windows(
        width=width,
        height=height,
        window_sizes=window_size_list,
        aspect_ratios=aspect_ratio_list,
        stride_ratio=stride_ratio,
    )
    print(
        f"[*] Wygenerowano {len(windows)} okien "
        f"({len(window_size_list)} rozmiarów x {len(aspect_ratio_list)} proporcji)."
    )

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    classes: list[int] = []

    print("[*] Skanowanie obrazka metodą Sliding Window (multi-scale)...")

    def process_batch(batch_windows: list[tuple[int, int, int, int]]) -> None:
        if not batch_windows:
            return
        tensors = torch.stack(
            [transform(original_image.crop(box)) for box in batch_windows]
        ).to(device)

        with torch.no_grad():
            logits = model(tensors)
            probs = F.softmax(logits, dim=1)

        for box, prob in zip(batch_windows, probs):
            # Dopuszczamy WIELE klas na jedno okno: każda klasa pierwszoplanowa
            # z prawdopodobieństwem >= threshold dostaje własny bounding box.
            for model_class in foreground_classes:
                score = prob[model_class].item()
                if score >= threshold:
                    boxes.append(box)
                    scores.append(score)
                    classes.append(model_class)

    batch_buffer: list[tuple[int, int, int, int]] = []
    for box in windows:
        batch_buffer.append(box)
        if len(batch_buffer) >= batch_size:
            process_batch(batch_buffer)
            batch_buffer = []
    process_batch(batch_buffer)

    print(f"[*] Przed NMS: {len(boxes)} potencjalnych detekcji.")

    if not boxes:
        print("[*] Nie wykryto niczego powyżej progu :(.")
        return

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    classes_tensor = torch.tensor(classes)

    # batched_nms działa per-klasa, więc box "person" i "gun" w tym samym
    # miejscu NIE tłumią się nawzajem - mogą współistnieć.
    keep_indices = ops.batched_nms(
        boxes_tensor, scores_tensor, classes_tensor, iou_threshold=iou_threshold
    ).tolist()
    print(f"[*] Po NMS: {len(keep_indices)} czystych detekcji.")

    draw = ImageDraw.Draw(original_image)
    for idx in keep_indices:
        box = boxes[idx]
        model_class = classes[idx]
        score = scores[idx]

        color = BOX_COLORS[model_class % len(BOX_COLORS)]
        label_text = class_name(model_class)

        draw.rectangle(box, outline=color, width=3)
        text_pos = (box[0] + 5, max(0, box[1] - 15))
        draw.text(text_pos, f"{label_text}: {score:.2f}", fill=color)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    original_image.save(output)
    print(f"[+] Zapisano wynik do: {output}")


if __name__ == "__main__":
    main()
