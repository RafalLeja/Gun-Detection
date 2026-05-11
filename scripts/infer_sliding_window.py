import random
from pathlib import Path

import click
import fiddle as fdl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image, ImageDraw

from src.utils.config import parse_fiddle_config


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("ckpt_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--window_size", type=int, default=128, help="Rozmiar pojedynczego okna w pikselach")
@click.option("--stride", type=int, default=32, help="O ile pikseli przesuwać okno (krok)")
@click.option("--threshold", type=float, default=0.8, help="Próg pewności sieci (od 0 do 1), powyżej którego rysujemy okno")
@click.option("--iou_threshold", type=float, default=0.1, help="Próg NMS (im mniejszy, tym agresywniej skleja nakładające się na siebie okna)")
def main(config_path, ckpt_path, window_size, stride, threshold, iou_threshold):
    # 1. Zbudowanie czystego modelu wg przepisu Fiddle
    print(f"[*] Budowanie modelu z pliku konfiguracyjnego: {config_path}")
    cfg = parse_fiddle_config(config_path)
    built_cfg = fdl.build(cfg)
    model = built_cfg.model

    # 2. Wczytanie wag wytrenowanych (Checkpointu Lightning)
    print(f"[*] Wczytywanie wag z checkpointu: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # PyTorch Lightning zapisuje wagi w słowniku "state_dict"
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # Wyłączamy tryb uczenia np. modyfikatory Dropout
    
    # Chcemy żeby obliczenia szły na GPU pod Mac (MPS) jeśli to możliwe
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # 3. Wylosowanie zdjęcia
    dataset_dir = Path("sources/Gunmen Dataset/All").resolve()
    images = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.jpeg")) + list(dataset_dir.glob("*.png"))
    if not images:
        print("Nie znaleziono zdjęć w sources/Gunmen Dataset/All!")
        return

    random_img_path = random.choice(images)
    print(f"[*] Wylosowano zdjęcie: {random_img_path.name}")
    original_image = Image.open(random_img_path).convert("RGB")
    width, height = original_image.size
    print(f"[*] Rozmiar obrazu wg (szer. x wys.): {width}x{height}")

    # 4. Inicjalizacja Sliding Window
    transform = T.Compose([
        T.Resize((built_cfg.data_module.crop_size, built_cfg.data_module.crop_size)),
        T.ToTensor()
    ])

    boxes = []
    scores = []
    classes = []

    print("[*] Skanowanie obrazka metodą Sliding Window...")
    
    # Podwójna pętla: y - rzędy w dół, x - kolumny w prawo
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Współrzędne obecnego okna: (left, top, right, bottom)
            box = (x, y, x + window_size, y + window_size)
            
            # Wycina kadr, transformuje go (Tensor + Resize) i dodaje dummy batch_size=1
            crop = original_image.crop(box)
            tensor_crop = transform(crop).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor_crop)
                probs = F.softmax(logits, dim=1)[0] # prawdopodobieństwa z przedziału 0.0 - 1.0
            
            pred_class = torch.argmax(probs).item()
            pred_score = probs[pred_class].item()
            
            # Klasa 0 = Tło, ignorujemy. 1 = Człowiek, 2 = Broń.
            if pred_class in [1, 2] and pred_score >= threshold:
                boxes.append(box)
                scores.append(pred_score)
                classes.append(pred_class)

    print(f"[*] Przed NMS: Znaleziono łącznie {len(boxes)} potencjalnych okien będących obiektami.")

    # 5. Non-Maximum Suppression (NMS) - Filtrowanie nakładających się okien
    if boxes:
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        classes_tensor = torch.tensor(classes)
        
        # ops.batched_nms grupuje NMS po klasach (tzn. okno Człowieka nie wyciszy okna Broni, nawet jak się mocno nakładają)
        keep_indices = ops.batched_nms(boxes_tensor, scores_tensor, classes_tensor, iou_threshold=iou_threshold)
        
        keep_indices = keep_indices.tolist()
        print(f"[*] Po NMS: Zostało {len(keep_indices)} czystych detekcji!")
        
        # 6. Rysowanie detekcji
        draw = ImageDraw.Draw(original_image)
        for idx in keep_indices:
            b = boxes[idx]
            cls = classes[idx]
            scr = scores[idx]
            
            # Kolory: Czerwony (Broń), Niebieski (Człowiek)
            color = "red" if cls == 2 else "blue"
            label_text = "Gun" if cls == 2 else "Human"
            
            # Rysujemy prostokąt i mały tekst pewności
            draw.rectangle(b, outline=color, width=3)
            # Cieniowany/Zaznaczony tekst
            text_pos = (b[0] + 5, max(0, b[1] - 15))
            draw.text(text_pos, f"{label_text}: {scr:.2f}", fill=color)

        output_path = "inference_result.jpg"
        original_image.save(output_path)
        print(f"[+] Zapisano wynik z nałożonymi ramkami do pliku: {output_path}")
    else:
        print("[*] Nie wykryto absolutnie niczego, co przebiłoby podany threshold :(. Zobacz inferencje w plikach.")


if __name__ == "__main__":
    main()
