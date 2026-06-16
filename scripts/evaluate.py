from __future__ import annotations

import json
from pathlib import Path

import click
import fiddle as fdl
import lightning as L

from src.utils.config import parse_fiddle_config


@click.command()
@click.argument(
    "config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "ckpt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--split",
    type=click.Choice(["test", "val"]),
    default="test",
    show_default=True,
    help="Na którym podziale liczyć metryki.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Opcjonalna ścieżka do zapisu metryk jako JSON.",
)
def main(config_path: Path, ckpt_path: Path, split: str, output: Path | None) -> None:
    print(f"[*] Budowanie modelu i danych z configu: {config_path}")
    cfg = parse_fiddle_config(str(config_path))
    built_cfg = fdl.build(cfg)

    model: L.LightningModule = built_cfg.model
    data_module: L.LightningDataModule = built_cfg.data_module

    # logger=False - ewaluacja nie tworzy runu WandB; chcemy tylko metryki.
    trainer = L.Trainer(logger=False, enable_checkpointing=False)

    print(f"[*] Wczytywanie wag i ewaluacja na podziale: {split}")
    if split == "test":
        results = trainer.test(
            model=model, datamodule=data_module, ckpt_path=str(ckpt_path)
        )
    else:
        results = trainer.validate(
            model=model, datamodule=data_module, ckpt_path=str(ckpt_path)
        )

    metrics = results[0] if results else {}

    print("\n=== Metryki ===")
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": str(config_path),
            "checkpoint": str(ckpt_path),
            "split": split,
            "metrics": metrics,
        }
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n[+] Zapisano metryki do: {output}")


if __name__ == "__main__":
    main()
