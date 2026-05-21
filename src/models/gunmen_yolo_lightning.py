from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import lightning as L
import torch


class GunmenYoloLightningModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        model_cfg: str = "yolov8n.yaml",
        pretrained_weights: str | None = "yolov8n.pt",
        learning_rate: float = 1e-4,
        weight_decay: float = 5e-4,
        box_loss_gain: float = 7.5,
        cls_loss_gain: float = 0.5,
        dfl_loss_gain: float = 1.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.detector = self._build_detector(
            num_classes=num_classes,
            model_cfg=model_cfg,
            pretrained_weights=pretrained_weights,
            box_loss_gain=box_loss_gain,
            cls_loss_gain=cls_loss_gain,
            dfl_loss_gain=dfl_loss_gain,
        )
        self.criterion = None

    @staticmethod
    def _build_detector(
        *,
        num_classes: int,
        model_cfg: str,
        pretrained_weights: str | None,
        box_loss_gain: float,
        cls_loss_gain: float,
        dfl_loss_gain: float,
    ) -> torch.nn.Module:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        detector = DetectionModel(cfg=model_cfg, ch=3, nc=num_classes, verbose=False)
        if pretrained_weights:
            pretrained_model = YOLO(
                pretrained_weights, task="detect", verbose=False
            ).model
            detector.load(pretrained_model)

        setattr(
            detector,
            "args",
            SimpleNamespace(
                box=box_loss_gain,
                cls=cls_loss_gain,
                dfl=dfl_loss_gain,
            ),
        )
        return detector

    def _build_criterion(self):
        from ultralytics.utils.loss import v8DetectionLoss

        return v8DetectionLoss(self.detector)

    def _get_criterion(self):
        if self.criterion is None:
            self.criterion = self._build_criterion()
        return self.criterion

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prepared_batch: dict[str, torch.Tensor] = {
            "img": batch["img"].to(self.device, non_blocking=True).float(),
            "cls": batch["cls"].to(self.device, non_blocking=True, dtype=torch.long),
            "bboxes": batch["bboxes"].to(self.device, non_blocking=True).float(),
            "batch_idx": batch["batch_idx"].to(
                self.device, non_blocking=True, dtype=torch.long
            ),
        }
        return prepared_batch

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        batch = self._prepare_batch(batch)
        preds = self.detector(batch["img"])
        loss, loss_items = self._get_criterion()(preds, batch)
        scalar_loss = loss[0]

        self.log(
            f"{stage}/loss",
            scalar_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch["img"].shape[0],
        )
        self.log_dict(
            {
                f"{stage}/box_loss": loss_items[0].mean()
                if torch.is_tensor(loss_items[0])
                else loss_items[0],
                f"{stage}/cls_loss": loss_items[1].mean()
                if torch.is_tensor(loss_items[1])
                else loss_items[1],
                f"{stage}/dfl_loss": loss_items[2].mean()
                if torch.is_tensor(loss_items[2])
                else loss_items[2],
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch["img"].shape[0],
        )
        return scalar_loss

    def forward(self, x: torch.Tensor) -> Any:
        return self.detector(x)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_fit_start(self) -> None:
        datamodule = getattr(self.trainer, "datamodule", None)
        class_names = getattr(datamodule, "class_names", None)
        if class_names:
            setattr(
                self.detector,
                "names",
                {index: name for index, name in enumerate(class_names)},
            )
            setattr(self.detector, "nc", len(class_names))

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
