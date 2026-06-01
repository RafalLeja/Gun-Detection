from __future__ import annotations

from typing import Any

import lightning as L
import torch

DEFAULT_PIXEL_MEAN = (0.485, 0.456, 0.406)
DEFAULT_PIXEL_STD = (0.229, 0.224, 0.225)


class GunmenRfDetrLightningModule(L.LightningModule):
    """LightningModule fine-tuning a HuggingFace RF-DETR object detector.

    It consumes the same batch format produced by ``GunmenYoloDataModule``
    (``img``, ``cls``, ``bboxes``, ``batch_idx`` with normalized cxcywh boxes)
    and converts it into the ``list[dict]`` label format expected by RF-DETR.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "Roboflow/rf-detr-medium",
        learning_rate: float = 1e-4,
        backbone_learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.image_processor = self._build_image_processor(model_name)
        self.detector = self._build_detector(
            model_name=model_name, num_classes=num_classes
        )

        mean = self._processor_stat("image_mean", DEFAULT_PIXEL_MEAN)
        std = self._processor_stat("image_std", DEFAULT_PIXEL_STD)
        self.register_buffer(
            "pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

    @staticmethod
    def _build_image_processor(model_name: str):
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(model_name)

    @staticmethod
    def _build_detector(*, model_name: str, num_classes: int) -> torch.nn.Module:
        from transformers import AutoModelForObjectDetection

        # ignore_mismatched_sizes lets us drop the pretrained COCO (90-class) head
        # and reinitialize a fresh classification head sized for our dataset.
        return AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def _processor_stat(self, attribute: str, fallback: tuple[float, ...]):
        value = getattr(self.image_processor, attribute, None)
        if value is None:
            return fallback
        return value

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _build_labels(
        self, batch: dict[str, torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        cls = batch["cls"].reshape(-1).to(self.device, dtype=torch.long)
        bboxes = batch["bboxes"].reshape(-1, 4).to(self.device, dtype=torch.float32)
        batch_idx = batch["batch_idx"].reshape(-1).to(self.device, dtype=torch.long)
        batch_size = batch["img"].shape[0]

        labels: list[dict[str, torch.Tensor]] = []
        for image_index in range(batch_size):
            mask = batch_idx == image_index
            labels.append(
                {
                    "class_labels": cls[mask],
                    "boxes": bboxes[mask],
                }
            )
        return labels

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        pixel_values = self._normalize(
            batch["img"].to(self.device, non_blocking=True).float()
        )
        labels = self._build_labels(batch)

        outputs = self.detector(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=pixel_values.shape[0],
        )

        loss_dict = getattr(outputs, "loss_dict", None)
        if loss_dict:
            self.log_dict(
                {
                    f"{stage}/{key}": value
                    for key, value in loss_dict.items()
                    if torch.is_tensor(value)
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=pixel_values.shape[0],
            )
        return loss

    def forward(self, pixel_values: torch.Tensor) -> Any:
        return self.detector(pixel_values=pixel_values)

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
            id2label = {index: name for index, name in enumerate(class_names)}
            self.detector.config.id2label = id2label
            self.detector.config.label2id = {
                name: index for index, name in id2label.items()
            }

    def configure_optimizers(self):
        backbone_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []
        for name, parameter in self.detector.named_parameters():
            if not parameter.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(parameter)
            else:
                other_params.append(parameter)

        param_groups: list[dict[str, Any]] = [
            {"params": other_params, "lr": self.hparams["learning_rate"]}
        ]
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.hparams["backbone_learning_rate"],
                }
            )

        return torch.optim.AdamW(
            param_groups,
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )
