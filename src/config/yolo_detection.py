import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.gunmen_yolo_datamodule import GunmenYoloDataModule
from src.models.gunmen_yolo_lightning import GunmenYoloLightningModule


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 25
    image_size = 640
    batch_size = 8
    num_classes = 2

    data_module = fdl.Config(
        GunmenYoloDataModule,
        dataset_root=None,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4,
        val_split=0.2,
        strict=False,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/loss",
        every_n_epochs=1,
        save_top_k=1,
        mode="min",
    )

    model = fdl.Config(
        GunmenYoloLightningModule,
        num_classes=num_classes,
        model_cfg="yolov8n.yaml",
        pretrained_weights="yolov8n.pt",
        learning_rate=1e-4,
        weight_decay=5e-4,
    )

    return fdl.Config(
        ExperimentConfig,
        "gunmen_yolo_finetune",
        model,
        data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger,
            checkpoints_callback,
            max_epochs,
            callbacks=[],
        ),
    )
