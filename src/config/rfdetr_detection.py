import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT, Constants as consts
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.gunmen_yolo_datamodule import GunmenYoloDataModule
from src.models.gunmen_rfdetr_lightning import GunmenRfDetrLightningModule


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 25
    image_size = 576
    batch_size = 4
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
        GunmenRfDetrLightningModule,
        num_classes=num_classes,
        model_name=consts.rf_detr_medium,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
        weight_decay=1e-4,
    )

    return fdl.Config(
        ExperimentConfig,
        "gunmen_rfdetr_finetune",
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
