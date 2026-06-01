import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.models import ResNet18_Weights

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.gunmen_crop_dataset import GunmenCropDataModule
from src.models.architectures.resnet_backbone import ResNetBackbone
from src.models.classification_model import ClassificationModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 10
    output_dim = 128
    image_size = 224
    num_classes = 17

    architecture = fdl.Config(
        ResNetBackbone,
        # input_shape=(3, image_size, image_size),  # Channels x Height x Width
        # hidden_dims=[512, 256],
        output_dim=output_dim,
        pretrained=True,
        # dropout=0.1,
    )

    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    data_module = fdl.Config(
        GunmenCropDataModule,
        dataset_root=None,
        batch_size=64,
        crop_size=image_size,
        transforms=transforms,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/acc",
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
    )

    model = fdl.Config(
        ClassificationModel,
        architecture,
        embed_dim=output_dim,
        num_classes=num_classes,
        attribute="label",
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        "sliding_window_baseline_3classes",
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
