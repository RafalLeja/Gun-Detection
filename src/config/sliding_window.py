import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.gunmen_crop_dataset import GunmenCropDataModule
from src.models.architectures.mlp_backbone import MLPBackbone
from src.models.classification_model import ClassificationModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 10
    embed_dim = 128
    image_size = 128

    # Opcja 1: Używamy MLP jako najprostszego, bardzo szybkiego baseline'u
    # Ponieważ MLP spłaszcza wejście globalne pod maską, podajemy oryginalny shape
    architecture = fdl.Config(
        MLPBackbone,
        input_shape=(3, image_size, image_size),  # Channels x Height x Width
        hidden_dims=[512, 256],
        output_dim=embed_dim,
        dropout=0.1,
    )

    # Opcja 2 (polecana po sprawdzeniu działania MLP): Możesz podmienić na swój ConvNet
    """
    architecture = fdl.Config(
        ConvNetBackbone,
        input_shape=(3, image_size, image_size),
        channel_dims=[32, 64, 128],
        output_dim=embed_dim,
        dropout=0.,
    )
    """

    data_module = fdl.Config(
        GunmenCropDataModule,
        dataset_root=None, # Znajdzie ze ścieżki defaultowej
        batch_size=64,
        crop_size=image_size,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/acc",  # Możesz też monitorować "val/loss"
        every_n_epochs=1,
        save_top_k=1,
        mode="max"          # Dla accuracy chcemy najwyższe, jak zmienisz na loss to mode="min"
    )

    model = fdl.Config(
        ClassificationModel,
        architecture,
        embed_dim=embed_dim,
        num_classes=3,  # 0 - Tło, 1 - Człowiek, 2 - Broń
        attribute="label", # To wyciągnie wartość ze słownika {"label": target}
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
        )
    )
