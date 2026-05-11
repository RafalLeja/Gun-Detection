import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowClassificationModel(L.LightningModule):
    """
    Model dla podejścia Sliding Window Baseline.
    Przyjmuje krotkę: (images, labels), oblicza Cross Entropy Loss.

    Args:
        backbone:        Sieć wyciągająca cechy z obrazka (B, C, H, W) → (B, embed_dim).
        embed_dim:       Wyjściowa liczba na cechy backbone'a.
        num_classes:     Liczba klas (0 - Tło, 1 - Człowiek, 2 - Broń).
        lr:              Learning rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int = 3,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        # Rozpakowujemy krotkę zgodnie z return z GunmenCropDataset
        images, labels = batch

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
