import torch
from torch import nn
from torchvision.models import resnet18

class ResNetBackbone(nn.Module):
    """
    A ResNet-based backbone for feature extraction.

    Args:
        output_dim: Size of the backbone's output embedding.
        pretrained: Whether to use a pretrained ResNet model.
    """

    def __init__(self, output_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
