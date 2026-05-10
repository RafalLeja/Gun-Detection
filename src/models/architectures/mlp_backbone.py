import math

import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """
    Args:
        input_shape: Shape of a single input sample, e.g. (784,) or (3, 32, 32).
        hidden_dims: Sizes of the hidden layers (e.g. [256, 128]).
        output_dim:  Size of the backbone's output embedding.
        dropout:     Dropout probability applied after every hidden activation.
    """

    def __init__(
        self,
        input_shape: tuple[int, ...],
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        input_dim = math.prod(input_shape)
        self.network = self._create_network(
            input_size=input_dim,
            hidden_sizes=hidden_dims,
            output_size=output_dim,
            dropout_rate=dropout,
        )

    def _create_network(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        dropout_rate: float,
    ):
        layers = []
        current_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))
