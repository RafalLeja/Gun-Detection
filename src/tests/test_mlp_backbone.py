import torch

from src.models.architectures.mlp_backbone import MLPBackbone


def test_mlp_backbone_forward_shape():
    batch_size = 4
    input_shape = (3, 32, 32)
    hidden_dims = [256, 128]
    output_dim = 64

    model = MLPBackbone(input_shape=input_shape, hidden_dims=hidden_dims, output_dim=output_dim)

    x = torch.randn(batch_size, *input_shape)
    out = model(x)

    assert out.shape == (batch_size, output_dim), f"Expected shape {(batch_size, output_dim)}, got {out.shape}"
