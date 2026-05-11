import torch
from src.models.classification_model import ClassificationModel
from src.models.architectures.mlp_backbone import MLPBackbone

def test_classification_model_forward():
    batch_size = 4
    input_shape = (3, 32, 32)
    embed_dim = 64
    num_classes = 10
    attribute = "target_class"

    backbone = MLPBackbone(
        input_shape=input_shape,
        hidden_dims=[128],
        output_dim=embed_dim
    )

    model = ClassificationModel(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        attribute=attribute
    )

    x = torch.randn(batch_size, *input_shape)
    out = model(x)

    assert out.shape == (batch_size, num_classes), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"

def test_classification_model_training_step():
    batch_size = 4
    input_shape = (3, 32, 32)
    embed_dim = 64
    num_classes = 5
    attribute = "target_class"

    backbone = MLPBackbone(
        input_shape=input_shape,
        hidden_dims=[128],
        output_dim=embed_dim
    )

    model = ClassificationModel(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        attribute=attribute
    )

    images = torch.randn(batch_size, *input_shape)
    attributes = {
        "target_class": torch.randint(0, num_classes, (batch_size,))
    }
    batch = (images, attributes)

    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Loss should be a scalar tensor"
