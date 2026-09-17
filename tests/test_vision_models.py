import pytest

torch = pytest.importorskip("torch")

from mathmodel.neural_models import build_neural_model
from mathmodel.vision_models import build_vision_model


@pytest.mark.parametrize(
    ("name", "shape", "params"),
    [
        ("lenet", (2, 1, 28, 28), {"in_channels": 1, "num_classes": 4}),
        ("nin", (2, 3, 96, 96), {"num_classes": 4}),
        ("googlenet", (2, 3, 96, 96), {"num_classes": 4}),
        ("resnet18", (2, 3, 64, 64), {"num_classes": 4}),
        ("densenet", (2, 3, 64, 64), {"num_classes": 4, "growth_rate": 8,
                                      "block_layers": (2, 2, 2, 2)}),
    ],
)
def test_classic_vision_models_forward_and_backward(name, shape, params):
    model = build_vision_model(name, params)
    inputs = torch.randn(*shape, requires_grad=True)
    output = model(inputs)
    assert output.shape == (shape[0], 4)
    output.mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_alexnet_and_vgg_build_without_allocating_training_batch():
    assert build_vision_model("alexnet", {"num_classes": 3}).__class__.__name__ == "AlexNet"
    assert build_vision_model("vgg", {"num_classes": 3}).__class__.__name__ == "VGG"


@pytest.mark.parametrize("name", ["rnn", "gru", "lstm", "transformer"])
def test_general_sequence_models(name):
    model = build_neural_model(name, {"input_size": 5, "num_classes": 3, "hidden_size": 8}
                               if name != "transformer" else
                               {"input_size": 5, "num_classes": 3, "d_model": 8,
                                "n_heads": 2, "layers": 1, "feedforward": 16})
    assert model(torch.randn(2, 7, 5)).shape == (2, 3)
