import matplotlib.pyplot as plt
import numpy as np
import pytest

from mathmodel.visualization import TrainingBoard, plot, show_heatmaps, show_images
from mathmodel.workflow import ComponentRegistry, Workflow, WorkflowStep, default_registry


def test_workflow_composes_task_neutral_steps():
    workflow = Workflow([
        WorkflowStep("prepare", lambda context: {"features": context["raw"] * 2},
                     requires=frozenset({"raw"}), provides=frozenset({"features"})),
        WorkflowStep("model", lambda context: {"prediction": context["features"] + 1},
                     requires=frozenset({"features"}), provides=frozenset({"prediction"})),
    ])
    result = workflow.run({"raw": np.array([1, 2])})
    assert result["prediction"].tolist() == [3, 5]


def test_workflow_validates_contracts_and_registry():
    with pytest.raises(ValueError, match="缺少输入"):
        Workflow([WorkflowStep("fit", lambda _: {}, requires=frozenset({"features"}))]).run()
    registry = ComponentRegistry()
    registry.register("model", "constant", lambda value=1: {"value": value})
    assert registry.create("model", "constant", value=3) == {"value": 3}
    assert registry.names("model") == ("constant",)
    assert "random_forest" in default_registry().names("classification")


def test_d2l_style_plotting_helpers_return_figures():
    figure, axes = plot(np.arange(5), np.arange(5) ** 2, xlabel="x", ylabel="y")
    assert axes.get_xlabel() == "x" and len(axes.lines) == 1
    plt.close(figure)
    figure, axes = show_images([np.zeros((4, 4)), np.ones((4, 4))], 1, 2)
    assert axes.shape == (1, 2)
    plt.close(figure)
    figure, axes = show_heatmaps(np.ones((1, 2, 3, 3)), titles=["a", "b"])
    assert axes.shape == (1, 2)
    plt.close(figure)
    board = TrainingBoard()
    board.add("loss", 1, 2)
    board.add("loss", 2, 1)
    figure, axes = board.figure()
    assert len(axes.lines) == 1
    plt.close(figure)
