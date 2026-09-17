from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mathmodel.deep_models import (
    STM3,
    LSTMForecaster,
    MambaForecaster,
    MambaSL,
    SMamba,
    TimePro,
)
from mathmodel.deep_runner import run


@pytest.mark.parametrize(
    ("model", "inputs", "expected"),
    [
        (SMamba(seq_len=8, pred_len=3, d_model=8, d_state=2, e_layers=1), (2, 8, 4), (2, 3, 4)),
        (
            TimePro(
                seq_len=8,
                pred_len=3,
                n_vars=4,
                patch_len=4,
                stride=2,
                d_model=8,
                d_state=2,
                e_layers=1,
            ),
            (2, 8, 4),
            (2, 3, 4),
        ),
        (
            STM3(
                seq_len=8,
                pred_len=3,
                num_nodes=4,
                d_model=8,
                node_emb_dim=4,
                d_state=2,
                num_experts=2,
                scales=(1, 3),
            ),
            (2, 8, 4, 1),
            (2, 3, 4, 1),
        ),
        (MambaSL(n_features=4, n_classes=3, d_model=8, d_state=2), (2, 8, 4), (2, 3)),
        (MambaForecaster(seq_len=8, pred_len=3, n_vars=4, d_model=8, d_state=2,
                         e_layers=1), (2, 8, 4), (2, 3, 4)),
        (LSTMForecaster(seq_len=8, pred_len=3, n_vars=4, hidden_size=8,
                        num_layers=1), (2, 8, 4), (2, 3, 4)),
    ],
)
def test_model_shapes_and_gradients(model, inputs, expected):
    values = torch.randn(*inputs, requires_grad=True)
    output = model(values)
    assert output.shape == expected
    output.square().mean().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_forecasting_runner_writes_unified_outputs(tmp_path: Path):
    rng = np.random.default_rng(7)
    data = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=28),
            "a": rng.normal(size=28),
            "b": rng.normal(size=28),
        }
    ).to_csv(data, index=False)
    output = run(
        {
            "task": "deep_timeseries",
            "model": "s_mamba",
            "data": str(data),
            "time_column": "date",
            "seq_len": 8,
            "pred_len": 2,
            "epochs": 1,
            "batch_size": 8,
            "test_size": 2,
            "model_params": {"d_model": 8, "d_state": 2, "e_layers": 1},
            "output_dir": str(tmp_path / "outputs"),
            "run_name": "smoke",
        }
    )
    for name in ["metrics.csv", "predictions.npz", "model.pt", "run.json", "training_loss.png"]:
        assert (output / name).is_file()
