"""Training pipeline for the portable deep time-series models."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from ..common import prepare_output, save_figure, save_manifest
from ..io import load_table
from .models import build_deep_timeseries_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前 PyTorch 环境无法使用 CUDA")
    return device


def _forecast_data(config: dict[str, Any]) -> tuple[Tensor, Tensor, dict[str, Any]]:
    frame = load_table(config["data"], config.get("sheet_name", 0))
    time_column = config.get("time_column")
    if time_column:
        frame[time_column] = pd.to_datetime(frame[time_column])
        frame = frame.sort_values(time_column)
    columns = config.get("feature_columns")
    if columns is None:
        columns = [column for column in frame.select_dtypes(include="number").columns]
    if not columns:
        raise ValueError("没有可用于预测的数值列")
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("深度时序模型输入不能包含缺失值或无穷值")
    seq_len = int(config.get("seq_len", 96))
    pred_len = int(config.get("pred_len", 24))
    count = len(values) - seq_len - pred_len + 1
    if count < 2:
        raise ValueError("数据长度不足以构造至少两个滑动窗口")
    x = np.stack([values[i : i + seq_len] for i in range(count)])
    y = np.stack([values[i + seq_len : i + seq_len + pred_len] for i in range(count)])
    metadata = {
        "seq_len": seq_len,
        "pred_len": pred_len,
        "n_vars": len(columns),
        "columns": columns,
    }
    return torch.from_numpy(x), torch.from_numpy(y), metadata


def _npz_data(
    config: dict[str, Any], classification: bool
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    archive = np.load(config["data"], allow_pickle=False)
    x_key, y_key = config.get("x_key", "X"), config.get("y_key", "y")
    if x_key not in archive or y_key not in archive:
        raise ValueError(f"NPZ 必须包含 {x_key!r} 和 {y_key!r}")
    x = np.asarray(archive[x_key], dtype=np.float32)
    y = np.asarray(archive[y_key], dtype=np.int64 if classification else np.float32)
    if not np.isfinite(x).all() or (not classification and not np.isfinite(y).all()):
        raise ValueError("NPZ 输入不能包含缺失值或无穷值")
    if classification:
        if x.ndim != 3 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("MambaSL 需要 X=[samples,length,features]、y=[samples]")
        classes, encoded = np.unique(y, return_inverse=True)
        return (
            torch.from_numpy(x),
            torch.from_numpy(encoded),
            {"n_features": x.shape[2], "n_classes": len(classes), "classes": classes.tolist()},
        )
    if x.ndim != 4 or y.ndim != 4 or len(x) != len(y):
        raise ValueError("STM3 需要 X=[samples,length,nodes,features] 和四维 y")
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        {
            "seq_len": x.shape[1],
            "num_nodes": x.shape[2],
            "input_dim": x.shape[3],
            "pred_len": y.shape[1],
            "output_dim": y.shape[3],
        },
    )


def _split(
    x: Tensor, y: Tensor, classification: bool, test_size: float, seed: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if classification:
        indices = np.arange(len(x))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y.numpy()
        )
        return x[train_idx], x[test_idx], y[train_idx], y[test_idx]
    count = int(test_size) if isinstance(test_size, int) else max(1, round(len(x) * test_size))
    if count < 1 or count >= len(x):
        raise ValueError("test_size 必须保留至少一个训练样本和一个测试样本")
    return x[:-count], x[-count:], y[:-count], y[-count:]


def _fit(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    config: dict[str, Any],
    device: torch.device,
    classification: bool,
) -> list[float]:
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=int(config.get("batch_size", 32)),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    criterion: nn.Module = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    losses = []
    model.train()
    for _ in range(int(config.get("epochs", 20))):
        total = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
        losses.append(total / len(train_x))
    return losses


def run(config: dict[str, Any]) -> Path:
    model_name = str(config["model"]).lower()
    classification = model_name == "mambasl"
    spatial = model_name == "stm3"
    if classification or spatial:
        x, y, inferred = _npz_data(config, classification)
    else:
        x, y, inferred = _forecast_data(config)
    seed = int(config.get("random_state", 42))
    _set_seed(seed)
    test_size = config.get("test_size", 0.2)
    train_x, test_x, train_y, test_y = _split(x, y, classification, test_size, seed)
    params = {**inferred, **dict(config.get("model_params", {}))}
    params.pop("columns", None)
    params.pop("classes", None)
    if model_name == "s_mamba":
        params.pop("n_vars", None)
    model = build_deep_timeseries_model(model_name, params)
    device = _device(str(config.get("device", "auto")))
    model.to(device)
    losses = _fit(model, train_x, train_y, config, device, classification)
    model.eval()
    with torch.no_grad():
        raw_prediction = model(test_x.to(device)).cpu()
    output = prepare_output(config)
    if classification:
        prediction = raw_prediction.argmax(dim=1).numpy()
        actual = test_y.numpy()
        metrics = {"accuracy": accuracy_score(actual, prediction)}
        pd.DataFrame({"actual": actual, "predicted": prediction}).to_csv(
            output / "predictions.csv", index=False
        )
    else:
        prediction = raw_prediction.numpy()
        actual = test_y.numpy()
        metrics = {
            "rmse": mean_squared_error(actual.reshape(-1), prediction.reshape(-1)) ** 0.5,
            "mae": mean_absolute_error(actual.reshape(-1), prediction.reshape(-1)),
        }
        np.savez_compressed(output / "predictions.npz", actual=actual, predicted=prediction)
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(output / "metrics.csv", index=False)
    torch.save(
        {"state_dict": model.state_dict(), "model": model_name, "params": params},
        output / "model.pt",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set(xlabel="Epoch", ylabel="Training loss", title=f"{model_name} training")
    save_figure(fig, output, "training_loss")
    save_manifest(
        config,
        output,
        {
            "model": model_name,
            "model_params": params,
            "device": str(device),
            "test_samples": len(test_x),
        },
    )
    return output
