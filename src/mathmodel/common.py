from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .io import write_json


def prepare_output(config: dict[str, Any]) -> Path:
    now = datetime.now().astimezone()
    name = config.get("run_name") or f"{config['task']}_{now:%Y%m%d_%H%M%S}"
    output = Path(config.get("output_dir", "outputs")) / name
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    """同时导出预览位图和论文排版常用的两种矢量格式。"""
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_manifest(config: dict[str, Any], output: Path, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
    }
    payload.update(extra or {})
    write_json(payload, output / "run.json")


def inspect_frame(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing": frame.isna().sum().sort_values(ascending=False).to_dict(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
    }
    if target:
        if target not in frame:
            raise ValueError(f"目标列不存在: {target}")
        result["target_counts"] = frame[target].value_counts(dropna=False).head(20).to_dict()
    return result
