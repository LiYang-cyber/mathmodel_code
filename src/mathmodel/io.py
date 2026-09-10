from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if "task" not in config:
        raise ValueError("配置缺少必填字段 task")
    return config


def load_table(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"不支持的数据格式: {suffix}；请使用 CSV、Excel 或 Parquet")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

