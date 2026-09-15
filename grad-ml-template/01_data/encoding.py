"""类别特征数字编码与 One-Hot 编码模板。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import repo_path

from mathmodel.io import load_table
from mathmodel.preprocessing import make_encoder


def encode_columns(data: str, columns: list[str], method: str, output: str) -> Path:
    frame = load_table(repo_path(data))
    encoder = make_encoder(method)
    encoded = encoder.fit_transform(frame[columns])
    names = encoder.get_feature_names_out(columns)
    result = pd.concat([
        frame.drop(columns=columns).reset_index(drop=True),
        pd.DataFrame(encoded, columns=names, index=frame.index).reset_index(drop=True),
    ], axis=1)
    out = repo_path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    joblib.dump(encoder, out.with_suffix(".joblib"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="类别特征编码")
    parser.add_argument("data")
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--method", choices=["ordinal", "onehot"], default="onehot")
    parser.add_argument("--output", default="data/processed/encoded.csv")
    args = parser.parse_args()
    print(encode_columns(args.data, args.columns, args.method, args.output).resolve())


if __name__ == "__main__":
    main()

