from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression


def main() -> None:
    output = Path("data/raw")
    output.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(n_samples=240, n_features=8, n_informative=5, random_state=42)
    classification = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    classification["category"] = np.where(classification["feature_0"] > 0, "A", "B")
    classification["target"] = y
    classification.to_csv(output / "classification.csv", index=False)
    X, y = make_regression(n_samples=240, n_features=8, noise=12, random_state=42)
    regression = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    regression["target"] = y
    regression.to_csv(output / "regression.csv", index=False)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    signal = 20 + .04 * np.arange(180) + 4 * np.sin(np.arange(180) * 2 * np.pi / 7) + rng.normal(0, .8, 180)
    related = 10 + 0.55 * signal + rng.normal(0, 0.5, 180)
    pd.DataFrame({"date": dates, "value": signal, "related_value": related}).to_csv(
        output / "timeseries.csv", index=False
    )


if __name__ == "__main__":
    main()
