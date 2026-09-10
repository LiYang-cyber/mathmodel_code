import py_compile
from pathlib import Path

EXPECTED = [
    "01_data/load.py",
    "02_eda/eda.py",
    "03_feature/feature_engineering.py",
    "03_feature/pca_svd.py",
    "03_feature/correlation.py",
    "04_model/tabular/baseline.py",
    "04_model/tabular/lightgbm.py",
    "04_model/tabular/xgboost.py",
    "04_model/tabular/catboost.py",
    "04_model/tabular/stacking.py",
    "04_model/tabular/mlp.py",
    "04_model/timeseries/arima.py",
    "04_model/timeseries/mlforecast.py",
    "04_model/timeseries/grey_forecast.py",
    "04_model/timeseries/var.py",
    "04_model/detection/pyod_compare.py",
    "04_model/clustering/cluster.py",
    "04_model/evaluation/entropy_topsis.py",
    "04_model/optimization/particle_swarm.py",
    "05_validation/cross_validation.py",
    "06_explain/shap.py",
    "07_visualization/plot.py",
    "08_report/export_figures.py",
]


def test_requested_template_layout_and_syntax():
    root = Path(__file__).parents[1] / "grad-ml-template"
    for relative in EXPECTED:
        path = root / relative
        assert path.is_file(), f"缺少指定模板: {relative}"
        py_compile.compile(str(path), doraise=True)
