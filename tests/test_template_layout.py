import py_compile
from pathlib import Path

EXPECTED = [
    "01_data/load.py", "01_data/eda.py",
    "02_tabular/baseline.py", "02_tabular/lightgbm.py", "02_tabular/xgboost.py",
    "02_tabular/catboost.py", "02_tabular/stacking.py",
    "03_feature/feature_engineering.py", "03_feature/shap.py",
    "04_timeseries/arima.py", "04_timeseries/mlforecast.py",
    "05_detection/pyod_compare.py", "06_clustering/cluster.py",
    "07_cv/cross_validation.py", "08_visualization/plot.py",
    "09_report/export_figures.py",
]


def test_requested_template_layout_and_syntax():
    root = Path(__file__).parents[1] / "grad-ml-template"
    for relative in EXPECTED:
        path = root / relative
        assert path.is_file(), f"缺少指定模板: {relative}"
        py_compile.compile(str(path), doraise=True)
