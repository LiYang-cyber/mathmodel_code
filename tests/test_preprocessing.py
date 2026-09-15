import numpy as np
import pandas as pd

from mathmodel.preprocessing import (
    DecimalScaler,
    LogisticScaler,
    RandomSampleImputer,
    SupervisedDiscretizer,
    detect_outliers,
    drop_missing,
    handle_outliers,
    make_encoder,
    make_imputer,
    make_normalizer,
    make_scaler,
    make_unsupervised_discretizer,
)


def test_encoders_handle_unknown_categories():
    train = np.array([["A"], ["B"], ["A"]])
    ordinal = make_encoder("ordinal").fit(train)
    onehot = make_encoder("onehot").fit(train)
    assert ordinal.transform([["C"]])[0, 0] == -1
    assert np.all(onehot.transform([["C"]]) == 0)


def test_missing_value_methods_and_deletion():
    values = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan], [5.0, 8.0]])
    assert np.isclose(make_imputer("mean").fit_transform(values)[1, 0], 3)
    random_a = RandomSampleImputer(7).fit_transform(values)
    random_b = RandomSampleImputer(7).fit_transform(values)
    assert np.array_equal(random_a, random_b)
    assert not np.isnan(random_a).any()
    model_result = make_imputer("model", 7).fit_transform(values)
    assert not np.isnan(model_result).any()
    frame = pd.DataFrame(values, columns=["a", "b"])
    assert len(drop_missing(frame, axis="rows")) == 2


def test_scalers_center_and_ranges():
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    assert np.allclose(make_scaler("center").fit_transform(values).mean(axis=0), 0)
    assert np.allclose(make_scaler("zscore").fit_transform(values).std(axis=0), 1)
    minmax = make_scaler("minmax").fit_transform(values)
    assert np.allclose(minmax.min(axis=0), 0) and np.allclose(minmax.max(axis=0), 1)
    decimal = DecimalScaler().fit(values)
    assert np.abs(decimal.transform(values)).max() < 1
    logistic = LogisticScaler().fit(values)
    transformed = logistic.transform(values)
    assert np.all((transformed > 0) & (transformed < 1))
    assert np.allclose(logistic.inverse_transform(transformed), values)


def test_discretizers_create_bounded_integer_bins():
    values = np.arange(30, dtype=float).reshape(-1, 1)
    labels = np.repeat([0, 1, 2], 10)
    for method in ("uniform", "quantile", "kmeans"):
        result = make_unsupervised_discretizer(method, 4).fit_transform(values)
        assert len(np.unique(result)) <= 4
    for method in ("information_gain", "chimerge", "caim"):
        result = SupervisedDiscretizer(method, max_bins=4, min_samples_leaf=2).fit_transform(values, labels)
        assert np.issubdtype(result.dtype, np.integer)
        assert len(np.unique(result)) <= 4


def test_outlier_detection_and_treatment():
    frame = pd.DataFrame({"x": [0.0] * 20 + [100.0], "y": np.arange(21, dtype=float)})
    for method in ("zscore", "iqr", "mad"):
        result = detect_outliers(frame[["x"]], method=method, threshold=3)
        assert result.row_mask.iloc[-1]
    lof = detect_outliers(frame, method="lof", contamination=.1)
    assert lof.row_mask.sum() >= 1
    iqr = detect_outliers(frame[["x"]], method="iqr")
    clipped = handle_outliers(frame[["x"]], iqr, "clip")
    assert clipped["x"].max() < 100
    assert len(handle_outliers(frame[["x"]], iqr, "remove")) == 20


def test_distribution_transforms_are_finite_and_reversible_for_log():
    values = np.array([[-3.0], [-1.0], [0.0], [10.0]])
    logarithm = make_normalizer("log").fit(values)
    transformed = logarithm.transform(values)
    assert np.isfinite(transformed).all()
    assert np.allclose(logarithm.inverse_transform(transformed), values)
    assert np.isfinite(make_normalizer("yeo-johnson").fit_transform(values)).all()
