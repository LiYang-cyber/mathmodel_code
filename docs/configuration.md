# 配置说明

所有任务共享 `task`、`data`、`run_name`、`output_dir` 和 `random_state`。Excel 可用 `sheet_name` 指定工作表。

## 表格分类与回归

- `target`：目标列，必填。
- `drop_columns`：编号、姓名等不应进入模型的列。
- `numeric_imputer`：`median`、`mean`、`random` 或 `model`，默认 `median`。
- `numeric_scaler`：`zscore`、`minmax`、`decimal`、`logistic` 或 `center`，默认 `zscore`。
- `categorical_encoder`：`onehot` 或 `ordinal`，默认 `onehot`。
- `models`：本次参与比较的模型名称。仓库内置若干可运行基线，但它们不是模型白名单。
- `custom_models`：通过完整 Python 类路径注册任意 sklearn 兼容估计器；第三方依赖需加入对应 Pixi feature。模型必须实现 `fit()` 和 `predict()`。
- `test_size`：最终测试集比例，默认 0.2。
- `cv_folds`：训练集交叉验证折数，默认 5。

例如增加仓库未预置的 Extra Trees，无需修改核心代码：

```yaml
models: [logistic, extra_trees]
custom_models:
  extra_trees:
    class_path: sklearn.ensemble.ExtraTreesClassifier
    params:
      n_estimators: 500
      min_samples_leaf: 2
      random_state: 42
      n_jobs: -1
```

如果模型不兼容 sklearn 接口，新增适配器或独立流程脚本即可，不应为了迁就本模板而排除更合适的模型。

## 聚类

- `features`：参与聚类的列；省略时使用全部数值列。
- `method`：`kmeans`、`gmm` 或 `dbscan`。
- `n_clusters`：KMeans/GMM 的簇数。
- `eps`、`min_samples`：DBSCAN 参数。

## 异常检测

- `features`：检测特征；省略时使用数值列。
- `contamination`：预期异常比例，或 `auto`。
- `label`：可选的真实异常标签列，要求正常为 0、异常为 1；提供后额外计算 F1 和 ROC-AUC。

## 时间序列

- `time_column`、`target`：时间列与数值目标列。
- `lags`：滞后阶数列表，应结合业务周期设置。
- `seasonal_period`：季节朴素基线周期；日数据的周周期为 7。
- `test_horizon`：按时间顺序留出的验证长度。

仓库内的 S-Mamba、TimePro、STM3 和 MambaSL 使用 `task: deep_timeseries`，详见
[`deep_timeseries.md`](deep_timeseries.md)。需要调用完整上游实验框架时使用
`task: external_timeseries`，详见 [`external_timeseries.md`](external_timeseries.md)。
