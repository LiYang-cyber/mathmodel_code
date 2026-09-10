# 配置说明

所有任务共享 `task`、`data`、`run_name`、`output_dir` 和 `random_state`。Excel 可用 `sheet_name` 指定工作表。

## 表格分类与回归

- `target`：目标列，必填。
- `drop_columns`：编号、姓名等不应进入模型的列。
- `models`：候选模型列表。分类支持 `logistic`、`svm`、`random_forest`、`gradient_boosting`；回归支持 `linear`、`ridge`、`random_forest`、`gradient_boosting`。执行 `pixi install -e full` 后，两者还支持 `lightgbm`、`xgboost`、`catboost`。
- `test_size`：最终测试集比例，默认 0.2。
- `cv_folds`：训练集交叉验证折数，默认 5。

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
