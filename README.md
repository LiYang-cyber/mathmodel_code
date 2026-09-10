# 数学建模机器学习模板库

面向研究生数学建模竞赛中高频的表格分类/回归、时间序列预测、异常检测、聚类和模型解释任务。模板强调统一入口、可复现实验、模型横向比较与可直接用于论文的结果导出。

## 30 秒开始

```powershell
pixi install
pixi run demo-data
pixi run mathmodel run --config configs/classification.yaml
```

结果保存在 `outputs/<run_name>/`：

- `metrics.csv`：模型指标与排名
- `predictions.csv`：测试集真实值、预测值（分类含概率）
- `best_model.joblib`：含预处理步骤的完整最佳模型
- `run.json`：数据、参数、随机种子和运行环境
- `*.png`：用于预览或 Word 的 300 dpi 位图
- `*.svg` / `*.pdf`：用于论文排版和后期编辑的矢量图

## 支持任务

| 任务 | `task` | 默认候选模型 | 推荐指标 |
|---|---|---|---|
| 分类/识别/故障诊断 | `classification` | Logistic、SVM、RandomForest、GradientBoosting | F1、ROC-AUC |
| 指标/风险预测 | `regression` | Linear、Ridge、RandomForest、GradientBoosting | RMSE、R² |
| 聚类/分群 | `clustering` | KMeans、GMM、DBSCAN | 轮廓系数 |
| 异常检测 | `anomaly` | IsolationForest、LOF | 已标注时用 F1/AUC |
| 时间序列 | `timeseries` | Naive、SeasonalNaive、线性滞后模型 | RMSE、MAE |

需要全部增强模型时执行 `pixi install -e full`，再用 `pixi run -e full ...` 运行。默认环境保持轻量；可选库未安装时会清晰提示，不影响核心模板运行。

## 常用命令

```powershell
# 查看数据概况
pixi run mathmodel inspect --data data/raw/classification.csv --target target

# 覆盖配置中的数据或目标列
pixi run mathmodel run -c configs/classification.yaml --data path/to/data.xlsx --target 标签列

# 时序预测
pixi run mathmodel run -c configs/timeseries.yaml

# 一次跑完规范检查、测试或五类演示
pixi run check
pixi run demo
```

配置字段说明见 [`docs/configuration.md`](docs/configuration.md)，比赛工作流见 [`docs/competition_workflow.md`](docs/competition_workflow.md)。视觉任务由于数据目录组织和硬件差异较大，提供选型指南而不强行包装训练器，见 [`docs/vision.md`](docs/vision.md)。

## 设计原则

1. 切分数据后再拟合缺失值、编码和标准化，避免数据泄漏。
2. 所有随机过程由 `random_state` 控制。
3. 最佳模型依据交叉验证选择，测试集只用于最终报告。
4. 类别列自动 one-hot；数值列自动中位数填补。
5. 每张结果图同时导出 PNG、SVG 和 PDF，原始数据和运行产物默认不进入 Git。

## 目录

```text
grad-ml-template/        比赛时直接复制和修改的模板
├── 01_data/             数据加载与 EDA
├── 02_tabular/          baseline、LightGBM、XGBoost、CatBoost、Stacking
├── 03_feature/          特征工程与 SHAP
├── 04_timeseries/       ARIMA 与 MLForecast
├── 05_detection/        PyOD 模型比较
├── 06_clustering/       KMeans/GMM/DBSCAN
├── 07_cv/               分类、回归、时序交叉验证
├── 08_visualization/    预测和残差图
└── 09_report/           论文图片汇总导出
configs/                 可直接修改的任务配置
data/raw/                原始附件
data/processed/          清洗后数据
src/mathmodel/           核心代码
examples/                演示数据生成器
tests/                   回归测试
outputs/                 每次实验结果
```

每个 `grad-ml-template` 文件既可作为代码片段导入，也可独立执行。例如：

```powershell
pixi run python grad-ml-template/01_data/eda.py data/raw/classification.csv --target target
pixi run python grad-ml-template/02_tabular/baseline.py -c configs/classification.yaml
pixi run python grad-ml-template/06_clustering/cluster.py data/raw/classification.csv --clusters 3
```
