# grad-ml-template

研究生数学建模比赛用的机器学习代码架子。文件按做题顺序存放；目录里的算法只负责给出可运行基线，不限定 Agent 后面采用哪些模型。

## 开始使用

环境由 Pixi 管理：

```powershell
pixi install
pixi run demo-data
pixi run template-smoke
```

`template-smoke` 会执行 EDA、表格分类基线和聚类样例。检查代码及测试用：

```powershell
pixi run check
```

LightGBM、XGBoost、CatBoost、SHAP、ARIMA、MLForecast 和 PyOD 放在 `full` 环境中，避免基础环境安装过慢：

```powershell
pixi install -e full
pixi run -e full python grad-ml-template/04_model/tabular/lightgbm.py -c configs/classification.yaml
```

只做 ARIMA、VAR 或 MLForecast 时可安装较小的时序环境：

```powershell
pixi install -e forecast
pixi run -e forecast python grad-ml-template/04_model/timeseries/var.py data/raw/timeseries.csv --time date --targets value related_value
```

## 文件按做题顺序放置

```text
grad-ml-template/
├── 01_data/
│   └── load.py                     读取 CSV、Excel、Parquet
├── 02_eda/
│   └── eda.py                      缺失、重复、类型和描述统计
├── 03_feature/
│   ├── feature_engineering.py      日期、对数和交互特征
│   ├── pca_svd.py                  PCA/SVD 降维
│   └── correlation.py              相关系数和灰色关联
├── 04_model/
│   ├── tabular/
│   │   ├── baseline.py             分类/回归基线比较
│   │   ├── lightgbm.py
│   │   ├── xgboost.py
│   │   ├── catboost.py
│   │   ├── stacking.py
│   │   └── mlp.py                  BP/MLP 基线
│   ├── timeseries/
│   │   ├── arima.py
│   │   ├── mlforecast.py
│   │   ├── grey_forecast.py        GM(1,1)
│   │   └── var.py                  向量自回归
│   ├── detection/
│   │   └── pyod_compare.py
│   ├── clustering/
│   │   └── cluster.py
│   ├── evaluation/
│   │   └── entropy_topsis.py
│   └── optimization/
│       └── particle_swarm.py
├── 05_validation/
│   └── cross_validation.py         随机、分层、时序切分
├── 06_explain/
│   └── shap.py                     特征贡献图
├── 07_visualization/
│   └── plot.py                     预测图和残差图
└── 08_report/
    └── export_figures.py           汇总论文图片
```

公共代码位于 `src/mathmodel/`。各阶段脚本可以独立运行，也可以导入其中的函数。原始附件放进 `data/raw/`，清洗后的数据写入 `data/processed/`，实验产物统一写入 `outputs/<run_name>/`。

## 建议的做题顺序

拿到附件后，先运行 `01_data/load.py` 和 `02_eda/eda.py`。这一步要弄清列类型、缺失值、重复记录、目标分布，以及预测时不可获得的泄漏字段。

特征写在 `03_feature/`。会从数据中学习参数的操作，比如填补、缩放、编码和特征选择，应放进 sklearn Pipeline，并且只在训练折上拟合。

随后判断问题属于哪种数据结构，再进入 `04_model/` 的相应子目录。表格、时间序列、异常检测和聚类不能共用同一种切分办法；验证方案放在 `05_validation/`，需要分组切分、滚动回测或嵌套交叉验证时，直接增加实现。

确定候选方案后再做解释和画图。`06_explain/` 存模型解释，`07_visualization/` 存误差诊断与论文图，`08_report/` 收集最终图片。

## 模型可以随题目增加

内置模型只是样例。模型选择应取决于目标、数据生成方式、样本量、类别不平衡、时间或空间关系、外部约束以及比赛评价指标。

兼容 sklearn `fit()`、`predict()` 接口的模型可直接在 YAML 中注册。例如加入 Extra Trees：

```yaml
task: classification
data: data/raw/classification.csv
target: target
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

然后运行：

```powershell
pixi run mathmodel run -c path/to/config.yaml
```

深度学习、图模型、贝叶斯模型、空间统计模型等接口不同的实现，可以放进 `04_model/` 新建的任务子目录，或在 `src/mathmodel/` 编写适配器。不要为了套用已有脚本而放弃更合适的方法。

## 输出约定

一次受监督学习实验通常产生：

- `metrics.csv`：交叉验证指标及测试集指标；
- `predictions.csv`：逐样本真实值、预测值和可用的预测概率；
- `best_model.joblib`：预处理与模型组成的 Pipeline；
- `run.json`：配置、Python 版本、平台、时间和数据量；
- 同名 `.png`、`.svg`、`.pdf`：300 dpi 预览图及两种矢量图。

聚类、异常检测和时序任务使用各自的结果表，但仍保留指标、模型、运行信息及三种图片格式。`outputs/` 默认不提交到 Git。

## 常用命令

```powershell
# 查看数据前五行
pixi run python grad-ml-template/01_data/load.py data/raw/classification.csv

# 生成 EDA 文件
pixi run python grad-ml-template/02_eda/eda.py data/raw/classification.csv --target target

# 比较表格基线
pixi run python grad-ml-template/04_model/tabular/baseline.py -c configs/classification.yaml

# 预览时序交叉验证切分
pixi run python grad-ml-template/05_validation/cross_validation.py --task timeseries --samples 100 --folds 5

# 汇总输出目录中的图片
pixi run python grad-ml-template/08_report/export_figures.py --source outputs --output outputs/report_figures
```

配置字段见 [`docs/configuration.md`](docs/configuration.md)，比赛操作记录见 [`docs/competition_workflow.md`](docs/competition_workflow.md)。历年题目算法清单与本轮取舍见 [`docs/huawei_review_algorithms.md`](docs/huawei_review_algorithms.md)，后续 Agent 的选型规则写在 [`AGENTS.md`](AGENTS.md)。
