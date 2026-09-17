# mathmodel-code Agent 指南

本文件是项目结构、能力、约定和工作流的唯一说明来源。不要新增独立说明文档；代码行为变化时同步修改本文件。

## 项目定位

这是面向数学建模与时序分析的 Python 工具箱。公共实现全部直接位于 `src/mathmodel/`，不建立按建模步骤、算法类别或兼容版本划分的深层目录。配置在 `configs/`，可执行示例数据生成器在 `examples/`，测试在 `tests/`，运行产物写入 `outputs/`。

项目只维护当前 API，不保留旧目录、旧导入路径、转发模块或外部工程兼容层。破坏性 API 调整应同时更新仓库内全部调用方、配置和测试。

## 模块地图

| 模块 | 主要职责 |
|---|---|
| `io.py`, `common.py` | 配置、表格读写、输出目录、运行清单和图片导出 |
| `preprocessing.py` | 表格编码、缺失处理、缩放、离散化、分布变换和异常值处理 |
| `signal.py` | MA、高斯、SG、Hampel 平滑；Z-score、Min-Max、Robust；Box-Cox、去趋势、差分和分数差分 |
| `anomaly.py` | 3σ、Z-score、IQR、Hampel、KNN、LOF、DBSCAN、One-Class SVM、Isolation Forest、EMD/VMD、ARIMA/Kalman 异常检测 |
| `correlation.py` | Pearson、Spearman、滚动相关、ACF/PACF、滞后相关、CCF、Granger 因果检验 |
| `decomposition.py` | STL、LOESS、EMD、VMD、三次样条插值 |
| `diagnostics.py` | 正态性、ADF、KPSS、GPH 和 ACF 长记忆诊断 |
| `forecast_models.py` | ARMA/ARIMA、ARFIMA、Prophet、Kalman/状态空间及 Prophet 数学构件 |
| `deep_models.py` | SSM、Mamba、S-Mamba、TimePro、STM3、MambaSL、LSTM |
| `deep_runner.py` | 深度时序模型的数据窗口、训练、评估和产物输出 |
| `validation.py` | 时间顺序切分、滚动/扩展窗口、AIC、RMSE 和预测区间 |
| `tabular.py` | sklearn 基线、自定义估计器、XGBoost、LightGBM、CatBoost |
| `timeseries.py` | 配置驱动的传统时序实验入口 |
| `statistics.py`, `decision.py`, `evaluation.py` | 统计、多元分析、决策和综合评价 |
| `clustering.py`, `optimization.py`, `forecasting.py` | 聚类、粒子群和 GM(1,1) |
| `cli.py` | `mathmodel` 命令行入口 |

## 环境

项目使用 Pixi 管理环境：

```powershell
pixi install
pixi run check
```

可选环境：

- `forecast`：statsmodels 与 MLForecast。
- `decomposition`：时序依赖、EMD 和 VMD。
- `prophet`：Prophet。
- `deep-timeseries`：PyTorch 深度时序模型。
- `full`：Boosting、解释、传统时序、异常检测和调参工具。

也可用 PEP 621 extra，例如 `pip install -e ".[timeseries]"`、`.[decomposition]`、`.[prophet]`、`.[deep-timeseries]` 或 `.[boosting]`。重量级能力必须延迟导入；缺少可选依赖时给出明确安装提示，不能静默切换算法。

## 运行

生成样例数据并运行配置：

```powershell
pixi run demo-data
pixi run mathmodel run -c configs/classification.yaml
pixi run mathmodel run -c configs/timeseries.yaml
pixi run -e deep-timeseries mathmodel run -c configs/s_mamba.yaml
```

检查数据：

```powershell
pixi run mathmodel inspect --data data/raw/classification.csv --target target
```

统一实验产物包括指标、逐样本预测、模型文件、`run.json` 和 PNG/SVG/PDF 图片。所有会学习参数的预处理只能在训练数据上拟合。时序数据必须使用时间顺序、滚动窗口或扩展窗口验证，禁止随机泄漏未来信息。

## 开发约定

- 新的公共算法优先加入现有扁平模块；只有单文件明显失控且形成独立子系统时才考虑子包。
- 非平凡实现先检查仓库和成熟开源库，按相关性、维护活跃度和集成成本决定复用方案。
- 优先原生 API，避免无必要依赖；新文件不添加版权头。
- 输入校验必须明确；不支持的算法、尺度或数据形状应抛出错误，不能输出看似有效的替代结果。
- 修改后至少执行 `pixi run lint` 与 `pixi run test`。深度模型修改还要执行 `pixi run -e deep-timeseries pytest tests/test_deep_timeseries.py -q`。
- 提交信息使用英文，格式为 `type(scope): description`。
- 不得在代码、注释、配置、测试、日志或提交中写入密钥和 token。

## 模型选择原则

根据目标、数据生成机制、样本量、时间/空间/分组结构、约束和评价指标选择模型，不把仓库已有算法当作白名单。候选方案至少包括可解释基线和一个与数据机制匹配的模型。ARFIMA 使用截断分数差分及递归逆变换，区间属于近似区间；Prophet、EMD、VMD 和深度模型属于可选依赖能力，报告结果时需注明实现与参数边界。
