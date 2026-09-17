# mathmodel-code Agent 指南

进入仓库后先读取本文件，了解项目结构、能力、工作流和开发约定。

## 项目定位

这是面向数学建模的 Python 模块武器库，重点服务适合机器学习求解的题目，同时覆盖表格分类与回归、聚类、异常检测、计算机视觉、深度学习、时间序列、统计分析、评价与优化。公共能力以扁平模块放在 `src/mathmodel/`，配置在 `configs/`，示例数据生成器在 `examples/`，测试在 `tests/`，运行产物写入 `outputs/`。

`workflow.py` 提供任务无关的步骤契约和组件注册表。独立算法应保持可组合，流程可以按具体赛题连接数据准备、特征工程、模型、验证、解释和报告模块；配置驱动的现有 runner 是可复用流程实例，而不是模型范围的限制。

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
| `neural_models.py` | MLP、RNN、GRU、LSTM 与 Transformer 通用分类模型 |
| `deep_runner.py` | 深度时序模型的数据窗口、训练、评估和产物输出 |
| `vision_models.py` | LeNet、AlexNet、VGG、NiN、GoogLeNet、ResNet18、DenseNet |
| `visualization.py` | 曲线、图像网格、热力图和训练指标面板 |
| `workflow.py` | 通用步骤契约、上下文传递和模型/变换/验证器组件注册 |
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
- `vision`：PyTorch 视觉与通用深度学习模型。
- `full`：Boosting、解释、传统时序、异常检测和调参工具。

也可用 PEP 621 extra，例如 `pip install -e ".[timeseries]"`、`.[decomposition]`、`.[prophet]`、`.[deep-timeseries]`、`.[vision]` 或 `.[boosting]`。重量级能力必须延迟导入；缺少可选依赖时给出明确安装提示，不能静默切换算法。

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
- 模块接口使用清晰的数据契约，既能单独调用，也能包装成 `WorkflowStep` 或注册进 `ComponentRegistry`；流程层不得绑定某一种题型。
- 非平凡实现先检查仓库和成熟开源库，按相关性、维护活跃度和集成成本决定复用方案。
- 优先原生 API，避免无必要依赖；新文件不添加版权头。
- 输入校验必须明确；不支持的算法、尺度或数据形状应抛出错误，不能输出看似有效的替代结果。
- 修改后至少执行 `pixi run lint` 与 `pixi run test`。深度模型修改还要在对应的 `deep-timeseries` 或 `vision` 环境执行模型测试。
- 提交信息使用英文，格式为 `type(scope): description`。
- 不得在代码、注释、配置、测试、日志或提交中写入密钥和 token。

## 模型选择原则

根据目标、数据生成机制、样本量、时间/空间/分组结构、约束和评价指标选择模型，不把仓库已有算法当作白名单。候选方案至少包括可解释基线和一个与数据机制匹配的模型。ARFIMA 使用截断分数差分及递归逆变换，区间属于近似区间；Prophet、EMD、VMD 和深度模型属于可选依赖能力，报告结果时需注明实现与参数边界。

## D2L 来源

绘图接口与经典深度学习结构参考并适配自 Dive into Deep Learning 官方稳定发行版 `v1.0.3`（commit `b2e2ae30898a9d0126a9699ae7e441de3e272715`）。适配代码使用本项目统一接口和当前依赖版本；具体许可归属见根目录 `THIRD_PARTY_NOTICES.md`。
