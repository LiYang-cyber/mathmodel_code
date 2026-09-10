# 华为杯历年题目算法整理

资料来源为 [HUAWEI_Math_Modeling_Knowledge README](https://github.com/LUORANCHENG/HUAWEI_Math_Modeling_Knowledge/tree/main)，读取日期为 2026-09-10。原仓库没有许可证文件，本仓库没有复制其中代码；新增实现只依据 README 的算法名称、公开公式和第三方库 API 编写。

代码优先调用仓库已有依赖：PCA、MLP 和相关系数使用 scikit-learn/pandas，粒子群的返回对象沿用 SciPy `OptimizeResult`，VAR 使用 statsmodels。GM(1,1)、灰色关联和熵权 TOPSIS 规格较小，本仓库按公式实现。第三方依赖版本由 `pixi.lock` 固定，许可证以各包随附元数据为准。

## README 中的主要算法和知识点

| 年份题号 | 主要算法和知识点 |
|---|---|
| 2023 A | 排队论、马尔科夫链、二分法、牛顿法 |
| 2022 A | 最小二乘、变分贝叶斯、压缩感知、超分辨定位 |
| 2021 A | 相关分析、SVD |
| 2020 A | LS、目标规划、蒙特卡洛、CR |
| 2024 B | FFT、并行算法、稀疏矩阵分解、GPU 计算 |
| 2023 B | DFT、稀疏优化、递推、蝶形分解、模型逼近 |
| 2022 B | 整数线性规划、启发式算法、深度优先搜索 |
| 2021 B | XGBoost、BP、随机森林、VAR、多元线性回归、灰色关联 |
| 2020 B | 灰色关联、BP、遗传算法、随机森林、相关分析、粒子群、非线性规划 |
| 2023 C | 混合整数规划、BP、随机森林、动态加权排序 |
| 2022 C | 多目标优化、遗传算法、马尔可夫决策、动态规划、模拟退火 |
| 2021 C | 最小二乘、Hodgkin-Huxley、多目标优化 |
| 2020 C | PCA、SVM、随机森林、K-Means |
| 2023 D | 回归、路径规划、随机森林、GA-MPSO、LSTM |
| 2022 D | 0-1 规划、图论、模拟退火、SVM、遗传算法 |
| 2021 D | SVM、神经网络、目标规划、遗传算法、灰色关联、随机森林 |
| 2020 D | 对抗规划、博弈论、BP |
| 2023 E | 决策树回归、随机森林、CNN、SVM、BP、XGBoost |
| 2022 E | 随机森林-LightGBM、熵权 TOPSIS、模拟退火、K-Means、XGBoost |
| 2021 E | BP、分类预测、最小二乘、粒子群、UWB 定位 |
| 2020 E | 灰色预测、CNN、多元回归、时间序列、相关分析 |
| 2023 F | 集成学习、XGBoost、BP、熵权 TOPSIS、随机森林、LSTM |
| 2022 F | 整数规划、遗传算法、模拟退火、ARIMA、粒子群、Dijkstra |
| 2021 F | 多目标规划、启发式算法、贪心算法、机组排班 |
| 2020 F | 贪心、模拟退火、差分进化、粒子群、分层优化 |

2024 年回顾还涉及非线性/随机优化、WLAN 机理、信号特征、方差分析、地理加权回归、GIS、交通流、实时预警、轨道力学、传播时延、泊松过程和脉冲轮廓折叠。

## 本轮加入仓库的部分

| 新增模板 | 选择原因 | 位置 |
|---|---|---|
| PCA/SVD | 2021 A、2020 C 及优秀论文复现中出现；适合高维相关特征 | `03_feature/pca_svd.py` |
| Pearson/Spearman/Kendall/灰色关联 | 多年 B、D、E 题出现；用于筛查关系与影响因素 | `03_feature/correlation.py` |
| MLP/BP | 多个年份反复出现；用 sklearn 提供轻量基线 | `04_model/tabular/mlp.py` |
| GM(1,1) | 2020 E 的小样本趋势预测 | `04_model/timeseries/grey_forecast.py` |
| VAR | 2021 B 的多变量时序预测 | `04_model/timeseries/var.py` |
| 熵权 TOPSIS | 2022 E、2023 F 的综合评价 | `04_model/evaluation/entropy_topsis.py` |
| 粒子群 | 2020 B、2021 E、2022 F、2020 F 出现 | `04_model/optimization/particle_swarm.py` |

未在本轮加入的算法不代表不重要。整数规划、图算法、LSTM/CNN、地理加权回归、马尔科夫模型、信号处理和机理模型需要不同的数据契约或运行环境，后续应按具体题目增加，不宜塞进一个通用函数。
