# 统计分析与随机运筹方法

本仓库不重复实现成熟算法。核心数值计算复用 NumPy/SciPy，估计器复用 scikit-learn；需要 ARIMA、VAR、单位根检验、完整回归诊断和多因素方差分析时，安装 `timeseries` 特性后复用 statsmodels；复杂层级模型或 MCMC 才单独引入 PyMC。

## 方法与模块

| 需求 | 本仓库入口 | 底层实现 | 使用建议 |
|---|---|---|---|
| 多元回归 | `multiple_regression` | sklearn + SciPy | 返回系数、标准误、t/F 检验和拟合诊断 |
| 逐步回归 | `stepwise_regression` | 上述 OLS | 只作探索基线；最终结论必须用留出集或嵌套验证确认 |
| 单因素方差分析 | `one_way_anova` | SciPy | 同时返回 ANOVA 表和 eta-squared；多因素/重复测量用 statsmodels |
| K-means 动态聚类 | `kmeans_cluster` | sklearn | 默认标准化、20 次初始化并返回轮廓系数 |
| 距离/Fisher/Bayes 判别 | `discriminant_classifier` | sklearn | 分别对应最近质心、LDA、QDA；高维条件独立基线可用 `gaussian_bayes_classifier` |
| 主成分分析 | `principal_components` | sklearn | 默认先标准化，可按累计解释率选维数 |
| 因子分析 | `factor_analysis` | sklearn | 支持 varimax/quartimax 旋转并返回载荷与得分 |
| 典型相关 | `canonical_correlation` | sklearn | 返回两组典型变量及逐对相关系数 |
| 共轭 Bayes | `beta_binomial_update` | SciPy | 适合比例/成功率；复杂 Bayes 模型建议 PyMC |
| 时间序列描述 | `autocorrelation`、`ljung_box`、`linear_trend` | SciPy | ARIMA、VAR、灰色预测已有独立模板 |
| 马尔可夫过程 | `markov_stationary`、`markov_n_step` | NumPy | 有限齐次离散链 |
| 排队论 | `mm1_queue` | 解析公式 | 仅适用于平稳 M/M/1；其他队列应仿真或使用专门模型 |
| 存储论 | `economic_order_quantity` | 解析公式 | 确定性 EOQ 与再订货点基线 |
| 决策论 | `expected_utility` | pandas | 风险型决策的期望收益排序 |

## 最小示例

```python
import pandas as pd
from mathmodel.statistics import multiple_regression, discriminant_classifier

result = multiple_regression(frame[["x1", "x2"]], frame["y"])
print(result.coefficients)

model = discriminant_classifier("fisher").fit(frame[["x1", "x2"]], frame["class"])
prediction = model.predict(frame[["x1", "x2"]])
```

所有会从数据学习参数的标准化、降维、变量选择和模型拟合，都应只在训练折执行。逐步回归的 p 值没有自动修正选择偏差；K-means 的类别编号也没有顺序含义。方差分析、Fisher 判别与经典队列公式各有分布假设，报告中应列出假设检查、敏感性分析和适用边界。
