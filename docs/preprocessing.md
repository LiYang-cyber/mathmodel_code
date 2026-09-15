# 数据预处理方法

预处理参数只能从训练集学习。做交叉验证时，把编码器、填补器、缩放器和离散器放进 Pipeline；先处理整份数据再切分，会把验证折的信息带进训练折。

## 编码与缺失值

`01_data/encoding.py` 支持数字编码和 One-Hot。数字编码会把新类别记为 `-1`，适合树模型或确有顺序含义的类别；线性模型通常使用 One-Hot，以免类别编号制造不存在的远近关系。

`01_data/missing_values.py` 提供四种处理办法：

| 方法 | 参数 | 用法与限制 |
|---|---|---|
| 删除 | `drop` | 缺失很少且近似随机时使用；删除后要检查样本分布是否改变 |
| 均值填补 | `mean` | 数值基线，速度快；会压低方差，不适合强偏态列 |
| 随机填补 | `random` | 从训练列已观测值抽样，能保留边际分布；随机种子固定为 42 |
| 模型填补 | `model` | Extra Trees 配合 IterativeImputer；列之间有预测关系时可用，计算较慢 |

模型训练配置可设置 `numeric_imputer: median|mean|random|model`、`categorical_encoder: onehot|ordinal`。类别缺失仍采用众数填补。

## 缩放、中心化与分布转换

`01_data/scaling.py` 支持以下方法：

- `zscore`：训练集均值为 0、标准差为 1；
- `minmax`：映射到 `[0, 1]`，新数据可能超出该范围；
- `decimal`：除以适当的 10 的幂；
- `logistic`：先按训练集均值和标准差处理，再映射到 `(0, 1)`；
- `center`：只减训练集均值，不改变尺度。

`03_feature/distribution_transform.py` 提供 `log`、`yeo-johnson` 和 `quantile-normal`。对数转换适合正值或经过平移的右偏变量；Yeo-Johnson 可处理零和负数；分位数正态转换对异常值不敏感，但会改变变量间距，样本少时也容易过拟合。

## 离散化

`03_feature/discretization.py` 把连续列转换为整数区间编号。`uniform`、`quantile`、`kmeans` 不使用标签；`information_gain`、`chimerge`、`caim` 必须传入目标列，因此只能在训练折拟合。

信息增益方法使用熵准则决策树找切点。ChiMerge 合并卡方值最小的相邻区间，CAIM 逐次加入能提高类别依赖指标的切点；两者都限制最大区间数。候选切点超过 256 个时先做分位数预分箱，避免在大表上耗费过多时间。监督离散化可能过度贴合训练标签，必须用独立验证折判断是否值得保留。

## 异常值

`01_data/outliers.py` 包含 Z-score、IQR、MAD 和局部离群因子（LOF）。前三种给出逐列异常标记；LOF 根据邻域密度给出整行标记，适合多变量局部异常。

处理动作有三种：`remove` 删除异常行，`clip` 截到统计边界，`nan` 置为空值后交给填补器。LOF 没有逐列上下界，只能删除或另行检查。异常点也可能是真实极端事件，不能看到标记就删；比赛论文应记录判定规则、删除数量及结论对处理阈值是否敏感。

## 命令示例

```powershell
pixi run python grad-ml-template/01_data/encoding.py data/raw/data.csv --columns category --method onehot
pixi run python grad-ml-template/01_data/missing_values.py data/raw/data.csv --method model --columns x1 x2 x3
pixi run python grad-ml-template/01_data/scaling.py data/raw/data.csv --columns x1 x2 --method zscore
pixi run python grad-ml-template/01_data/outliers.py data/raw/data.csv --columns x1 x2 --method mad --action nan
pixi run python grad-ml-template/03_feature/discretization.py data/raw/data.csv --column age --method caim --target label
pixi run python grad-ml-template/03_feature/distribution_transform.py data/raw/data.csv --columns income --method yeo-johnson
```
