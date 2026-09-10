# 按流程使用模板

这里的编号表示建模流程，不表示必须依次运行每个算法，也不表示模型范围受文件名限制。

```text
题目与附件
  ↓
01_data：加载原始数据
  ↓
02_eda：质量检查、探索数据和识别泄漏
  ↓
03_feature：结合业务机制做特征工程
  ↓
04_model：判断任务结构并进入对应子目录
  ├─ 普通表格分类/回归 → tabular
  ├─ 时间依赖预测       → timeseries
  ├─ 无监督异常/故障    → detection
  └─ 分群与结构发现     → clustering
  ├─ 多指标排序         → evaluation
  └─ 决策变量寻优       → optimization
  ↓
05_validation：按数据结构验证并比较候选方案
  ↓
06_explain：解释最终模型与特征贡献
  ↓
07_visualization：结论图和误差诊断
  ↓
08_report：汇总指标、预测、模型、运行清单和论文图片
```

## 模板不是模型白名单

`04_model/` 中的 `baseline.py`、`lightgbm.py`、`xgboost.py`、`catboost.py`、`stacking.py` 等只是常见基线。实际选择应由问题机制和验证结果决定。可以：

- 在 YAML 的 `custom_models` 中注册任何 sklearn 兼容估计器；
- 为深度学习、图模型、概率模型、空间模型等非 sklearn 接口增加独立适配器；
- 按数据结构增加 GroupKFold、嵌套交叉验证、滚动回测等验证方案；
- 替换现有特征工程和评价指标。

每个新方案应继续输出 `metrics.csv`、逐样本预测、模型/参数、运行环境以及 PNG/SVG/PDF 图片，确保论文结论可追溯。
