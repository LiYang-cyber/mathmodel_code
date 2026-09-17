# Mamba 系列时序模型

仓库内置四个可训练的 PyTorch 模型，统一使用纯 PyTorch 状态空间扫描，不要求编译
`mamba-ssm`、DCNv4 或项目专用 CUDA 扩展。默认实现可在 CPU 上运行，也会在可用时使用
CUDA。

| 配置名 | 输入 | 输出 | 适用任务 | 核心结构 |
| --- | --- | --- | --- | --- |
| `s_mamba` | `[B,L,N]` | `[B,H,N]` | 多变量长期预测 | 变量维双向 SSM 与前馈块 |
| `timepro` | `[B,L,N]` | `[B,H,N]` | 多变量长期预测 | 时间分块、变量嵌入、超状态门控 |
| `stm3` | `[B,L,N,F]` | `[B,H,N,O]` | 时空预测 | 多尺度专家、节点路由、自适应图融合 |
| `mambasl` | `[B,L,F]` | `[B,C]` | 时序分类 | 单层 SSM 与多头自适应池化 |

其中 `B` 为批量大小，`L` 为输入长度，`H` 为预测长度，`N` 为变量或节点数，`F` 为
输入特征数，`O` 为节点输出维度，`C` 为类别数。

实现思路分别来自：

- [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba)
- [TimePro](https://github.com/xwmaxwma/TimePro)
- [STM3_KDD26](https://github.com/IfReasonable/STM3_KDD26)
- [MambaSL](https://github.com/yoom618/MambaSL)

## 环境与运行

```powershell
pixi install -e deep-timeseries
pixi run -e deep-timeseries mathmodel run -c configs/s_mamba.yaml
pixi run -e deep-timeseries mathmodel run -c configs/timepro.yaml
pixi run -e deep-timeseries mathmodel run -c configs/stm3.yaml
pixi run -e deep-timeseries mathmodel run -c configs/mambasl.yaml
```

S-Mamba 和 TimePro 直接读取 CSV/Excel。`feature_columns` 指定参与预测的数值列；省略时
使用所有数值列。流水线按 `seq_len` 和 `pred_len` 构造滑动窗口，并严格按时间顺序保留
测试集。

STM3 使用 NPZ，必须包含：

- `X`：`[samples, seq_len, num_nodes, input_dim]`；
- `y`：`[samples, pred_len, num_nodes, output_dim]`。

MambaSL 使用 NPZ，必须包含：

- `X`：`[samples, seq_len, features]`；
- `y`：`[samples]`，标签可以不是从零开始的整数，流水线会重新编码。

公共训练字段包括 `epochs`、`batch_size`、`learning_rate`、`weight_decay`、
`max_grad_norm`、`test_size`、`random_state` 和 `device`。`device: auto` 会优先使用 CUDA。
模型结构参数放在 `model_params` 下。

每次训练统一输出指标、逐样本预测、`model.pt`、`run.json` 和训练损失的
PNG/SVG/PDF 图片。保存的 `model.pt` 包含模型名称、构造参数和 `state_dict`，可以通过
`build_deep_timeseries_model()` 重建模型后加载。

## 与上游实验框架的关系

仓库内实现保留四个模型的主要数据流，但用统一纯 PyTorch SSM 替换了上游的定制 CUDA
扫描，因此适合比赛数据、跨平台训练和统一消融。如果需要逐项复现论文仓库的原始结果，
使用 `external_model.py` 和 `task: external_timeseries` 调用完整上游工程。
