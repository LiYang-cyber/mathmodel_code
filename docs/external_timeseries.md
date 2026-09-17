# 研究型时序模型适配

仓库通过统一适配器运行四个独立研究工程，用于复现上游实验。日常建模优先使用仓库内
实现，参见 [`deep_timeseries.md`](deep_timeseries.md)。外部适配器使用参数列表调用上游 `run.py`，不会经过 shell；
每次运行在 `outputs/<run_name>/` 保存 `command.json`、`run.json`、`stdout.log` 和退出码。

| 适配器 | 任务 | 上游仓库 |
| --- | --- | --- |
| `s_mamba` | 多变量长期预测 | [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) |
| `timepro` | 多变量长期预测 | [TimePro](https://github.com/xwmaxwma/TimePro) |
| `stm3` | 长期时空预测 | [STM3](https://github.com/IfReasonable/STM3_KDD26) |
| `mambasl` | 多变量时序分类 | [MambaSL](https://github.com/yoom618/MambaSL) |

## 使用方法

先在仓库根目录下准备外部工程。也可以放在其他位置，只需相应修改配置中的
`repository_path`。

```powershell
git clone https://github.com/wzhwzhwzh0921/S-D-Mamba external/S-D-Mamba
git clone https://github.com/xwmaxwma/TimePro external/TimePro
git clone https://github.com/IfReasonable/STM3_KDD26 external/STM3_KDD26
git clone https://github.com/yoom618/MambaSL external/MambaSL
```

按照各上游 README 创建独立环境并安装依赖。然后修改 `configs/s_mamba.yaml`、
`configs/timepro.yaml`、`configs/stm3.yaml` 或 `configs/mambasl.yaml`：

- `python` 可填写对应环境的 Python 绝对路径；省略时使用当前解释器；
- `repository_path` 指向上游仓库；
- `arguments` 原样对应上游 `run.py` 参数，列表值会展开成多个参数；
- 示例默认 `dry_run: true`，只校验并记录命令；确认命令后改为 `false` 才会训练。

YAML 布尔值 `true` 用于 `--use_amp` 这类无值开关，`false` 表示不传该开关。对于上游
定义为 `--use_gpu True` 这类带值参数，应使用带引号的字符串 `"True"` 或 `"False"`。

路径类上游参数相对于 `repository_path` 解析。为避免歧义，外部数据建议使用绝对路径。

```powershell
pixi run python grad-ml-template/04_model/timeseries/external_model.py --list
pixi run mathmodel run -c configs/s_mamba.yaml
```

这些模型不是可互换的同类基线。S-Mamba 与 TimePro 用于多变量长期预测；STM3 还要求
节点化的时空数据；MambaSL 是分类模型。应先按任务结构和验证方式筛选，再与朴素预测、
ARIMA、VAR 或传统时序分类器比较，不宜只报告单个深度模型结果。
