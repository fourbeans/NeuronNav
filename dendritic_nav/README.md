# Sigma-D + 简化树突记忆（独立实验线）

本目录只增加树突残差选点，不修改 `poni/`、`semexp/`、`hlab/` 中任何原文件。
空间网络使用已训练 Sigma-D checkpoint，冻结全部参数；原目标可见接管、语义建图、FMM 均复用。
没有蒸馏、教师前向或运行时退回原 PONI。`--baseline` 是独立对照实验，不是回退开关。

## 模块与边界

| 目录 | 职责 | 不应承担的职责 |
|---|---|---|
| `core/` | 固定稀疏连接、整数事件训练参考、候选集、残差评分 | 环境、GT、文件管理 |
| `memory/` | 显式全局坐标历史、四类观察特征 | GT 距离、奖励、神经网络权重 |
| `adapters/` | 新的 Gibson policy 子类、终止转移捕获、episode 生命周期 | 复制整个 mapper / evaluator |
| `learning/` | 奖励、完整 episode 回放、循环 Double DQN | 再跑空间网络、生成教师标签 |
| `runners/` | 独立启动、快照、数据划分、资源检查、训练、报告 | 算法核心逻辑 |
| `visualization/` | 离线轨迹、候选分数、分支活动图 | 在线决策 |
| `tests/` | 不依赖 Habitat 数据的数值和协议测试 | 最终导航效果证明 |

所有功能 Python 文件使用 `_dend.py`；`__init__.py` 仅为标准包入口。
运行生成的 `command.sh`、`config.json`、`resolved_habitat_args.json` 是实际参数记录；不是另一套隐藏配置。

## 当前范围

- Gibson：独立采样、冻结 Sigma-D、树突前向、真实终止奖励、完整回放、DDQN 训练、同 episode 开发集报告。
- MP3D：遵循先通过 Gibson 门槛再接入的顺序。**本版不提供伪装成可运行的 MP3D 入口**。
  MP3D 窗口原点改变需要在新适配器显式处理，不能直接用 Gibson 固定窗口入口。
- 首轮是 50k 环境步试训，不做消融或多 seed 搜索。各 actor 的 `123 + actor_id` 是同一训练运行的随机流编号。
- 不承诺训练后一定超过空间基线；必须用配对 Success/SPL 实测。初始零残差一致性不等于训练后保证。

详细原理、命令、路径和验证记录见 `dairy/dendritic_memory_20261004/README.md`。

## 快速入口

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
python -m unittest dendritic_nav.tests.test_core_dend -v
python -m dendritic_nav.runners.prepare_dend --help
python -m dendritic_nav.runners.launch_dend --help
python -m dendritic_nav.runners.train_dend --help
python -m dendritic_nav.runners.collect_dend --help
```

新运行拒绝覆写同名目录。续训只接受 `train_dend --resume`，空间 checkpoint 不会被写入。
