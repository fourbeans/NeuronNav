# 树突记忆 v2：故障定位、修订与 Gibson 试跑

更新日期：2026-10-04。本文只记录树突分支；原始 PONI、Sigma-D U-Net、FMM 波前代码及 v1 产物均未改动。v2 新实验目录为 `dairy/dendritic_memory_20261004/gibson/pilot_v02/`。

## 1. 为什么启动 v2

v1 的 50k 步试跑实际得到 54,480 环境步、625 次参数更新。在相同的 100 个 Gibson 内部开发集 episode 上，原空间策略 Success 57%、SPL 0.3604；v1 树突策略 Success 47%、SPL 0.3498。此结果**不满足**不退化门槛，v1 没有产生 `best_dev.ckpt`。相关原始轨迹、配对结果和逐例诊断保留在 `gibson/pilot_v01/`、`diagnosis_0700/` 和 `DIAGNOSIS_0700.md`。

诊断确认两个可直接修复的原因：目标网络每 1000 次更新才同步，而整轮只有 625 次更新，因而始终与初始网络相同；旧评分头在候选 embedding 上有独立线性项，即使不看树突历史也可以改变排名。在 5,048 个离线决策状态中，v1 最终模型有 5,041 次选点偏离空间锚点，静态候选项的分数跨度约 0.533，大于历史交互项约 0.099。八个确定性闭环重跑轨迹的 episode ID 与轨迹哈希均已核对；多个失败 episode 存在大量目标跳变，但另有目标可见后的碰撞失败，不应将所有失败归因于跳变。

## 2. 接口及结构边界

前端仍为冻结的 Sigma-D U-Net，继续在每个真实导航步对更新地图产生空间分数 `S_spatial(c)`。Gibson 的局部导航仍复用原 FMM。树突只接收四组各 32 维、由在线可观测地图与运动历史构造的特征；每个真实步更新一次 `4 × 4 × 128` 隔室及胞体/读出状态。候选生成、显式坐标访问统计、目标可见 override、episode replay、DDQN learner 均由独立 `dendritic_nav/` 模块负责。在线候选没有使用 GT 或教师前向；GT 距离仅在训练奖励中出现。

v2 评分改为

```text
e(c) = signed_spike(SparseLinear(candidate_features(c)))
h_t = dendritic_readout_t
u_t(c) = W_interaction [e(c) ⊙ h_t]
Δ_t(c) = 0.03 × clip(u_t(c), -1, 1)
Q_t(c) = S_spatial(c) + Δ_t(c) + V(h_t)
```

`V(h_t)` 是所有候选共享的状态价值项，便于 DDQN 拟合与动作无关的回报，但不改变选点。只有 `e(c) ⊙ h_t` 能改变候选间排序。`Δ` 单点绝对值最多 0.03，两候选间最多影响 0.06 的相对分数；这是**数值约束**，不是运行时回退，仍允许低分差候选被树突历史重排。交互头和共享价值头零初始化，首次前向与空间锚点选点完全一致；其余稀疏投影正常初始化并在后续训练中学习。没有蒸馏、Softmax、RMSNorm、QK 外积、矩阵记忆或重复 T 次输入地图。

训练回报在原距离势函数塑形、成功/SPL 奖励与碰撞惩罚之外，增加每次**实际执行**长期目标切换 `-0.01`。目标可见后 override 的步不记为树突切换。树突状态不接触奖励中的 GT 距离。Double-DQN 的 target 每 100 次优化更新同步一次，50k 步试跑预计可同步约 6 次，而非 v1 的 0 次。训练指标增加 `target_syncs` 字段。

模块改动：`core/cell_dend.py` 仅负责单元和评分；`learning/reward_dend.py` 仅负责奖励；`adapters/bridge_dend.py` 在完成历史目标更新后计算奖励；`runners/train_dend.py` 负责目标同步与训练；`runners/prepare_dend.py` 支持 v2 checkpoint 作为后续独立大实验的初始化。模型族标识改为 `sigma_d_dendritic_v2`，避免 v1 权重误载。启动参数、源码哈希、环境与源码快照保存在每个实验目录。旧 v1 诊断脚本按当时源码快照复现；不要将 v2 模型直接送入 `diagnostics/inspect_training_dend.py`（它的权重分解只适用于 v1 评分头）。

## 3. 训练 / 评估数据与产物

Gibson 训练集按 `scene_split.json` 使用 23 个场景；两个不重合的内部开发场景为 Klickitat 和 Shelbyville。空间 checkpoint 固定为 `dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt`，其 SHA-256 记录于 v2 目录。树突训练不重训 U-Net；16 个并行导航 actor 产生完整 episode replay，learner 先热身 10k 环境步，此后每 64 步一个优化更新，batch 16、截断 BPTT 窗口 32、Adam 学习率 `1e-4`。原算法的 3 个**实际决策步**回报、`γ=0.995`、Huber DDQN loss 保持不变。训练数据来自在线 Gibson 导航轨迹，而非 PONI 教师离线标签。

预期/已建立的目录：

```text
pilot_v02/
  command.sh, config.json, manifest.json, source_snapshot.zip, scene_split.json
  checkpoints/initial.ckpt, checkpoints/published.ckpt
  actor_*/replay/episode_*.npz, actor_*/episodes.jsonl, actor_*/visualizations/*.png
  learner/metrics.jsonl, learner/tensorboard/, learner/checkpoints/last.ckpt
  learner/checkpoints/pilot_final.ckpt       # 完成训练后
  dev_baseline_*/episodes.jsonl, dev_dend_*/episodes.jsonl
  development_comparison.json                # 完成配对开发集后
  learner/checkpoints/best_dev.ckpt          # Success 和 SPL 点估计均不退化才产生
  logs_train/, logs_dev/, logs_dev_baseline/, pipeline_progress.json
```

不应把 `published.ckpt` 当成最终评估结果。内部开发集固定配对 100 episode，比较同 episode 的 Success/SPL 及配对 bootstrap 区间；仅作开发闸门，最终结论还需官方 val。v2 的树突权重与 replay、log 不与 v1 混用。

## 4. 启动命令

在 `xty-node1-direct` 的 `/mnt/jfzn/shensj/PONI` 执行，Python 路径为 `/mnt/jfzn/miniconda3/envs/poni/bin/python`：

```bash
python -m dendritic_nav.runners.prepare_dend \
  --root /mnt/jfzn/shensj/PONI \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v02 \
  --spatial-checkpoint /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  --seed 123

python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v02 \
  --gpus 1,2,3,0,5,4,6 --actors 16 --steps 50000 --learner-gpu 7 \
  --with-dev-pipeline --dev-actors 2 --baseline-first
```

这里的命令已经在 2026-10-04 启动；**不要重复在同一目录执行**。查看实时状态可读 `launch_train.json`、`pipeline_progress.json`、`logs_train/*.log`、`learner/metrics.jsonl` 和各 actor 的 `episodes.jsonl`。管线先在 actor 稳定后运行空间基线开发集，训练完成后对固定 `pilot_final.ckpt` 运行树突开发集并写比较报告。

若要在**另一个新实验目录**手动分别运行相同空间策略（原 Sigma-D 选点 + 原 FMM）与树突策略的配对开发集，可将下面的 `EXP` 换成那个目录。这里的 baseline 标志只关闭树突对长期目标的重排，地图、空间 checkpoint、目标可见处理和 FMM 均与新策略共用，因此是本实验最严格的原策略对照；不是另行训练原始 U-Net：

```bash
EXP=/mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/另一个全新实验目录
python -m dendritic_nav.runners.launch_dend --experiment "$EXP" \
  --role dev --no-learner --baseline --actors 2 --episodes 100 \
  --steps 51000 --gpus 1,2,3,0,5,4,6
python -m dendritic_nav.runners.launch_dend --experiment "$EXP" \
  --role dev --no-learner --actors 2 --episodes 100 \
  --steps 51000 --gpus 1,2,3,0,5,4,6 \
  --checkpoint "$EXP/learner/checkpoints/pilot_final.ckpt"
```

## 5. 已验证与后续闸门

- 修订前远端原代码：17 项单元测试通过；修订后：19 项通过，包括零初始化锚点、候选无关价值项、残差幅度上界、奖励切换项、时序梯度和 DDQN 反向传播。
- 使用原始 `inventory/original_sha256.json` 复核 PONI、Sigma-D、配置和固定空间 checkpoint 等 88 个文件：`changed: []`。
- v2 试跑已创建 23/2 场景划分、独立初始 checkpoint、源码快照；16 个 actor、learner 和 pipeline 共 18 个作业已启动。启动时 GPU 实时空余显存经过 admission check，actor 被分配至 GPU 0/1/2/3/4/5/6，learner 使用 GPU 7；未停止其他用户任务。
- 2026-10-04 约 07:38（北京时间）观察到 replay 已达 11,027 环境步、learner 已完成 15 次优化更新，最近三条 Huber loss 为 0.0439/0.0456/0.0417，均有限值；尚未到第一次 100 次 target 同步。配对 baseline 评估已并行启动，训练及 baseline 日志扫描暂未见 traceback/OOM。此为**运行健康快照**，不是效果结论。
- 训练与配对评估结果**尚未完成**；不得以单元测试或训练 loss 宣称导航效果改善。
- 若 `development_comparison.json` 的 Success 或 SPL 点估计低于空间基线，先分析闭环失败，不启动长期大实验。若两项均不退化、学习审计证实交互头和树突输入参数已更新，并且目标切换无异常，再从 v2 `best_dev.ckpt` 初始化新的独立大规模目录，保持原版、v1、v2 pilot 可各自启动。不得在旧目录直接续写新版本日志。

监控：已创建当前任务的按小时主动检查，跟进训练、actor 异常、GPU、配对结果和后续大实验闸门。每次重要变化写回 dairy；不能将“持续监控”理解为当前进程永久占用一个交互会话。

## 6. 2026-10-04 08:31 北京时间监控记录

- Learner 已正常结束：`learner/completion.json` 记录 50,092 已摄取环境步、625 次优化更新、训练历时 1,394.6 秒；训练指标最终记录 6 次 target 同步。16 个 actor 的完整 episode replay 合计 53,592 步，较 learner 截止点多出的步数来自各 actor 结束当前完整 episode，不表示额外优化更新。
- 配对开发集仍在运行。监控瞬间空间基线已完成 71/100 episode，v2 已完成 41/100 episode；`pipeline_progress.json` 指向 `launch_dev.json`，两个 v2 开发集 actor 仍为 pending。**尚无** `development_comparison.json`、`pipeline_complete.json` 或 `best_dev.ckpt`，不能宣称效果合格，也不应启动大规模训练。
- 训练 replay 的目标切换原始计数：v1 为 31,289 / 54,480 真实步（57.43%），v2 为 24,408 / 53,592 真实步（45.54%），相差约 -11.89 个百分点。此为不同在线训练轨迹的非配对诊断，含探索随机行为，不能代替配对开发集 Success/SPL，也不能断言已解决全部跳变问题。
- 已扫描训练、基线及 v2 开发集日志，未发现 traceback、OOM 或进程被杀的标记。GPU 显存（0–7 号）使用约为 37/12/19/19/66/55/32/12 GiB；其中还包括其他用户进程，未对其做任何操作。当前正在进行训练参数学习审计。
- 同次监控稍后对**已完成且 episode ID 相同的 45 对**作只读中途比较：基线 Success 37.78%、SPL 0.2107；v2 Success 28.89%、SPL 0.1856。差值为 -8.89 个百分点和 -0.0251。这是按完成速度截取的非随机部分样本，**不能代替 100 对最终结果**，但提示当前版本可能仍有闭环退化；在最终闸门与失败轨迹分析前不启动大规模训练。

## 7. 2026-10-04 11:12 北京时间：v2 完整试跑结论

100 对开发集 episode 全部完成，每个评估 actor 50 条。`pipeline_complete.json` 和 `development_comparison.json` 于服务器时间 09:28 产生；没有 `pipeline_failure.json`，也没有合格的 `learner/checkpoints/best_dev.ckpt`。

| 指标 | 相同 100 对的空间基线 | v2 树突 | 配对差值 | 配对 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Success | 43.00% | 30.00% | -13.00 个百分点 | [-22.00, -5.00] 个百分点 |
| SPL | 0.21082 | 0.18171 | -0.02911 | [-0.06478, 0.00166] |

按事先约定的两项指标点估计均不低于基线的闸门，v2 **失败**；`pilot_gross_regression` 为 true。Success 区间整体为负，SPL 区间包含 0。此开发集结果不是官方验证集的最终结论，但足以否决从 v2 启动长训练及 MP3D 集成。v1 的开发集基线是 57%，v2 为 43%，二者评估 actor 数和 episode 样本不同；**不能直接比较两轮原始 Success/SPL 均值**，只能各自与同轮的配对基线比较。v1 配对差为 Success -10 个百分点、SPL 约 -0.0106；v2 为 -13 个百分点、-0.0291，不能据此声称 v2 已改进 v1。

`learning_audit.json` 在 53,592 条训练 replay 步上核查了 49,725 个实际树突决策状态：4 条树突输入投影的最大权重变化均约 0.055–0.067，候选编码器 0.1116、交互头 0.0873、共享价值头 0.0299，说明它们确实参与了训练；平均树突隔室非零发放比例约 8.73%，53,109 步有非零残差。但是，**将最终模型回放到训练轨迹时有 32,892 / 49,725（66.15%）个决策改变空间锚点**。有界到每候选 ±0.03 的残差仍会在空间分数近似持平时频繁重排候选；这与在线训练目标切换率仍达 45.54% 相互吻合，是下一轮重点诊断对象。回放比例不是闭环导航性能，也不是最终评估的实际选点率。

11:12 检查发现 `xty-node1-direct` 一度 SSH 超时，短超时重连成功；随后 8 张 GPU 的显存读数均约 1 MiB、利用率 0%，可能是节点作业清空或重启，**没有证据表明是树突实验主动终止了其他任务**。v2 的训练与开发集在此之前已完整结束。待新实验完成设计后再做显存 admission；不以占显存为由启动无意义任务。

进一步的只读 replay 分析使用 `python -m dendritic_nav.diagnostics.spatial_margin_dend <v1目录> <v2目录>`。v2 训练期实际执行的非锚点动作是 13,048 / 49,725（26.24%），而最终权重回放同一批状态是 66.15%；后期学到的评分头显著提高了重排频率。v2 有 81.35% 的决策状态中最高与次高空间候选分差不超过 0.02，97.79% 不超过 0.06；因此单候选 ±0.03 的限制**不足以保护锚点排名**。v2 实际选点离锚点超过 2 m 的比例 15.45%，v1 为 32.30%，但 v2 仍在配对开发集明显退化。以上 v1/v2 训练轨迹不成对，不可归因于模型改动的单一效应。

为查清闭环失利类型，另在全新 `dairy/dendritic_memory_20261004/diagnosis_v02/` 启动了 4 条确定性诊断轨迹：开发 actor 0 的前 2 个 episode、actor 1 的前 4 个 episode，各自同时跑原空间选点和 v2 权重。对应启动命令为：

```bash
python -m dendritic_nav.diagnostics.launch_cases_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v02 \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/diagnosis_v02 \
  --case 0:2 --case 1:4 --gpus 0,1,2,3
```

诊断只读取冻结的 v2 `pilot_final.ckpt` 并写入新目录，不改变训练、原版或 v2 评估结果。待比对 episode ID 与轨迹哈希后再据此设计下一版，不能直接把残差再缩小就宣称解决问题。
