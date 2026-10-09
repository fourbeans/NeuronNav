# 树突记忆 v3：基于时序证据激活的候选残差

日期：2026-10-04。只修改 `dendritic_nav/` 下的树突实验分支；原 PONI、Sigma-D U-Net、FMM 和 v1/v2 结果均保留。模型族为 `sigma_d_dendritic_v3`，与 v2 checkpoint 不兼容；新实验须使用独立目录。

## 1. v2 失败证据与 v3 的单一修正目标

v2 在同 100 个 Gibson 开发集 episode 上比空间基线 Success 低 13 个百分点（43% → 30%，配对 95% CI [-22,-5] pp），SPL 低 0.0291（0.2108 → 0.1817）。未产生合格 `best_dev.ckpt`，长训练及 MP3D 接线未启动。虽然 v2 target 网络同步 6 次、交互头和树突输入都确实学习，它的最终权重回放使 32,892 / 49,725（66.15%）个训练决策改动空间锚点。因为最高和次高空间候选的分差在 81.35% 的决策状态里不超过 0.02，`±0.03` 的残差仍可频繁翻转排名。

`diagnosis_v02/summary.json` 中 12 条确定性轨迹（两组 actor 的 6 对）全部与原评估的 episode ID 和轨迹 SHA-256 完全一致。三个原基线成功、v2 失败的代表案例：

| 案例 | 空间基线 | v2 | v2 非锚点 / 切换 / >2m 跳点 | 启示 |
| --- | --- | --- | --- | --- |
| actor 0, ep 2 | 27 步成功 | 20 步失败 | 4 / 6 / 4 | 早期少量局部改点就可能改变短程成功 |
| actor 1, ep 3 | 218 步成功 | 500 步失败 | 365 / 430 / 414 | 远距离反复跳点，虽无碰撞仍失败 |
| actor 1, ep 4 | 49 步成功 | 500 步失败 | 419 / 461 / 440 | 低空间分差掩盖了很大的目标坐标跳变 |

这些案例中 v2 平均牺牲的空间分数仅约 0.000008 / 0.0087 / 0.0029；只缩小分数残差并不一定能消除跨米级跳点。v3 的单一目标是：**没有可观测的时序受困证据时，不让树突改变候选排名；有证据时才允许学习到的树突残差发挥作用。**这是事件触发评分结构，不是根据效果在运行时切换回某个教师/基线模型。

## 2. 数据流与算法

空间网络与候选生成、局部 FMM 均维持原流程。四分支树突仍每个真实导航步更新一次，状态规模不变；只增加一个由在线可观测历史确定的二值事件门。显式坐标历史为每个 25 cm 栅格统计**离开后重新进入的次数**，连续停留在同一格不重复计数。这个 `entry_count` 写到探索分支特征 `f[1,9]`，其余时序特征含义保持不变。

设 `E_t` 为当前坐标格的累计进入次数，`L_t` 为当前长期目标连续停滞步数，`C_t` 为当前坐标格记录的碰撞次数：

```text
g_t = 1[E_t >= 3 OR L_t >= 10 OR C_t >= 2]
e_t(c) = signed_spike(SparseLinear(candidate_features_t(c)))
h_t = dendritic_readout_t
Δ_t(c) = g_t × 0.03 × clip(W_interaction[e_t(c) ⊙ h_t], -1, 1)
Q_t(c) = S_spatial,t(c) + Δ_t(c) + V(h_t)
```

`V(h_t)` 仍为候选共享价值项，不影响排名。比较、乘法和有界脉冲足够实现事件门，不引入 Softmax、RMSNorm、QK 外积、矩阵状态或地图重复 T 次输入。门的判据来自实际路径与历史统计，不读取 GT、未来信息或原 PONI 教师输出。即使门未激活，树突本身仍按真实导航步更新，以便后续触发时利用之前的时间信息；仅候选评分修正为零。

选择门槛前使用 `diagnostics/evidence_gate_dend.py` 对 v2 轨迹进行**反事实可用性审计**（不能当 v3 导航结果）。上述门在 v2 训练轨迹约 28.0% 的步上开启；在 actor 0 ep 2 的 20 步中为 0 次，在 actor 1 ep 3 的 500 步中为 8 次、ep 4 为 0 次。对这三个轨迹，门覆盖的 v2 非锚点动作分别为 0/4、1/365、0/419。观察结果支持它抑制已知的失败模式，但闭环轨迹会因策略改变而分叉，**不能由此推断 v3 会成功**。

## 3. 训练、版本和产物

从随机初始化训练树突，不蒸馏、不加载 v2 权重。冻结的 Sigma-D U-Net checkpoint、23 个 Gibson 训练场景、2 个内部开发场景保持不变。仍采用 online actor 完整 episode replay、10k 步 warmup、每 64 环境步一次更新、batch 16、BPTT 窗口 32、3 个实际决策步的 Double-DQN、Adam `1e-4`、成功/SPL 与目标距离塑形及每次目标切换 `-0.01` 的训练奖励。Target 网络每 100 次优化更新同步。增加 `learner/checkpoints/update_00100.ckpt` 等每 100 次更新的可回溯快照，避免只留下最终权重；最终是否合格仍须固定 checkpoint 的**完整配对导航评估**，不能凭训练 loss 或局部轨迹挑选。

v3 新目录：

```text
dairy/dendritic_memory_20261004/gibson/
  smoke_v03/          # 2k 环境步的接线/格式测试；不做效果结论
  pilot_v03/          # 待 smoke 通过后启动的正式小试跑及配对开发集
```

每个新目录含 `source_snapshot.zip`、`scene_split.json`、原空间权重哈希、`manifest.json`、`command.sh` 和运行参数；actor replay/轨迹可视化、learner JSONL/TensorBoard、初始/中间/最终 checkpoint、配对评估及日志互不混用。v1/v2 的完整结果见 `REVISION_V02.md`、`gibson/pilot_v01/` 和 `gibson/pilot_v02/`。

## 4. 启动与验证记录

在 `xty-node1-direct` 的 `/mnt/jfzn/shensj/PONI` 使用 `/mnt/jfzn/miniconda3/envs/poni/bin/python`，执行前须确认 GPU 空余显存。已完成 21 项树突单元测试，包括新增的坐标进入计数、门触发/关闭、零残差锚点、稀疏连接和 DDQN 反向传播。

已启动的 smoke 命令：

```bash
python -m dendritic_nav.runners.prepare_dend \
  --root /mnt/jfzn/shensj/PONI \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/smoke_v03 \
  --spatial-checkpoint /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  --seed 123
python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/smoke_v03 \
  --gpus 0,1,2,3 --actors 4 --steps 2000 --learner-gpu 7
```

Smoke 的 learner 因环境步数小于 10k warmup，不会产生有意义的梯度更新；检查点是环境、候选、版本和 replay 接线。待所有 4 个 actor 与 learner 均写 `completion.json` 且 replay 校验通过，才启动 50k 左右的 `pilot_v03`，使用更多独立 actor 和配对开发集。该完整结果若仍低于基线，继续诊断，不启动长训练；若两项点估计均不退化且树突确实学习，再准备独立大规模目录。

### 16:00 北京时间进展

`smoke_v03` 的 4 个 actor 和 learner 均正常完成，没有 traceback/OOM；learner 摄取 2,368 个真实步、10 个完整 replay episode，`updates=0` 符合 10k warmup 预期。实测新进入次数特征 `f[1,9]` 的最大值为 7，证明在线特征接线存在。原 PONI/Sigma-D 等 88 个文件与固定 checkpoint 哈希复核仍为 `changed: []`。此检查**不涉及导航质量**。

已另建 `gibson/pilot_v03/` 并启动 23 个场景独立 actor、50k 步 learner、10 个并行开发集 actor 的自动配对评估。实际命令：

```bash
python -m dendritic_nav.runners.prepare_dend \
  --root /mnt/jfzn/shensj/PONI \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v03 \
  --spatial-checkpoint /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  --seed 123
python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v03 \
  --gpus 0,1,2,3,4,5,6,7 --actors 23 --steps 50000 --learner-gpu 7 \
  --with-dev-pipeline --dev-actors 10 --baseline-first
```

本次 `dev-actors=10` 且每 actor 恰好 10 episode，总计 100 对；相较 v2 的 2 actor 可以缩短评估等待。v3 的具体 episode 样本和基线结果未必与 v2 相同，因此比较必须以 `pilot_v03/development_comparison.json` 的**同轮配对差值**为准，不做跨轮原始均值直比。`launch_train.json` 记录了 25 个作业（23 actor、learner、pipeline）及 GPU 分配；运行时还需继续观察显存、每个 actor 的完成标志和目标同步次数。

启动后健康检查：`launch_dev_baseline.json` 已出现，说明 10 个配对空间基线 actor 已并行进入评估；learner 约 13,869 环境步、39 次更新，最近损失为有限值，尚未到第 100 次 target 同步。训练/基线日志扫描暂未见 traceback 或 OOM。八卡显存使用约 29–32 GiB/卡，仍留有较多空余；不通过虚占用任务填满显存。当前阶段**尚无** Success/SPL 结论。

### 16:06 北京时间自动检查

`pilot_v03` 继续正常运行。learner 最新记录 23,865 已摄取环境步、178 次优化更新、1 次目标网络同步；`learner/checkpoints/update_00100.ckpt` 已生成。23 个训练 actor 与 learner 仍处于 `pipeline_progress.json` 的训练待完成清单；10 个空间基线 actor 已产生部分 episode。训练、基线和开发集日志扫描未见 traceback/OOM。八卡显存使用约 33–42 GiB/卡（含其他进程），最高卡仍余约 39 GiB。尚无 `development_comparison.json` 或 `best_dev.ckpt`，**未进行效果闸门判定，也未启动长训练**。

### 17:00 北京时间：100 对结果及后续检查

`pilot_v03` 的训练及 100 对内部开发集评估已完整结束，没有 `pipeline_failure.json`，训练/评估日志未见 traceback/OOM。Learner 摄取 54,913 环境步、625 次优化更新；每 100 次的 6 个中间 checkpoint 已写入 `learner/checkpoints/update_00100.ckpt` 至 `update_00600.ckpt`。

| 指标 | 同轮空间基线 | v3 最终 checkpoint | 配对差值 | 配对 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Success | 50.00% | 44.00% | -6.00 个百分点 | [-11.00, -2.00] 个百分点 |
| SPL | 0.26274 | 0.25282 | -0.00993 | [-0.02018, -0.00245] |

v3 最终 checkpoint **仍不符合**预先确定的两项不退化闸门，两项配对 CI 均为负；没有 `best_dev.ckpt`，不会从它启动长训练。它的原始均值不能和 v2 直接比较（评估 actor/episode 流不同），只能说各自对其同轮空间基线的点估计差为 v2 的 -13 pp/-0.0291 与 v3 的 -6 pp/-0.00993；这不是严格的跨版本配对改善证明。

为确定是否只有最终权重恶化，在独立 `gibson/checkpoint_screen_v03/` 中启动**六个已保存 checkpoint 的同 episode 评估**，每轮并行三个 checkpoint、每个 10 actor / 100 episode，显存 admission 逐轮检查；原 `pilot_v03/dev_baseline_*/episodes.jsonl` 作为同 ID 的固定空间对照。它只是训练过程 checkpoint 选择，不是消融或多 seed，更不改训练数据、原版或 v3 评估结果。启动命令（执行一次，控制器后台运行，日志在 `gibson/checkpoint_screen_v03.log`）：

```bash
python -u -m dendritic_nav.runners.screen_checkpoints_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v03 \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/checkpoint_screen_v03 \
  --gpus 0,1,2,3,4,5,6,7 --parallel-checkpoints 3
```

控制器 `launch.json` 已记录首轮 30 个真实评估作业，按 checkpoint 分别存放 `actor_*/episodes.jsonl`、日志和可视化；`results.json` 会在每个 checkpoint 完成后写同轮配对 Success/SPL，`completion.json` 标记六个权重筛查全部完成。筛查多个权重会带来开发集选择偏差：即使其中某个通过，也只能称为内部候选，仍需独立 Gibson 官方 val 确认，再考虑大规模训练。若全部失败，保留全套证据并重新检查学习目标/算法，而不是用显存占用掩盖失败。

额外只读 replay 审计：v3 在线训练轨迹中，实际非空间锚点决策为 6,952 / 51,292（13.55%），目标切换 19,378 / 54,913 步（35.29%），非锚点目标距锚点超过 2 m 的比例 8.66%。v2 相应为 26.24%、45.54%、15.45%。这些比例来自不同在线训练轨迹，不能作成对因果比较；但它们显示事件门确实减少了重排/跳点，同时开发集最终仍退化，说明“少改点”本身不足以保证导航效果。

### 17:09 北京时间 checkpoint 筛查进度

首轮 `update_00100/00200/00300` 的 30 个 actor 已启动，合计完成 171 / 300 个目标 episode；控制器按整轮全部完成后才写三项 `results.json`，目前尚无可用于闸门的完整结果。评估日志未检出 traceback/OOM。GPU 0 使用约 69.4 GiB、剩余 11.6 GiB，其余卡约 25–34 GiB 使用；GPU 0 已接近预留边界，下一轮须由控制器重新按实时空余显存 admission，不追加外部占卡作业。

### 18:09 北京时间：首轮 checkpoint 完成，缩小第二轮并恢复

首轮每个权重均在与 v3 原开发集基线**完全相同的 100 个 episode ID**上评估完毕，空间基线 Success 50%、SPL 0.262743。`checkpoint_screen_v03/results.json` 的逐项结果：

| checkpoint | Success | 对基线差 | SPL | 对基线差 | 双指标点估计不退化 |
| --- | ---: | ---: | ---: | ---: | --- |
| `update_00100.ckpt` | 49% | -1 pp | 0.261901 | -0.000842 | 否 |
| `update_00200.ckpt` | 50% | 0 pp | 0.262351 | -0.000393 | 否 |
| `update_00300.ckpt` | 49% | -1 pp | 0.262211 | -0.000532 | 否 |

三者都比最终 checkpoint（44%、0.252818）更接近空间基线，但**没有一个满足预先确定的 Success/SPL 双指标闸门**。筛选多个训练节点还会产生开发集选择偏差，不能挑接近的数值就宣布有效。

首轮 30 个 actor 全部完成后，控制器按实时空余显存尝试为第二轮再接纳 30 个 actor，因空余不足而抛出 `Insufficient free VRAM for the checkpoint wave`，**未**强行启动第二轮，也没有中断或调整其他进程；首轮结果/日志完整保留。这个错误发生在已完成 actor 写完 `completion.json` 与 CUDA 进程完全退出/释放显存之间的边界，且 GPU 0/1 当时还有较高占用。随后只修改 checkpoint 筛查器：新轮次须等待进程真正退出才算前轮完成；新增 `--resume` 从现有 `results.json` 跳过已完成三个权重，保留原输出及启动库存。编译与 21 项单元测试通过。

恢复命令改用 GPU 2–7、每轮 2 个 checkpoint / 20 actor，已有旧 30 个作业仍保留于 `launch.json`，恢复后新轮已追加 20 个真实评估作业；剩余 `update_00400/00500/00600` 不重复首轮：

```bash
python -u -m dendritic_nav.runners.screen_checkpoints_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v03 \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/checkpoint_screen_v03 \
  --gpus 2,3,4,5,6,7 --parallel-checkpoints 2 --resume
```

恢复日志：`gibson/checkpoint_screen_v03_resume.log`；恢复参数与时间：`checkpoint_screen_v03/resumes.jsonl`。尚未进行官方 Gibson val，也没有启动大规模训练。

### 22:19 北京时间：六个 checkpoint 筛查完成

`checkpoint_screen_v03/completion.json` 已生成，六个 checkpoint 各完成与同一空间基线配对的 100 episode，合计 600 条树突评估轨迹；评估 actor 日志未发现 traceback/OOM。固定基线 Success 50.00%、SPL 0.262743。完整结果汇总：

| 训练更新数 | Success | Success 差值 | SPL | SPL 差值 | 双指标点估计不退化 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 49% | -1 pp | 0.261901 | -0.000842 | 否 |
| 200 | 50% | 0 pp | 0.262351 | -0.000393 | 否 |
| 300 | 49% | -1 pp | 0.262211 | -0.000532 | 否 |
| 400 | 48% | -2 pp | 0.260453 | -0.002290 | 否 |
| 500 | 46% | -4 pp | 0.255636 | -0.007107 | 否 |
| 600 | 45% | -5 pp | 0.253606 | -0.009138 | 否 |
| 最终 625 | 44% | -6 pp | 0.252818 | -0.009925 | 否 |

早期权重接近空间基线，但没有一个通过事前确定的两指标点估计闸门；训练继续后，Success/SPL 持续走低。第 200 次权重 Success 与基线相同，但 SPL 仍低 0.000393，不能把“接近”当“不退化”。由于全部未通过，**没有从任何 v3 权重启动独立官方 val 或大规模训练**。当前八卡读数均约 1 MiB/0% 利用率，说明筛查已结束、GPU 释放；不以无目标占用任务制造假进展。

下一步先诊断训练目标：当前 DDQN 将原空间势函数 `S_spatial` 当作固定 Q 项，而势函数并非经过 Bellman 校准的动作价值；树突交互头只见到已执行候选的回报，缺少其他候选的反事实结果。约 500 步终止奖励又被 `γ=0.995` 折扣，晚期成功信号较弱。由此**推测**继续优化会逐渐学到对 TD 误差有利、对导航选点不利的残差；这是待验证的机制，不应宣称已证实。需要结合真实失败轨迹确认是否仍是目标跳变或出现其他故障模式，再设计新的训练目标，而非单纯延长训练。

为此，在全新 `diagnosis_v03/` 启动三个基线成功/v3 最终权重失败的确定性成对重跑：开发 actor 1 的前 3 个 episode、actor 2 和 5 的前 5 个 episode，分别覆盖原基线 218/236/401 步成功而 v3 均达 500 步失败的三个案例；共 6 个诊断作业，使用 GPU 0–5。启动命令：

```bash
python -m dendritic_nav.diagnostics.launch_cases_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v03 \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/diagnosis_v03 \
  --actors 10 --case 1:3 --case 2:5 --case 5:5 --gpus 0,1,2,3,4,5
```

诊断写新目录，不修改 v3 结果。完成后须先核验 26 条重跑 episode 的 ID 和轨迹哈希与原评估完全相同，才解释其中选点/碰撞/目标跳变行为。

### 22:27 北京时间：失败轨迹重跑进度

六个诊断作业中，actor 1 的基线与 v3 两项均已完成各 3 个 episode；actor 5 的基线已完成 5 个，v3 仍在最后一个；actor 2 的两项均完成 4/5 个。正在运行的四个进程健康，六份日志尚未检出 traceback、RuntimeError 或 CUDA OOM。GPU 2–5 各用约 8.3 GiB，GPU 0/1/6/7 读数约 1 MiB；这些是当前真实需求，不启动无关占卡作业。轨迹尚未全部完成，因此尚未执行 26 条哈希核验，也不从部分案例推断退化原因。长期训练继续保持未启动。

### 22:36 北京时间：26 条轨迹完全复现后的失败诊断

六个诊断作业均正常完成，`diagnosis_v03/summary.json` 收录 26 条重跑 episode，全部与原 `pilot_v03` 评估的 episode ID 和**完整轨迹 SHA-256**一致（`all_reproduced_exactly=true`）。因此下表对同一 episode 的描述是可复核的；但只选了三个原基线成功、v3 失败的案例，不代表全部 100 对的失败机制。

| 案例 | 基线成功步数 / 目标切换 / 25cm 独立位置格 | v3 结果 / 非锚点选择 / 切换 / >2m 目标跳点 / 碰撞 | v3 事件门开启 / 首次开启 / 首次目标分叉 |
| --- | --- | --- | --- |
| actor 1, ep 3 | 218 / 78 / 90 | 500 步失败 / 375 / 401 / 375 / 0；仅 40 个位置格 | 384/500 / 第 31 步 / 第 37 步 |
| actor 2, ep 5 | 236 / 97 / 85 | 500 步失败 / 241 / 373 / 142 / 253；87 个位置格 | 281/500 / 第 74 步 / 第 75 步 |
| actor 5, ep 5 | 401 / 122 / 127 | 500 步失败 / 356 / 442 / 408 / 0；仅 52 个位置格 | 385/500 / 第 38 步 / 第 38 步 |

三个案例的非锚点选择**全部**发生在事件门开启时；门关闭时非锚点数均为 0。这排除了“门未接线”的实现错误，却揭示了更核心的失败：门一旦触发，它在闭环中可以连续高频保持开启，仍让很小的空间分数差导致数米级候选翻转。平均被牺牲的空间分数仅 0.00510 / 0.00870 / 0.01154，对应非锚点候选到空间锚点的平均距离却达 5.05 / 1.24 / 3.73 米。门触发源分别是：actor 1 重访 377 步、停滞 7 步、碰撞 0 步；actor 2 重访 28、停滞 4、碰撞 249 步；actor 5 重访 383、停滞 2、碰撞 0 步。触发源的统计有重叠，不能相加为门开启步数。

actor 1 和 5 属于**无碰撞的反复重访/大幅改点**，探索覆盖显著低于同 episode 的空间基线；actor 2 则有大量碰撞，不能简单归因于同一种“跳点”机制。三例的首个目标分叉都不早于事件门首次触发，与门内重排导致闭环分叉相符；这仍是相关性证据，非对每个动作的反事实因果证明。

配对图保存在 `figures/diagnosis_v03/`，例如 [actor 1 空间基线](figures/diagnosis_v03/actor1_ep3_baseline.png) 与 [actor 1 v3](figures/diagnosis_v03/actor1_ep3_v3.png)；另有 actor 2、5 各两张。图含地图/轨迹、四分支脉冲活动、累计奖励与候选得分，可检查树突在后期持续活跃且导航覆盖收缩的现象。原始 `.npz`、逐 episode 记录及 `summary.json` 仍在独立 `diagnosis_v03/`，不修改训练或评估产物。

**当前判断与下一步：**v3 的“有受困证据才可重排”只限制了触发时刻，没有约束触发后的改点距离与持续性；原 DDQN 仍用未 Bellman 校准的空间势函数作 Q 的固定项，且没有候选的反事实目标。因此只调大/调小事件门阈值或继续训练都没有充分依据。下一轮应先做一个结构上受限、可学习的候选修正小试验：保留零初始化树突残差和在线历史，不用运行时 PONI 回退；把跨米级改点成本与长期目标持续性明确纳入候选排序/学习目标，并在同 100 对开发集上检查 Success、SPL、切换率和轨迹覆盖。它必须独立命名与存档，并再次通过双指标闸门后才考虑官方 val 与大规模训练。此处是诊断提出的**待验证方案**，不是已实现结果。
