# 树突记忆 v4：局部候选修正 + 势函数与 TD 价值解耦

日期：2026-10-04 至 2026-10-05。状态：v4 pilot 与内部开发集评估已完成，学习审计进行中；**内部开发集点估计持平不等于已证实提升**。只修改 `dendritic_nav/` 树突实验分支并创建本文件；原 PONI、Sigma-D、v1/v2/v3 检查点和实验目录不得覆盖。模型族为 `sigma_d_dendritic_v4`，与 v3 checkpoint 不兼容，从随机初始化训练，不蒸馏、不加载教师权重。

## 1. 为什么改

`REVISION_V03.md` 记录了六个中间 checkpoint 和最终权重的同 100 对开发集结果：所有权重均未同时达到空间基线 Success/SPL。最终权重为 44% / 0.252818，空间基线为 50% / 0.262743。`diagnosis_v03/summary.json` 的 26 条重跑轨迹与原评估完全一致。三个基线成功/v3 失败的案例中，树突事件门开启后产生 241–375 次非锚点选择；其中两例平均非锚点到空间锚点的距离为 3.73–5.05 米，并反复目标跳变、探索覆盖收缩。事件门并未坏掉；仅限制触发条件不足以稳定触发后的选择。

此外，v3 训练把未做 Bellman 校准的 U-Net 空间势函数 `S_spatial` 直接加到 Q 预测里，回归折扣回报。空间分数是选点先验，不是以导航奖励为单位的动作价值；这可使 TD 学习目标与实际策略排序错位。该机制是有根据的假设，不应在 v4 结果出来前宣称已得到因果证明。

## 2. 接口、数据流与算法

环境每个真实导航步生成一张更新后的语义地图；冻结的 Sigma-D U-Net 一次增量前向得到势函数，原后处理产生空间锚点。原 128 候选生成、Gibson FMM、四分支×四隔室×128 维树突状态、显式坐标统计、事件门和每真实步一次更新均保持不变；**不把同一地图重复输入 T 次**。本次只改候选评分器及训练损失：

```text
语义地图 --冻结 Sigma-D U-Net--> 空间势函数 S(c)
       --原候选生成--> 128 个以内候选，c=0 为原空间锚点
真实导航步的 4×32 证据 --四分支树突状态 h_t--> 候选交互 a_t(c)
候选坐标相对锚点的 L1 距离 d(c,0) --> 定点友好系数 κ(c)
策略评分 P_t(c)=S_t(c)+0.03κ(c)a_t(c)+V(h_t)
训练 critic Q_t(c)=V(h_t)+a_t(c)；执行前只按 P 排序
```

形式化地，`a_t(c)=g_t·clip(wᵀ[spike(E c)⊙h_t],-1,1)`，其中 `g_t` 为 v3 已验证接线的重访/停滞/碰撞证据门。`κ(c)` 只由候选和空间锚点的**在线可见坐标**求得：

| `d(c,0)=|Δx|+|Δy|` | `κ(c)` |
| --- | ---: |
| ≤0.5 m | 1 |
| (0.5,1] m | 1/2 |
| (1,2] m | 1/4 |
| >2 m | 1/16 |

比较、绝对值、加法和二进制移位即可实现；没有 Softmax、RMSNorm、QK 外积、矩阵记忆或专用 CUDA kernel。远点**仍可**因学习到的优势和微小空间分差获选，并非运行时调用/回退原 PONI；但相同树突输出对远点的排名扰动变小。候选不足仍沿用有效掩码。

训练时 Bellman 回归的是独立 `Q_t(c)`，不是 `S_t(c)+…`：对 replay 中实际选择的候选 `c_t`，采用 3 个真实决策间隔的折扣回报 `R_t^{(3)}` 与在线策略选出的后继候选 `c'=argmax_c P_online(s',c)`，目标为

```text
y_t = R_t^(3) + γ^k Q_target(s', c')
L_TD = mean_t SmoothL1(Q_online(s_t,c_t), stopgrad(y_t)),  γ=0.995
```

终止步没有 bootstrap。完整 episode replay 重构前缀树突状态，窗口长 32、batch 16、warmup 10k 环境步、每 64 环境步一次更新、目标网络每 100 更新同步；优化器 Adam `1e-4`。冻结空间网络只在导航 actor 前向，不参加树突训练。训练奖励仍用现有环境成功/SPL、距离势差、碰撞和目标切换；GT 距离只在训练奖励桥接中使用，不进入部署特征。v4 不额外运行 PONI 教师、不做监督蒸馏。策略 `P` 与 critic `Q` 共用树突候选优势，这比 v3 的固定 `S`-Q 更接近所选策略的价值估计，但离线偏差、候选未执行的反事实缺失仍是风险，必须由成对导航结果判定。

代码边界：`core/ranking_v4_dend.py` 只做分档距离系数；`core/cell_dend.py` 只提供一次候选 `evaluate()` 的策略/残差/critic；`learning/td_v4_dend.py` 只做 recurrent TD；`runners/train_dend.py` 只切换到新损失；`tests/test_v4_dend.py` 测几何、解耦和梯度。v3 的原代码副本保存在其 `source_snapshot.zip`；旧 checkpoint 的模型族检查防止误载。

## 3. 实验顺序、命令与产物

服务器：`ssh xty-node1-direct`，根目录 `/mnt/jfzn/shensj/PONI`，Python `/mnt/jfzn/miniconda3/envs/poni/bin/python`。所有命令在根目录执行，保留命令本身及参数到实验目录 `command.sh`/`manifest.json`。空间网络固定为：

```text
dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt
```

先运行 `python -m unittest discover -s dendritic_nav/tests -v`，再准备并接线 smoke：

```bash
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.prepare_dend \
  --root /mnt/jfzn/shensj/PONI \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/smoke_v04 \
  --spatial-checkpoint /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  --seed 123
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/smoke_v04 \
  --gpus 0,1,2,3 --actors 4 --steps 2000 --learner-gpu 7
```

Smoke 的 2k 步小于 10k warmup，**只证明接线、版本、episode/replay 和 checkpoint 格式，不证明学习或导航质量**。通过后再建 `gibson/pilot_v04/`，同样用 `prepare_dend` 与同一空间 checkpoint/seed，启动 23 个独立训练场景 actor、约 50k 环境步、10 个开发集 actor 的自动 100 对配对评估：

```bash
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04 \
  --gpus 0,1,2,3,4,5,6,7 --actors 23 --steps 50000 --learner-gpu 7 \
  --with-dev-pipeline --dev-actors 10 --baseline-first
```

`smoke_v04/` 与 `pilot_v04/` 均独立保存 `source_snapshot.zip`、场景 split、空间权重哈希、初始/发布权重、actor episode/replay、learner `metrics.jsonl`/TensorBoard/每 100 更新 checkpoint/最终 checkpoint、训练与评估日志及 `development_comparison.json`。所有评估使用同一轮相同 episode ID 的空间基线比较，Success 与 SPL **两项点估计均不低于基线**且学习审计通过才考虑官方 Gibson val 和独立大规模训练；否则保留失败结果、诊断、不启动长训练。MP3D 未在此轮小试跑验证，不得推断其效果。GPU 只给真实训练/评估任务使用；未过闸门时不以空占显存代替实验。

## 4. 实际执行记录

第一轮服务器测试 24 项中 17 项通过、7 项在同一新 `locality_gain()` 调用处因旧版 PyTorch 不接受 `torch.where` 的混合标量类型而报 `expected scalar type double but found float`；当时未启动环境实验。已改用 `ones_like` 构造同 dtype 的分档值。第二轮完整测试 **24/24 通过**，包括原有树突不变量、v4 几何分档、critic 不直接依赖固定空间势函数，以及 v4 TD 对价值头和候选头的梯度。

随后用本文件第 3 节的命令创建 `gibson/smoke_v04/`：固定 23 个训练场景、2 个内部开发场景，原 Sigma-D 权重及哈希记录在 `spatial_checkpoint.json`；`source_snapshot.zip`、v4 初始/发布 checkpoint 已生成。`launch_train.json` 登记 4 个真实导航 actor 分布在 GPU 0–3、learner 在 GPU 7，共 5 个子进程。smoke **刚启动，尚待检查完成标志和异常日志**；因 2k 步小于 10k warmup，即使完成也不能推断训练效果。

### Smoke 结论

`smoke_v04` 的 4 个 actor 与 learner 均写出 `completion.json`，共 10 个完整 episode、learner 摄取 2,368 个真实环境步；`updates=0` 与 10k warmup 一致。训练日志未检出 traceback、RuntimeError 或 CUDA OOM。树突分支以 `sigma_d_dendritic_v4` 初始化、前向、写 replay 与保存最终 checkpoint 均通过接线检查。原 PONI、Sigma-D 文件及固定 checkpoint 的 88 项哈希再次核验为 `changed: []`。**这只是 smoke，不是 Success/SPL 效果验证。**下一步在独立 `pilot_v04/` 进行有真实梯度更新的 50k 步训练与同 100 对开发集评估。

### Pilot 启动

已用第 3 节同一 `prepare_dend` 参数建立 `gibson/pilot_v04/`，并执行该节列出的 `launch_dend` 命令。`launch_train.json` 登记 **25 个真实作业**：23 个训练 actor、1 个 learner、1 个配对开发集控制器；开发集先运行空间基线，再使用 v4 的最终权重评价相同 episode。23 actor 的 GPU 分配为 `[6,2,3,6,2,7,5,1,0,4,3,6,2,7,5,1,0,4,3,6,2,7,5]`，由启动器根据实际空余显存给出。启动前八卡各已有约 11–17 GiB 的其他作业占用，因此未人工强占显存，也未干扰那些进程。启动后的初次日志扫描未见异常；当时 actor 仍在加载、尚无完整 episode，**不能据此报告训练或效果进度**。后续须检查 `learner/metrics.jsonl`、每 100 更新 checkpoint、`pipeline_progress.json`、所有评估 actor 的完成标志和最终 `development_comparison.json`。

### 约 23:55 北京时间首个训练审计

Learner 已摄取 21,682 个真实环境步，完成 126 次优化更新，`target_syncs=1`，`learner/checkpoints/update_00100.ckpt` 已存在。最近一次训练 loss 为有限值 0.03193、梯度范数 0.01112；这不是效果指标。第 100 次更新权重相对随机初始值的最大绝对变化：`interaction_head.weight` 0.00609、`value_head.weight` 0.00405、`cell.projections.0.weight` 0.00414、`cell.gain` 0.00622，说明候选头、价值头和树突输入/隔室增益均收到实际训练更新，不能说只是空间基线在跑。10 个空间基线开发集 actor 已启动，初次检查累计完成 5 条 episode。`pipeline_failure.json` 不存在；训练与基线评估日志未检出 traceback/RuntimeError/OOM。八卡使用约 35–44 GiB/卡，仍有约 37–46 GiB/卡空余；该用量包含其他用户作业。尚无 v4 配对 Success/SPL，暂不启动官方 val 或大规模实验。

### 00:33 北京时间：pilot 与内部开发集完成

23 个训练 actor 全部完成，留下 275 个完整 replay episode。Learner 总计摄取 54,700 个真实环境步、625 次优化更新、6 次目标网络同步；`update_00100.ckpt` 至 `update_00600.ckpt`、`pilot_final.ckpt` 均存在。最后一条训练 loss 0.003443，为有限值；**loss 下降不等于导航成功**。10 个空间基线与 10 个 v4 开发集 actor 全部完成各 100 episode，`pipeline_complete.json` 与 `development_comparison.json` 存在，无 `pipeline_failure.json`，训练/评估日志未检出 traceback、RuntimeError 或 OOM。八卡现均约 1 MiB / 0% 利用率，说明 pilot 已结束并释放 GPU。

| 指标 | 同 episode 空间基线 | v4 最终 checkpoint | 配对差值 | 配对 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Success | 50.00% | 50.00% | 0 pp | [-3, +3] pp |
| SPL | 0.262743 | 0.262854 | +0.000111 | [-0.002527, +0.002858] |

`development_comparison.json` 的**双指标点估计不退化**门为 true，并复制 `learner/checkpoints/best_dev.ckpt`。但两个置信区间均含 0，不能据此宣称优于空间基线或泛化有效。100 个 episode ID 严格配对；逐 episode 核对发现 29 对完整闭环轨迹哈希不同，Success 有 1 对改善、1 对恶化，净值 0。v4 并非完全没有改变行为，也不能把 SPL 的 +0.000111 当成稳健获益。

正在执行全量 replay 学习审计：

```bash
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.audit_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04
```

该审计遍历 275 个 replay episode，计算参数变化、树突活动、非零残差与最终权重的回放改点数；运行几分钟时 CPU 100% 仍在正常计算，`learning_audit.json` 尚未生成。**在其完成并通过前不启动官方 val。**

官方 Gibson val 预检确认有五个 `.glb.json.gz` 场景及对应 `.glb` 资产，但其 `content/` 还混有五个 `_episodes.json.gz` 辅助文件；此前仅树突分支的 val 枚举使用 `*.json.gz`，会误把辅助文件当成场景。已在 `dendritic_nav/runners/collect_dend.py` 新增 `official_val_scenes()`，只选 `*.glb.json.gz` 并核对资产；测试同时构造辅助文件证明其被排除。服务器完整树突测试现在 **25/25 通过**，实际返回五场景：Collierville、Corozal、Darden、Markleeville、Wiconisco。此修正只影响尚未启动的官方 val，不改上述开发集或训练结果。官方 val 须使用独立目录和新源码快照；即使通过，也仍需明确它只是有限样本的无退化验证。

### 00:43 北京时间：完整学习审计与官方 val 接线准备

`gibson/pilot_v04/learning_audit.json` 已完成 275 episode / 54,700 步 / 51,512 有效决策的最终权重回放。`interaction_head.weight` 最大绝对变化 0.03794，四条 `cell.projections.{0..3}.weight` 的最大变化分别为 0.01917、0.02775、0.04263、0.01875，`encoder.weight` 为 0.06402，`value_head.weight` 为 0.02243；固定漏电常数与稀疏 mask 变化为零，符合设计。树突事件平均活跃比例 8.48%，最终权重在训练 replay 状态中有 19,899 步产生非零残差、1,162 / 51,512 个贪心决策改变空间锚点。**审计通过的是连接与学习，不是导航质量。**

另对实际在线训练轨迹做无模型前向的只读统计：非锚点动作 5,216 / 51,512（10.13%），全有效决策中与锚点相距 >2 m 的动作 4,308（8.36%），总目标切换 18,721 / 54,700 步（34.23%），当时策略记录有非零残差的步 11,514。在线数据来自不断更新的权重并含 epsilon 探索；它不能与最终权重回放的 1,162 次直接相减，也不能与 v3 的不同训练轨迹作严格配对因果比较。几何衰减**没有消灭**远点改选，内部开发集持平也未证明大幅改善。

为准备独立验证，只在 `dendritic_nav/` 中把官方 val 场景分片扩展为 20 个独立 actor 循环覆盖 5 个 val 场景，每 actor 25 episode，合计 500 对；同编号 baseline/v4 actor 使用相同场景与 seed，最终报告严格检查 episode ID 完全配对。新增 `runners/report_val_dend.py` 将其写到 `official_val_comparison.json`，不覆盖内部开发集报告。服务器完整测试 **26/26 通过**。已新建 `gibson/official_val_v04/`，用相同 Sigma-D checkpoint/seed、当前 val 修正后的 `source_snapshot.zip` 独立存档；该目录目前只是准备产物，不代表已验证。

正式 val 启动方法（先 baseline、确认真实 actor 已加载并产出 episode，再启动 v4，避免显存误判）：

```bash
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04 \
  --gpus 0,1,2,3,4,5,6,7 --role val --no-learner \
  --actors 20 --episodes 500 --steps 260000 --baseline
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.launch_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04 \
  --gpus 0,1,2,3,4,5,6,7 --role val --no-learner \
  --actors 20 --episodes 500 --steps 260000 \
  --checkpoint /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04/learner/checkpoints/best_dev.ckpt
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.report_val_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04 \
  --checkpoint /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04/learner/checkpoints/best_dev.ckpt
```

只有 40 个评估 actor 均写 `completion.json` 且同 500 个 episode ID 完全配对，最后一条报告命令才可执行。无论点估计如何，仍须报告 CI 与逐 episode 得失，不把这次内部开发集选择后的独立验证误称为随机多 seed 复现。

### 04:08 北京时间：官方 val 基线完成，v4 配对臂启动

官方 val 空间基线 `launch_val_baseline.json` 登记 20 个独立 actor，实际完成标记 **20/20**、`episodes.jsonl` 合计 **500/500**，日志未检出 traceback、RuntimeError、CUDA OOM 或 FileNotFoundError。中途远程 SSH 曾一次连接超时，重试后成功；服务器上的基线进程已自行完整结束，不是实验错误。第二臂启动前八卡均约 1 MiB/0% 利用率，显存释放。随后按本文件的命令使用固定 `pilot_v04/learner/checkpoints/best_dev.ckpt` 启动 v4 官方 val 20 个 actor，`official_val_v04/launch_val.json` 已登记全部 20 个作业，分配 GPU `[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3]`。此刻尚无 v4 官方 val 结果；必须等 20 个 actor 全部完成并执行配对报告后才判定闸门。

启动后的首次健康检查：20 个 v4 actor 中已有 8 个写出首条 episode，尚无完成标记；日志未检出上述异常，八卡显存约 16.5–25.0 GiB/卡。这是加载与早期运行进度，**不是中途效果结论**。自动跟踪任务已改为监控 `official_val_v04`，完成后严格核对 500 个 episode ID 并生成独立报告。

### 约 05:00 北京时间：官方 val 完成，但只能算 125 个独立 episode

官方 val 两臂的评估 actor 均 **20/20 完成**，各自 `episodes.jsonl` 都有 500 条记录；日志未见 traceback、RuntimeError、CUDA OOM 或 FileNotFoundError，八卡全部释放。原 `report_val_dend.py` 严格按 episode ID 要求 500 个唯一配对时拒绝生成报告，提示重复 ID；这是**正确地阻止虚假的 500 样本结论**，不是导航 actor 失败。

逐 actor 检查发现，每个基线/v4 对应 actor 的 25 条 episode **顺序、ID 与场景都逐项一致**；但 20 个 actor 是五个官方 val 场景各重复四次，且同场景四个 actor 均从相同的前 25 条 episode 开始。各重复 ID 在同一策略内的成功、SPL 与完整轨迹哈希均完全相同。因此 500 条配对运行实际上是 **125 个不同 episode × 4 次完全重复**，不能视为 500 个独立样本。此问题是固定官方 val episode 枚举与循环场景分片的组合；内部开发集 100 对 episode 的 ID 曾独立核对为唯一，不能把这次重复反推到已有内部开发集结果。

已仅修正树突分支的 `runners/report_val_dend.py`：先按 actor ID 与顺序确认两臂严格配对，只有同一 episode ID 的重复记录在场景、Success、SPL、轨迹哈希均完全一致时才折叠为一条独立样本；否则报错，不静默平均或增加有效样本量。新增拒绝非精确重复的单元测试，服务器完整测试 **27/27 通过**。修正报告源码另存为 `official_val_v04/report_val_dend_corrected.py`，SHA-256 `a4c7aa04accc81eb0f392e331653282410bdc695f905e42b3cb40d416d4afbac`；该报告修正发生在评估完成后，不修改已执行策略或原轨迹。正式 `official_val_comparison.json` 已写出，含 `recorded_pair_runs=500`、`distinct_episode_ids=125`、`exact_duplicate_pair_runs=375`。

| 指标 | 空间基线 | v4 固定 `best_dev.ckpt` | 125 对配对差值 | 配对 bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Success | 73.60% | 73.60% | 0 pp | [0, 0] pp |
| SPL | 0.415058668 | 0.414926682 | **-0.000131986** | [-0.000527945, 0] |

125 个独立 episode 中有 19 个闭环轨迹哈希改变，Success 没有改善或恶化的 episode；仅一个 Darden episode 的 SPL 发生变化：两臂都成功，基线 194 步 / SPL 0.1296607，v4 214 步 / SPL 0.1131624。这单例拖低整体 SPL，CI 的上端等于 0，不能声称具有可靠统计劣势，但**事前规定的 Success 和 SPL 两项点估计均不低于基线的闸门为 false**。不能把内部开发集 +0.000111 与官方 val -0.000132 的微小差额解释成稳健提升。v4 更接近“不明显破坏空间策略”，还没有证明树突记忆带来导航收益。

该次评估 checkpoint 为 `gibson/pilot_v04/learner/checkpoints/best_dev.ckpt`（SHA-256 `72e719424aacefb665efe7ec19a356f1c1516118dfab32e0f255c2953e5675cf`），结果 JSON 的 SHA-256 为 `fbf37f0fa8293b8505cdb0eccf8a9c069f2123edd969b295ec49b0ca5225659b`。原 PONI、Sigma-D 源码/权重 88 项哈希复核仍是 `changed: []`。**不从 v4 启动大规模训练，不启动 MP3D 扩展，也不为了寻找有利样本而事后重复挑选 checkpoint 或评估集。**若继续研究，应先针对“真实改变目标却几乎没有成功收益”的学习目标/候选策略进行新版本的小规模验证，并保持 v4 全套产物只读。

### 约 05:50 北京时间：v4 收尾与唯一 SPL 差异的只读复现

再次核查官方 val：基线与 v4 均为 20/20 actor 完成、去重报告存在、全部日志无上述异常，`dendritic_nav` 遗留进程为 0。GPU 1、6 当时有其他用户的显存使用，其余基本空闲；不会为占卡而运行无关实验。官方 val 严格不退化闸门已失败，长训练及 MP3D 仍不启动。

为定位唯一使 SPL 降低的 Darden episode，仅新增独立诊断启动器 `dendritic_nav/diagnostics/launch_val_case_dend.py`（不改变 v4 策略），在 `diagnosis_val_v04_darden_ep7/` 启动 actor 2 的前 7 个 episode 的基线/v4 轨迹重跑，分别使用 GPU 0、2 和原固定 `best_dev.ckpt`：

```bash
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.diagnostics.launch_val_case_dend \
  --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04 \
  --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/diagnosis_val_v04_darden_ep7 \
  --checkpoint /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04/learner/checkpoints/best_dev.ckpt \
  --actor-id 2 --actors 20 --episodes 7 --gpus 0,2
```

`launch.json` 已记录两个真实子进程与日志。**必须先核对重跑的 14 条轨迹 episode ID 和 SHA-256 均与原官方 val 完全一致**，再分析第 7 条的首次目标分叉、非锚点选择、目标切换、碰撞和局部覆盖；尚无诊断结论，不得把重跑当新的独立 val 样本。

### Darden 第 7 条诊断结果及下一版的可检验假设

两个诊断 actor 均完成 7/7 episode，无 traceback/OOM；两臂 14 条重跑的 episode ID 与完整轨迹 SHA-256 均和原官方 val **逐条完全一致**。第 7 条配对统计：

| 行为 | 空间基线 | v4 |
| --- | ---: | ---: |
| 到达成功所用步数 | 194 | 214 |
| 非锚点选点 | 0 | 8 |
| 长期目标切换 | 69 | 77 |
| 相邻目标 >2 m 跳点 | 42 | 43 |
| 碰撞 | 4 | 4 |
| 到访不同 25 cm 位置格 | 37 | 37 |

两臂最终坐标均为约 `(12.85 m, 10.05 m)`；首个目标分叉在第 118 步，首个实际位置分叉在第 161 步。v4 的 8 次非锚点选择发生在第 118、160、168、173–177 步，均选择候选 1；候选相对空间锚点的距离分别约 5.62、9.60、1.38、8.63–8.63 米。其空间分数劣势只有约 `3.6×10⁻⁶` 至 `6.28×10⁻⁵`，而树突残差相对锚点的优势约 `2.2×10⁻⁴` 至 `9.09×10⁻⁴`。首次触发来自可观测重访，第 160/168 步有局部碰撞历史证据，后期再次重访。**虽然 v4 对远候选已乘 1/16，极小的空间分差仍允许相距数米的目标翻转。**

这支持“局部目标来回改动造成额外路径”的机制：目标切换 +8，而碰撞、到访位置格和最终点没有变化。不过闭环在第 118 步后分叉，不能仅凭观测轨迹严格证明每个改点都导致 20 步延迟。现有图：[基线 Darden ep7](figures/diagnosis_val_v04/darden_ep7_baseline.png)、[v4 Darden ep7](figures/diagnosis_val_v04/darden_ep7_v4.png)；原始 `.npz` 和全部 14 条日志在 `diagnosis_val_v04_darden_ep7/`。

**下一版只提出假设，不把结果写成已完成：**固定远距衰减仍不足以抵抗近乎平坦的空间势函数。可在全新 v5 候选评分中引入显式、芯片友好的“改变长期目标的额外行程成本/排名间隔”，与当前空间分数和学习残差在同一尺度比较，而不是再次只提高事件门阈值或延长 v4 训练。它是候选决策代价，不是调用原 PONI 教师或运行时回退；必须先做单元测试、接线 smoke、有真实更新的小 pilot 和同 episode 配对开发集。因为这一机制已从官方 val 的 Darden 案例获得启发，若 v5 进入独立确认，**不能再把当前这 125 个已看过的官方 episode 当作未见验证集**；需使用尚未查看的后续官方 episode 区段或另一预先冻结的独立集合，明确记录 episode 唯一性，避免再次把并行重复当独立样本。v4 全套模型、训练与评估产物维持只读。
