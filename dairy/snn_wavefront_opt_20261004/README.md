# SNN 波前短期目标读出优化（2026-10-04）

## 问题、目标与固定项

Gibson 原严格 SNN 把八邻接图上的首次到达时间交给 FMM 环形读出器；MP3D 原严格 SNN 取路径的第五个格点作为短期目标。这些决策没有直接针对脉冲波前的前驱路径和真实地图分辨率。此次只改变“已有路径 → 短期目标”这一步，不训练网络，不改变 Sigma-D 检查点、波前代价、障碍膨胀、地图、语义目标、STOP/replan 条件或原评测入口。

正式验收固定同一 Sigma-D 权重、同一 episode cohort：

| 数据集 | 原规划器同权重基线 | 联合严格 SNN 旧版 | 必须同时达到的下限 |
|---|---|---|---|
| Gibson seed234，1000 episodes | FMM，Success 0.729，SPL 0.42174572 | Success 0.713，SPL 0.40489853 | Success ≥0.729 且 SPL ≥0.42174572 |
| MP3D seed123，2195 episodes | 加权 A*，Success 0.30523918，SPL 0.11818924 | Success 0.29931663，SPL 0.12177395 | Success ≥0.30523918 且 SPL ≥0.11818924 |

这些是结果验收门槛，不是对收益的保证。完整评测若有一项低于对应下限，应明确写“未通过”。

## 算法、接口与数据流

既有首次脉冲波前在八邻接网格上计算到达时间和前驱指针。设路径为 `p_0=start, p_1, ..., p_n=target`，地图分辨率为 `m=5 cm/cell`。新增读出使用原 `PlanNeuron(stg, stop, replan, path, info)` 接口；只改 `stg`，保留 `stop/replan/path` 与波前计数、路径成本。

1. `length`：计算前缀弧长 `L_k=m Σ_{i=1}^k ||p_i-p_{i-1}||₂`，取 `k* = max{k : L_k≤L}`，默认 `L=25 cm`。如路径只有一个格点，保持原 STG；一般非空路径至少给出第一段。
2. `visible`：取路径中最远的 `p_k`，使起点到 `p_k` 的 Bresenham 栅格射线所经格点均可通行。仍沿用原八邻接允许的对角角点约定；这个几何检查不代表对机器人连续碰撞风险的完整验证。

两种方案独立对照，不叠加。`length` 的 20、30 cm 只在默认25 cm 没达到小样本筛选门槛时追加。Gibson 不可达目标的投影和 MP3D 的无路径随机重选保持原逻辑，审计分开统计。

Gibson：RGB-D → 原语义分割/在线建图 → 固定 Sigma-D U-Net → 原严格 SNN 波前 → 新 STG 读出 → 原动作控制。Gibson 环境由 forkserver 子进程建立，读出模式在 `make_env_fn_opt_neuron` 中安装；不会只在主进程设置而实际继续走旧读出。

MP3D：RGB-D → 原 RedNet/在线建图 → 固定 `PFExpSigmaD` → 原严格加权 SNN 波前 → 新 STG 读出 → 原动作控制。MP3D 规划 worker 启动时安装同一进程内的读出模式。

对应新增模块：`poni/waypoint_opt_neuron.py` 实现纯路径读出和直线检查；`poni/planner_adapter_opt_neuron.py` 封装原严格规划器；`semexp/eval_poni_sigma_d_opt_neuron.py`、`hlab/eval_poni_sigma_d_opt_neuron.py` 是独立入口。Gibson/MP3D 的 worker 连接分别位于新增 `semexp/envs/habitat/sem_exp_opt_neuron.py`、`hlab/utils/planners_opt_neuron.py`。所有原 `.py` 保持原样。

本轮新增源码、启动器和诊断脚本的 SHA-256 列于 `artifact_sha256.txt`，便于复核运行版本。

## 本轮实验、命令和产物

工程根目录 `/mnt/jfzn/shensj/PONI`，环境：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=.:hlab:dependencies/astar_pycpp python scripts/validate_waveopt_neuron.py
```

已执行的一集链路冒烟：Gibson 五场景各 1 episode；MP3D part0 为 1 episode。产物分别位于 `gibson_length25_smoke1/` 和 `mp3d_length25_smoke1/`，启动器日志在本目录根层。两边的审计均已出现新读出 `readout_mode=length`、严格 `used=neuron` 和逐步位姿记录；冒烟只验证链路，不用于效果结论。

已启动的小范围评测：Gibson 两个读出各 5 场景×20=100 episodes，MP3D 两个读出各 11 个分片×10=110 episodes。命令如下，每个脚本拒绝覆盖已有 run 目录：

```bash
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_opt.sh \
  length 25 0 20 gibson_length25_pilot100 \
  > dairy/snn_wavefront_opt_20261004/gibson_length25_pilot100_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_opt.sh \
  visible 25 1 20 gibson_visible_pilot100 \
  > dairy/snn_wavefront_opt_20261004/gibson_visible_pilot100_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_opt.sh \
  length 25 10 pilot mp3d_length25_pilot110 \
  > dairy/snn_wavefront_opt_20261004/mp3d_length25_pilot110_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_opt.sh \
  visible 25 10 pilot mp3d_visible_pilot110 \
  > dairy/snn_wavefront_opt_20261004/mp3d_visible_pilot110_launcher.log 2>&1 &
```

MP3D 默认 25 cm 已完成且未通过小样本筛选，因此按预先规定补两个距离对照：

```bash
nohup bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_opt.sh \
  length 20 10 pilot mp3d_length20_pilot110 \
  > dairy/snn_wavefront_opt_20261004/mp3d_length20_pilot110_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_opt.sh \
  length 30 10 pilot mp3d_length30_pilot110 \
  > dairy/snn_wavefront_opt_20261004/mp3d_length30_pilot110_launcher.log 2>&1 &
```

Gibson 25 cm 同样未通过闸门，因此按计划补 20/30 cm 两组。它们不会影响已独立启动的可视捷径 1000 集全量评测：

```bash
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_opt.sh \
  length 20 4 20 gibson_length20_pilot100 \
  > dairy/snn_wavefront_opt_20261004/gibson_length20_pilot100_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_opt.sh \
  length 30 5 20 gibson_length30_pilot100 \
  > dairy/snn_wavefront_opt_20261004/gibson_length30_pilot100_launcher.log 2>&1 &
```

`run_gibson_opt.sh` 的第四个参数是每场景 episode 数，`launch_mp3d_opt.sh` 的第三个参数是每分片 episode 数；全量 MP3D 用 `0 full`，表示原来的 `TEST_EPISODE_COUNT=-1`。完整评测只对筛选通过的候选以全新标签启动，不复用小范围结果目录。Gibson 结果为 `<run>/gibson_objectnav/dump/seed_100/final_eval_stats.json`；MP3D 每分片为 `<run>/tb_seed_100_val_part_N/stats.json`，启动器生成 `exit_codes.tsv` 和 `summary_waveopt.json`。两种数据集另有 `audit*/*.jsonl`、采样地图快照 `*.npz`、`driver*.log`、可视化 `figures/*.png`。不保存新训练检查点；固定训练产物：

- Gibson：`dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt`
- MP3D：`dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`

原版对照及完整结果位于 `dairy/sigma_d_unet_20260924/navigation/`；原严格联合结果位于 `dairy/joint_sigma_d_strict_neuron_20261004/`。新 `scripts/summarize_waveopt_neuron.py` 从结果文件按 episode 配对，先检查 cohort 才给出差值和配对标准误。可视化脚本 `scripts/plot_waveopt_neuron.py` 输出地图叠图、前 40 步位姿轨迹和 Success 翻转图；图样例不能替代完整评测结论。

如果要**另开目录重跑原规划器**（以下是复现命令，本轮没有重新执行；切勿使用现存 run 标签，因为脚本会拒绝覆盖），Gibson 和 MP3D 分别用原仓库的入口：

```bash
CUDA_VISIBLE_DEVICES=0 bash experiment_scripts/gibson/eval_poni_sigma_d.sh \
  dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  gibson_seed234_best32k_repro
bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh \
  dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt \
  4 5 6 7 mp3d_seed123_best2k_4gpu_repro
```

两条命令分别用 FMM 和加权 A*；已完成的同协议基线原目录为 `navigation/gibson_seed234_best32k/` 和 `navigation/mp3d_seed123_best2k_4gpu/`。旧严格 SNN 不覆盖：可用本页后面的 `run_gibson_trace_neuron.sh` / `launch_mp3d_trace_neuron.sh` 在新目录独立复现，其规划函数仍是旧严格版本，只增加轨迹审计。

## 已完成验证与下一闸门

新增 Python 文件在远端 `poni` 环境通过语法检查；三个启动器通过 Bash 语法检查。`scripts/validate_waveopt_neuron.py` 在空地图、L 形走廊、窄门、斜向通道、不可达目标和起点已到目标等样例通过 27 项断言：两种读出均保持原波前路径、脉冲数、到达代价、STOP 与 replan，并检查了可视目标的栅格直线及 L 形拐角。Gibson/MP3D 一集冒烟完成并产出正式格式的统计文件；地图叠图及轨迹图均已成功生成。

小样本完成后：检查五场景/11分片数量和相同 episode cohort，全量审计 `fallback=0`、`reference=null`、`used=neuron`。对照旧严格 SNN，只有 Success/SPL 均不降且至少一项改善的候选可进全量评测；多个候选则按 SPL、Success、规划耗时依次选择。Gibson 与 MP3D 可分别采用不同模式。完整结果再和本节首表两个原规划器门槛比较，并记录成功翻转、配对标准误、没有改善的案例和 no-path 分布。

## 旧结果的针对性诊断（不涉及新实验的中途指标）

`scripts/diagnose_nopath_neuron.py` 解析之前已经完成的 Gibson Sigma-D+严格 SNN 全量审计，并与同 cohort Sigma-D+FMM 配对。统计和图位于 `diagnostics/gibson_nopath_repetition.json`、`diagnostics/gibson_nopath_repetition.png`、`diagnostics/gibson_nopath_outcomes.png`。共 144574 次规划，26451 次触发投影，涉及 105 个 episode。将这 105 个 episode 作为描述性分组，旧 SNN 的 Success/SPL 为 **0.22857/0.04504**，FMM 为 **0.42857/0.20267**；其余 895 个 episode，旧 SNN 为 **0.76983/0.44712**，FMM 为 **0.76425/0.44745**。这说明当前总体差距集中在“旧 SNN 曾触发投影”的轨迹，但不等于单独证明投影造成了全部差距。

在有投影的 105 个 episode 中，投影调用的中位数为 342；103 个 episode 的相邻投影调用有至少一半重复选择同一代理点。最极端案例 489 次投影只涉及 5 个代理点。重复目标与效率下降具有明确关联，需要在当前两个短期目标读出候选上观察是否改善；目前仍不追加复杂的目标重选协议。

## 已完成的第一项小样本结果

MP3D `mp3d_length25_pilot110/summary_waveopt.json`：11 个分片全部正常完成，110 个 episode ID 与旧严格 SNN 及同权重 A* 基线完全匹配。候选 Success/SPL 为 **0.36364/0.14721**；旧严格 SNN 在相同子集为 **0.37273/0.15156**，分别低 0.91、0.44 个百分点，因此**未通过**“同时不低于旧 SNN”的小样本闸门。A* 在相同子集为 0.34545/0.12454；与 A* 相比的正差异不能替代旧 SNN 闸门，也不等于完整 2195 episode 结论。其余候选结束后再决定是否进行全量评测。

MP3D `mp3d_visible_pilot110/summary_waveopt.json`：相同 110 个 episode ID 匹配。候选 Success/SPL 为 **0.35455/0.15632**；旧 SNN 为 **0.37273/0.15156**。SPL 提高 0.48 个百分点，但 Success 降低 1.82 个百分点，因此同样**未通过**双指标闸门。相对于同 cohort A* 的 0.34545/0.12454，二者虽均高，但只能视作该小样本的描述性数据。

MP3D 20 cm 和 30 cm 也已完成，均为相同 110 个 episode ID：20 cm 为 **0.35455/0.14820**，30 cm 为 **0.29091/0.11134**；旧 SNN 恒为 **0.37273/0.15156**。20 cm 两项均低 1.82、0.34 个百分点；30 cm 两项均低 8.18、4.02 个百分点。四个预定候选没有一个同时达到旧 SNN 双指标，故严格按小样本闸门**不启动 MP3D 全量评测**。这仅否定本轮简单 STG 读出候选，不说明 MP3D 波前 SNN 无法进一步优化。尤其 30 cm 恶化明显，说明增大前瞻并非普遍有利。

Gibson 两个 100 集候选与旧 SNN 和 FMM 的同 cohort `goal_distance` 顺序均匹配。旧 SNN 为 **0.78000/0.42269**，FMM 为 **0.68000/0.39904**。25 cm 读出为 **0.77000/0.37439**，相对旧 SNN 下降 1.00、4.83 个百分点，未通过闸门；可视捷径为 **0.83000/0.45626**，相对旧 SNN 上升 5.00、3.36 个百分点，通过小样本闸门。可视捷径相对旧 SNN 的配对成功差异标准误为 4.35 个百分点，SPL 差异标准误为 3.18 个百分点；其不确定性足以要求完整 1000 集复核。可视捷径审计共 12339 次规划，均为 `used=neuron`、`fallback=false`、`reference=null`，规划结果的 `candidate.info.readout_mode=visible`；没有把 FMM 用于正式动作。

Gibson 固定旧 SNN 的 no-path 分组，100 集中有 11 集曾触发旧投影。可视捷径在这 11 集的 Success/SPL 为 **0.45455/0.14844**，旧 SNN **0.36364/0.07152**；其余 89 集可视捷径 **0.87640/0.49430**，旧 SNN **0.83146/0.46610**。25 cm 在旧投影组虽改善到 **0.63636/0.21779**，其余组却降至 **0.78652/0.39374**，解释了整体失败。11 集分组太小，不能据此将改善归因于投影修复。`gibson_visible_pilot100/figures/` 已生成地图分歧、40 步轨迹、Success 翻转和固定旧投影分组对照图；`gibson_length25_pilot100/figures/` 同样保留变差案例图。

通过闸门的 Gibson 可视捷径已在 GPU 0 启动 5 场景×200=1000 episode 全量评测，仍用原 seed234 Sigma-D 权重、原 1000 episode 协议、严格 SNN，无新训练/检查点，产物隔离到 `gibson_visible_full1000/`。命令：

```bash
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_opt.sh \
  visible 25 0 200 gibson_visible_full1000 \
  > dairy/snn_wavefront_opt_20261004/gibson_visible_full1000_launcher.log 2>&1 &
PYTHONPATH=.:hlab:dependencies/astar_pycpp python scripts/summarize_waveopt_neuron.py \
  --dataset gibson --run-root dairy/snn_wavefront_opt_20261004/gibson_visible_full1000
```

第二条命令**只在** `gibson_visible_full1000/gibson_objectnav/dump/seed_100/final_eval_stats.json` 写成后执行；输出 `summary_waveopt.json`。评测尚在运行，不能用 pilot 结果宣称达到 72.90%/42.17% 全量门槛。

## 配对行为诊断和已发现的局限

新增独立 `scripts/analyze_waveopt_pilot_neuron.py`，输出每个候选的 `behavior_comparison.json` 和 `figures/paired_behavior.png`：逐 episode 的 Success 翻转、SPL 差值，以及审计里的转向/no-path 调用数；同时列出 SPL 下降和提高最多的 8 个案例。候选轨迹有逐步位姿，可统计“执行前进行动作后，下一步位移 <3 cm”的**前进行停滞代理**。它不是 Habitat 的真实碰撞计数：转弯、定位误差、运动受限等也会影响位移，且旧全量审计没有逐步位姿，绝不将代理量伪称碰撞率。

Gibson 可视捷径的 pilot 对旧 SNN Success 翻转为 **12 集变好、7 集变差**；每集平均转向 **63.89 vs 旧 74.31**，但转向动作占比只是 **45.35% vs 46.13%**，不能把转向次数下降全归功于更平滑的方向，episode 长度变化也会影响它。no-path 调用为 **22.61 vs 22.03 次/集**，并没有降低投影触发频率；改善更可能来自路径读出行为，而不是少触发投影。25 cm 候选为 15 集变好、16 集变差，SPL 总体下降。

MP3D 可视捷径的同 cohort 翻转为 **6 集变好、8 集变差**。最明显退步之一 `part5/episode5`（`X7HyMhZNoso_124`）从旧版成功变为失败，SPL 差 -0.796；该候选只有 10 次规划，所选可视路径点的前瞻距离中位数 56.6 cm、最大 265.7 cm，而 25 cm 候选在该案例的前瞻距离不超过 25 cm、运行 112 次规划。`part4/episode0` 的可视前瞻中位数 107.4 cm、最大 397.4 cm，同样从成功变失败。这些是关联而非因果证明：当前“最远可视点”可能对 MP3D 过于激进；下一轮值得测试受限可视读出，但本轮已达到预先限定的四个候选，不临时增加第五个参数组合。

`part5/episode5` 的可视候选和 30 cm 候选最后均在 `found_goal=true` 的预测状态下给出 STOP，离线评估仍判失败。这说明该案例不是运行时 `NoPathNeuron` 异常，而是新轨迹改变后在错误地点结束；语义预测、地图与终止条件也参与结果，不能仅凭一张路径图断言“射线撞墙”。相应事件序列保留在两个候选的 `audit_val_part_5/*.jsonl`。

严格模式全量审计（此处指六组 pilot 的每次规划，不是 1000/2195 集全量导航）已逐条核查：Gibson 25 cm/可视分别 15313/12339 次规划；MP3D 25 cm/可视/20 cm/30 cm 分别 32813/30744/31398/31772 次规划。所有记录均 `used=neuron`、`fallback=false`、`reference=null`、`error=null`、`readout_mode` 与配置相符，`plan_failure=0`。这只证明实现没有悄悄回退原规划器，不证明效果达标。

为了补齐旧版缺失的逐步位姿而不覆盖旧 `.py`，新增仅用于诊断重跑的 `sem_exp_trace_neuron.py`、`planners_trace_neuron.py` 和独立评估入口/启动器。它们调用原严格 SNN 规划函数，只附加 `episode_start/pose` 审计；结果必须先与历史旧 SNN 的逐 episode 成绩完全核对，才可作为配对轨迹证据。当前一集 smoke 正在运行；若复现不一致，不用于效果对照。可复现命令：

```bash
nohup bash experiment_scripts/snn_wavefront_opt_20261004/run_gibson_trace_neuron.sh \
  1 20 gibson_old_trace_pilot100 \
  > dairy/snn_wavefront_opt_20261004/gibson_old_trace_pilot100_launcher.log 2>&1 &
nohup bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_trace_neuron.sh \
  10 mp3d_old_trace_pilot110 \
  > dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110_launcher.log 2>&1 &
```

Gibson 5 场景各 1 集、MP3D part0 1 集的旧版追踪 smoke 均正常产出 `plan/action/pose`，没有切换读出。Gibson 5 集的 Success、SPL、goal_distance 与旧严格 SNN 全量前一集逐项一致；MP3D 的 `2azQ1b91cZZ_149` 与旧严格 SNN 的 Success/SPL 一致。已按上面两条命令启动旧版追踪 pilot，待结束后必须继续检查全部 100/110 集是否逐项复现，才把转向与前进行停滞代理做配对。真实碰撞计数仍需单独加环境指标，不应从现有审计硬推。

## 完整评测结束后的结论与验收

Gibson 可视捷径的 **1000/1000 episodes 已完成，双指标通过**。`gibson_visible_full1000/summary_waveopt.json` 与同权重、同 seed/episode 顺序的两个历史结果配对，`goal_distance` 顺序一致：

| Gibson 1000 集 | Success | SPL | 相对可视捷径的配对差值 |
|---|---:|---:|---|
| Sigma-D + FMM（原规划器） | 72.90% | 42.1746% | 可视捷径 +1.80 / +0.7474 个百分点 |
| Sigma-D + 旧严格 SNN | 71.30% | 40.4899% | 可视捷径 +3.40 / +2.4321 个百分点 |
| **Sigma-D + 严格 SNN + 可视捷径** | **74.70%** | **42.9219%** | **两项均过 FMM 验收门槛** |

相对 FMM，Success 的配对标准误为 **1.4829 个百分点**、SPL 为 **0.9838 个百分点**；数值上过门槛，但不能据此夸大为统计上稳健地优于 FMM。相对旧严格 SNN，Success/SPL 的配对标准误为 **1.4801/1.0151 个百分点**，Success 翻转 **127 集改善、93 集退步**。这说明提升不是每集单调改善。正式结果文件：`gibson_visible_full1000/gibson_objectnav/dump/seed_100/final_eval_stats.json`；汇总：`gibson_visible_full1000/summary_waveopt.json`；审计：`gibson_visible_full1000/audit/*.jsonl`。

全量审计逐条核对 **139435 次规划**：全部 `used=neuron`、`fallback=false`、`reference=null`、`error=null`、`readout_mode=visible`；没有以原 FMM 偷换动作。相对旧严格 SNN 的 26451 次投影，候选记录了 **20008 次** no-path 投影，涉及 **83 集**（旧版 105 集）；相邻投影仍有约 **84.18%** 代理点重复，所以可视读出改善了整体导航，但并没有从算法上消除重复代理问题。后者仍是单独的后续优化课题。

在**固定旧 SNN 曾投影的 105 集**内，可视捷径 Success/SPL **48.57%/18.92%**，旧 SNN **22.86%/4.50%**，FMM **42.86%/20.27%**；其余 895 集，可视捷径 **77.77%/45.74%**，旧 SNN **76.98%/44.71%**，FMM **76.42%/44.74%**。这显示总体改善主要集中在旧版困难组，同时普通组也没有整体退步；分组是在旧版轨迹上定义，不能当作因果证明。图：`gibson_visible_full1000/figures/old_repair_cohort_comparison.png`、`paired_behavior.png`、`success_flips.png`。

Gibson 20/30 cm 补充对照也在相同 100 集完成：20 cm **69.00%/35.26%**，30 cm **73.00%/36.52%**；旧 SNN 同子集 **78.00%/42.27%**，两者都未通过小样本闸门，不做全量。结合早先 25 cm 的 **77.00%/37.44%**，说明简单固定长度读出在 Gibson 上不如可视捷径。四个预先限定候选均已评估，没有事后追加第五个版本。

MP3D 四候选均未通过 110 集旧 SNN 双指标闸门，故**没有 2195 集新版本结果，也不能宣称达到 MP3D 的 30.52%/11.82% 全量验收目标**。下一处最小改动由失败轨迹指向“限制最远可视点的物理前瞻”，但本轮不超出预设四候选范围。

旧严格 SNN 追踪重跑：MP3D 110 集的 episode ID、Success、SPL、distance_to_goal 与历史结果**逐集完全一致**，可用于 `scripts/analyze_waveopt_pilot_neuron.py --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110` 的转向/no-path/前进行停滞代理配对。Gibson 100 集 Success 与旧版逐集一致，SPL **99/100 集一致**；仅 `Collierville/episode18` 从历史 **0.77724927** 变为 **0.78746121**，差 0.01021193。故 Gibson 的严格行动计数对比仍使用历史审计，追踪重跑只作定性轨迹参考，不把那一集当成完全复现。`mp3d_old_trace_pilot110/summary_waveopt.json`、`gibson_old_trace_pilot100/summary_waveopt.json` 保存复核。

MP3D 可视 pilot 的配对审计（旧追踪完全复现后）显示每集平均转向 **138.61 vs 旧 145.62**，no-path 调用 **5.09 vs 6.19**，但前进行停滞代理 **3.42 vs 2.41**。因此少转向、少 no-path 不能保证 Success 不下降；过远的短期目标可能增加局部控制问题，不过停滞代理不是真实碰撞计数。MP3D 四候选与 Gibson 两个补充候选均有各自 `behavior_comparison.json` 和 `figures/paired_behavior.png`，列出变好/变差案例，便于下轮定点诊断。

最后复核：Gibson 20/30 cm 新增审计分别 17151/15541 次规划，均无非 SNN、FMM reference、fallback、异常或错误读出模式；全部新增 23 个源码/脚本与 `artifact_sha256.txt` 的远端 SHA-256 校验 **23/23 匹配**。远端 `dairy/snn_wavefront_opt_20261004/` 是完整产物，当前工作区的 `result_mirror/` 仅复制关键 JSON，`review_figures/` 仅复制关键 PNG；大体量逐步审计、地图快照、driver 日志仍保留在远端各 run 目录。

新增 `scripts/plot_paired_trace_neuron.py`，在相同 episode 上叠画旧 SNN 与候选前 40 步位姿，各选一例 SPL 改善/退步，输出 `<run>/figures/paired_trajectory_{improved,worsened}.png` 与 `paired_trajectory_examples.json`。Gibson 可视 pilot 的案例为 `Collierville/2`（+1.000）和 `Wiconisco/3`（-0.951）；MP3D 可视 pilot 为 `part4/episode4`（+0.806）和 `part4/episode0`（-0.619）。这四例不包含 Gibson 重跑中那一例非精确复现。图中只画前 40 步，标题 SPL 差是**整集最终指标**，不能把前 40 步当成整个 episode。启动方式：

```bash
PYTHONPATH=.:hlab:dependencies/astar_pycpp python scripts/plot_paired_trace_neuron.py \
  --dataset gibson --run-root dairy/snn_wavefront_opt_20261004/gibson_visible_pilot100 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/gibson_old_trace_pilot100 --steps 40
PYTHONPATH=.:hlab:dependencies/astar_pycpp python scripts/plot_paired_trace_neuron.py \
  --dataset mp3d --run-root dairy/snn_wavefront_opt_20261004/mp3d_visible_pilot110 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 --steps 40
```

完成状态复核：Gibson 四个候选/旧追踪均写出期望的 100 集 pilot 文件，可视捷径全量文件含 1000 集，`driver.log` 无 Python traceback；MP3D 四候选及旧追踪均有 11/11 分片 `stats.json` 且 `exit_codes.tsv` 为 11 个零退出码。聚焦验证再次得到 `WAVEOPT_VALIDATION_OK: 27 focused checks`。

新增 `scripts/compare_gibson_waveopt_subgroups.py` 会固定用旧 SNN 全量审计里定义的“曾经 no-path 投影”episode 分组，在候选完成后并排比较候选、旧 SNN、FMM 的 Success/SPL；输出 `<run>/subgroup_comparison.json` 和 `figures/old_repair_cohort_comparison.png`。固定旧分组避免把候选轨迹改变后才出现的投影事件用作比较分组。执行方式：

```bash
PYTHONPATH=.:hlab:dependencies/astar_pycpp python scripts/compare_gibson_waveopt_subgroups.py \
  --run-root dairy/snn_wavefront_opt_20261004/gibson_length25_pilot100
```
