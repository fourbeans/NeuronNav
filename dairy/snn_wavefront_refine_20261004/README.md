# MP3D 严格 SNN 波前：有限可视路径读出实验（2026-10-04）

## 目的与已知问题

冻结 MP3D Sigma-D U-Net 的 `best.ckpt`、语义地图、加权 SNN 波前传播、障碍膨胀、STOP、no-path 处理、评测数据和随机种子，仅改“前驱路径 → 短期目标”读出。旧 MP3D 严格 SNN 固定取第五个路径格点；上一轮无距离上限的可视捷径虽提高 SPL 小样本值，但 Success 降低，且可能一次指向 1–4 m 外的点。上一轮 110-episode 配对小样本：旧严格 SNN Success 37.2727%、SPL 15.15596%；25 cm 路径长度读出 36.3636%/14.7205%；无遮挡可视捷径 35.4545%/15.6317%。二者均未达到双指标门槛。因此本轮不叠加复杂网络，优先限制捷径的物理长度，并检验是否需要保留路径的近障加权代价。

## 接口与数据流

`hlab/eval_poni_sigma_d_mp3d_refine_neuron.py` 注册同一个 `PFExpSigmaD` 和同一个评估入口；`hlab/utils/planners_mp3d_refine_neuron.py` 只在本实验 worker 内安装 `mp3d_plan_refine`；`poni/planner_adapter_mp3d_refine_neuron.py` 先调用未改动的 `mp3d_plan_strict`，然后从 `plan.path` 选新 `stg`。输出仍是 `PlanNeuron(stg, stop, replan, path, info)`，`stop/replan/path/arrival_cost/path_cost/spikes` 保持不变。没有加 A* 或原规划器兜底。原始实验入口与 `.py` 不覆盖。

对于路径 `p_0,...,p_n`，锚点是旧读出 `p_min(4,n)`。只考虑其后的路径顶点 `p_j`，并要求路径弧长 `L_j=Δ·Σ_{i=1}^j ||p_i-p_{i-1}||_2 ≤ C`，其中地图分辨率 `Δ=5 cm`，上限 `C=50 cm`。从远到近选择第一条 Bresenham 直线全在可通行栅格内的候选。可选代价守卫还要求 `R_j = Σ_{(a,b)∈ray} ||b-a||_2 w_b / Σ_{i=1}^j ||p_i-p_{i-1}||_2 w_{p_i} ≤ 1.02`。`w` 与原 MP3D 波前一致，来自膨胀障碍图的 `proximity_weights(wscale=4,niters=1)`；守卫避免直线虽不撞墙、却贴障碍边缘的情形。没有满足条件时仍取旧锚点。评测参数分两版：A=`C=50 cm, guard=0`，B=`C=50 cm, guard=1`。二者必须在相同 110 个 episode 上配对比较。

## 代码与验证

新增文件：

- `poni/waypoint_mp3d_refine_neuron.py`：局部 Bresenham、限距、可选近障加权守卫。
- `poni/planner_adapter_mp3d_refine_neuron.py`：复用严格 SNN 计划，仅更新 STG，并记录读出诊断字段。
- `hlab/utils/planners_mp3d_refine_neuron.py`：实验专用 worker 安装与位姿审计。
- `hlab/eval_poni_sigma_d_mp3d_refine_neuron.py`：独立评测入口。
- `experiment_scripts/snn_wavefront_refine_20261004/*.sh`：分片与并行启动。
- `scripts/validate_mp3d_refine_neuron.py`：合成路径与规划器不变量检查。

已完成：Python 语法检查、Bash 语法检查；在 node1 的 `poni` 环境运行合成检查，输出 `MP3D_REFINE_UNIT_OK`。测试涵盖上限内/外路径点、障碍阻挡、近障加权代价守卫、短路径、到达代价/路径/STOP 保持、no-path 保持。第一次测试的 30 cm 预期写错：第五格点后一个斜向格点距起点约 27.07 cm，故应被 30 cm 上限允许；更正测试断言后通过，算法代码未因此改动。实时审计的首条计划可见 `readout_mode=bounded_visible_cost`、`fallback=false`，确认实验入口生效。

## 启动命令、数据与产物

服务器：`ssh xty-node1-direct`；工程根目录 `/mnt/jfzn/shensj/PONI`；环境 `source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh && conda activate poni`。权重固定为 `dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`。数据为 `data/datasets/objectnav/mp3d/v1/val_parts`，`TASK_CONFIG.SEED=100`，11 分片各前 10 episode。此轮无训练、无新 checkpoint。

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_refine_20261004/launch_mp3d_refine_neuron.sh 50 0 10 pilot_a mp3d_cap50_no_guard_pilot110
bash experiment_scripts/snn_wavefront_refine_20261004/launch_mp3d_refine_neuron.sh 50 1 10 pilot_b mp3d_cap50_cost_guard_pilot110
```

以上两条已用 `nohup` 在 node1 后台启动；启动日志分别为 `dairy/snn_wavefront_refine_20261004/launcher_cap50_no_guard.log` 和 `launcher_cap50_cost_guard.log`。每版独立目录 `dairy/snn_wavefront_refine_20261004/<label>/`，含 `config.txt`、`processes.tsv`、`exit_codes.tsv`、各分片 `driver_*.log`、`logs_seed_*.txt`、`tb_seed_100_val_part_*/stats.json`、`audit_val_part_*/*.jsonl` 和抽样 `*.npz` 地图快照、`summary_waveopt.json`。完整评测只有通过下述小样本闸门才可启动，此时把第三个参数改为 `0`、GPU 配置改为 `full`、用新的目录标签；**不要复用已有目录**。

旧严格 SNN：`dairy/joint_sigma_d_strict_neuron_20261004/mp3d_seed123_best2k`；旧同样本轨迹审计：`dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110`；原 Sigma-D + 加权 A*：`dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu`。新旧结果的统计脚本按 episode ID 配对，不能仅按平均数推断同一批样本。

## 决策闸门与待办

小样本必须同时满足 Success ≥ 37.2727%、SPL ≥ 15.15596%，且至少一项严格提高；否则不能扩大到完整 2195 episode。通过者按 SPL、Success、耗时顺序择优。完整集还须分别达到原加权 A* 的 Success ≥ 30.523918%、SPL ≥ 11.818924%；否则明确报告未通过。小样本数据带有较大方差，不等于正式结论。

实验结束后运行配对轨迹诊断，统计成功翻转、SPL、转向、no-path、位姿前进停滞代理指标，绘制成功/失败双向案例。碰撞数没有直接传感器指标时不能把停滞代理称作碰撞。若 A/B 都失败，先定位失败案例再决定下一项简单改动，不盲目上全量。

## 三版实际结果与分析（每版 110 集，全部完成）

三版各有 11/11 分片 `stats.json`，`summary_waveopt.json` 按相同 episode ID 对比旧严格 SNN（Success 0.37272727，SPL 0.15155956）。

| 候选 | Success | SPL | 相对旧 SNN 的成功翻转 | 结论 |
|---|---:|---:|---:|---|
| 50 cm、无代价守卫 | 0.29090909 | 0.11489423 | 5 集变好、14 集变差 | 双指标未通过 |
| 50 cm、有代价守卫 | 0.30909091 | 0.13477287 | 3 集变好、10 集变差 | 双指标未通过 |
| 50 cm、代价守卫、方向偏差≤15° | 0.33636364 | 0.13381615 | 6 集变好、10 集变差 | 双指标未通过 |

无代价守卫版在 31889 次规划中 30662 次（96.15%）改了 STG；所选路径弧长均值 46.60 cm，捷径加权代价 / 原路径前缀加权代价均值 1.371，最大 3.689。这意味着“未撞障碍”不等于“代价合理”。守卫版在 31470 次规划中 25079 次（79.69%）改了 STG；所选弧长均值 45.95 cm，加权代价比最大 1.018，说明守卫按设计限制了近障捷径，但改动频率仍高。两版审计 `fallback_calls=0`。

两版相对旧 SNN 的平均每集转向次数分别为 129.33、136.57（旧 145.62），但 Success/SPL 反而下降；少转向不等于导航更好。前进停滞**代理**分别为 3.40、2.20（旧 2.41），不是传感器真实碰撞计数。地图快照抽样中，读出引起的即时动作分歧分别为 81/319、49/320。守卫版最明显退步例 `part9/episode4`：旧版 43 步成功，新版 500 步失败，SPL 差 -0.7096；改善例 `part4/episode7` 的 SPL 差 +0.5624。`figures/map_*.png` 和 `figures/paired_trajectory_{improved,worsened}.png` 保留双向案例；轨迹图只显示前 40 步，标题为整集最终 SPL 差，不能把两者混同。这些现象支持“频繁把短期目标推远可能破坏局部转向”这一待检验解释，尚不能证明唯一因果。

因此两版**都不进入 2195 集完整评测**。结果目录内有 `behavior_comparison.json`、`readout_usage.json`、`figures_manifest.json`、`figures/paired_behavior.png`、`figures/success_flips.png`、`figures/trajectory_40_steps.png`、地图图和正负轨迹图。

## 第二轮单因素收紧：方向偏差 ≤15°

保留第一轮更好的 50 cm + 代价守卫，只新增一条局部条件：候选方向与旧第五格点方向的夹角

`θ_j = arccos( ((p_4-p_0)·(p_j-p_0)) / (||p_4-p_0|| ||p_j-p_0||) ) ≤ 15°`。

若不满足，就从更近的路径候选继续查找；都不满足则保留旧 STG。该约束不增加地图重复脉冲时步、不引入神经网络，也不调用 A*。15° 是在看过第一轮失败轨迹后确定的一次前瞻性候选，因此仍只作为 pilot；不能根据这一批数据宣称泛化性能。原两个命令仍可运行（不传角度参数即完全保留第一轮行为）。修改前的源码哈希保存在 `code_sha256.txt`；第二轮新哈希另存 `code_sha256_phase2.txt`，保证可区分。

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_refine_20261004/launch_mp3d_refine_neuron.sh \
  50 1 10 full mp3d_cap50_cost_angle15_pilot110 15
```

该命令已于 2026-10-05 00:06 左右在 node1 后台启动，启动日志 `dairy/snn_wavefront_refine_20261004/launcher_cap50_cost_angle15.log`，独立产物目录 `mp3d_cap50_cost_angle15_pilot110/`。入口日志确认 `cost_guard=1, angle_deg=15.0`。合成测试重新运行得到 `MP3D_REFINE_UNIT_OK`，其中新增断言：转角超过阈值的远点被拒绝，较近的允许点被选中。第二轮仍须先过旧 SNN 110 集双指标门槛才可考虑全量评测。

第二轮现已完成 11/11 分片，`exit_codes.tsv` 全部 0。按相同 110 个 episode ID，Success **0.33636364**、SPL **0.13381615**，相对旧严格 SNN 0.37272727/0.15155956 分别低 3.64、1.77 个百分点，**未通过小样本闸门，不启动 2195 集全量**。31248 次规划里 24971 次（79.91%）仍改变 STG，弧长均值 45.65 cm、最大 50 cm；近障代价比最大约 1.0。`fallback_calls=0`、`invalid_strict_calls=0`、`plan_failures=0`。平均每集转向 136.39（旧 145.62），no-path 6.24（旧 6.19），前进行停滞代理 2.06（旧 2.41）；转向与停滞下降却仍未提高导航 Success/SPL。

`figures_manifest.json` 统计抽样地图即时动作分歧为 48/321，几乎没比未加角度限制的代价守卫版 49/320 少，说明 15° 约束没有真正压住会改变离散动作的读出。`behavior_comparison.json` 保留 6 集变好、10 集变差；最佳/最差可视轨迹仍分别为 `part4/episode7`（SPL +0.5624）和 `part9/episode4`（SPL -0.7096），对应地图图和正负轨迹图在 `figures/`。避免继续在同一 110 集上盲调 10°、5° 等参数；若要提高 MP3D，应先对差异路径/目标选择和失败 episode 做成因定位，再预注册下一条简单规则。当前三版读出实验不能宣称 MP3D 达到原 A* 的全量验收要求。

## 补充定位：旧 SNN 与原加权 A* 的局部决策

独立只读脚本 `scripts/compare_mp3d_astar_stg_neuron.py` 对旧严格 SNN 追踪的冻结地图快照同时离线计算两种规划器，不把 A* 接入 SNN 运行时。输出 `old_snn_astar_snapshot_agreement.json`：330 个抽样快照中 326 个可比，4 个因无可比路径等情况跳过；326 张里 186 张（57.06%）STG 完全一致，263 张（80.67%）按同一朝向/转角离散为相同动作。它说明原版和 SNN 的局部读出确有差异，却不能证明哪种动作带来更高整集成功率；旧 SNN 在这 110 集本身比 A* 成功更多，因此不应机械模仿 A*。第二轮 pilot 仍以旧严格 SNN 作门槛。

## 独立复现与分析命令

原 Sigma-D + 加权 A* 的完整基线复现命令（本轮**未重跑**；给新的标签，避免覆盖历史结果）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh \
  dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt \
  4 5 6 7 mp3d_seed123_best2k_4gpu_repro_newlabel
```

旧严格 SNN 波前的同批 110 集追踪基线复现命令（本轮也未重跑；历史版本 `mp3d_old_trace_pilot110` 已验证与旧结果逐 episode 一致）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_trace_neuron.sh \
  10 mp3d_old_trace_pilot110_repro_newlabel
```

新候选完成后运行（`<label>` 和角度参数按具体版本替换）：

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/summarize_waveopt_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_refine_20261004/<label>
python scripts/analyze_waveopt_pilot_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_refine_20261004/<label> \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
python scripts/analyze_mp3d_refine_readout_neuron.py \
  --run-root dairy/snn_wavefront_refine_20261004/<label>
python scripts/plot_mp3d_refine_neuron.py \
  --run-root dairy/snn_wavefront_refine_20261004/<label> \
  --cap-cm 50 --cost-guard 1 --angle-deg 15 --max-maps 4
python scripts/plot_paired_trace_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_refine_20261004/<label> \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 --steps 40
```

评测启动器已自动在各自目录生成 `summary_waveopt.json`。本地 `dairy/snn_wavefront_refine_20261004/result_mirror/` 镜像关键 JSON/PNG；服务器原目录保留完整审计与地图快照，不复制或删除历史运行。

## 第三阶段：近障权重边界修正与独立复测（2026-10-05）

检查原 `poni/planner_adapter_neuron.py:mp3d_plan` 后发现：原 MP3D 在计算近障权重前用 `add_boundary(traversible)`，默认填充值为 **1（可通行）**。本实验最初重建代价守卫的权重时用了 `np.pad(free,1,constant_values=False)`，等同于把边界视为障碍。这是实现不一致，不应掩盖。无守卫版不读取该权重，不受影响；带守卫的两个旧目录保留，但不再被称为“边界完全一致”的版本。

仅将新实验模块 `poni/planner_adapter_mp3d_refine_neuron.py` 的权重重建改成自由边界，并拆出 `_weights_for_ray` 便于单测。原 PONI、严格 SNN 波前和训练权重均未改。新增边缘测试：全自由地图所有权重应为 1；左上角设置障碍后相邻格权重为 4，远格仍为 1。完整合成测试在 node1 输出 `MP3D_REFINE_UNIT_OK`。第二阶段代码哈希保留在 `code_sha256_phase2.txt`，修正后哈希单独记录在 `code_sha256_phase3.txt`，不能混为同一实现。

用独立目录重新运行同样的 11×10 集、相同 seed/权重/分片：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_refine_20261004/launch_mp3d_refine_neuron.sh \
  50 1 10 pilot_a mp3d_cap50_cost_guard_edgefix_pilot110
bash experiment_scripts/snn_wavefront_refine_20261004/launch_mp3d_refine_neuron.sh \
  50 1 10 pilot_b mp3d_cap50_cost_angle15_edgefix_pilot110 15
```

两个修正版都有 11/11 分片正常结束、cohort 与旧严格 SNN 匹配、`fallback_calls=0`、`invalid_strict_calls=0`、`plan_failures=0`。各自的 11 个 `stats.json` 与对应修正前目录**逐文件字节相同**；`scripts/compare_mp3d_refine_replays_neuron.py` 对每分片审计逐条比较 episode 标识、STG、STOP/replan、规划动作及实际 action，也得到 11/11 完全一致。故边界修正对此 110 集的导航行为无影响，原结论保持：修正版无角度守卫 **0.30909091/0.13477287**，修正版 15° **0.33636364/0.13381615**，均未通过旧 SNN **0.37272727/0.15155956** 门槛，不启动 MP3D 全量。不要把“本批无影响”外推到所有边界地图。

原旧实现与修正后行为一致的可重复检查：

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/compare_mp3d_refine_replays_neuron.py \
  --before dairy/snn_wavefront_refine_20261004/mp3d_cap50_cost_guard_pilot110 \
  --after dairy/snn_wavefront_refine_20261004/mp3d_cap50_cost_guard_edgefix_pilot110
python scripts/compare_mp3d_refine_replays_neuron.py \
  --before dairy/snn_wavefront_refine_20261004/mp3d_cap50_cost_angle15_pilot110 \
  --after dairy/snn_wavefront_refine_20261004/mp3d_cap50_cost_angle15_edgefix_pilot110
```

## 下一处应分析的机制：等代价路径的选择

`scripts/diagnose_mp3d_path_gap_neuron.py` 对旧 SNN 追踪的冻结地图离线重建原加权 A* 与 SNN 的完整路径及加权代价；输出 `old_snn_astar_path_gap.json`。326 张可比快照中，186 张 STG 相同、271 张终点目标格相同、273 张加权路径代价在 `1e-4` 内相同；140 张 STG 不同的快照里有 **116 张仍是等代价路径**。在全部 326 张里，A* 路径代价高于 SNN 有 53 张，低于 SNN 为 0 张。该诊断支持将下一轮焦点从“加大前瞻”移到“单次脉冲波前在等到达代价时，如何简单且稳定地选择前驱路径”；它不证明哪种平局规则会提高 Success。下一轮应先离线筛选极简 tie-break，再预注册一个新 110 集导航 pilot，仍以同批旧 SNN Success/SPL 双指标作闸门，不能直接做 2195 集。

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/diagnose_mp3d_path_gap_neuron.py \
  --snapshot-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 \
  --output dairy/snn_wavefront_refine_20261004/old_snn_astar_path_gap.json
```
