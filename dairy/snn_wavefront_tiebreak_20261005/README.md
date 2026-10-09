# MP3D SNN 波前：等到达代价的低转弯前驱选择

## 实验目的与依据

前一轮 `dairy/snn_wavefront_refine_20261004/` 的 50 cm 可视路径读出、近障代价守卫、15° 方向守卫均未达到旧严格 SNN 的 MP3D 110 集双指标门槛；边界权重修正后的复测也逐规划保持同样动作。停止继续堆读出参数，改查 SNN 波前本身的前驱平局。冻结地图的离线诊断 `old_snn_astar_path_gap.json`：326 张可比地图里，SNN 与加权 A* 的 STG 有 140 张不同，其中 116 张两者最终路径总代价仍相同（`1e-4` 容差）。原 SNN 的单次脉冲传播和加权到达时间已能得到不高于 A* 的路径代价；下一处简单的自由度是**相同代价时怎样选路径**，而不是新网络或更远的短期目标。

本实验假设：在等到达代价的前驱之间，选使当前路径方向尽量连续的前驱，可以减少栅格路径的任意折返，并可能改善局部动作稳定性。它仍是待验证假设，离线转弯更少不等于整集 Success/SPL 更高。

## 模块接口、算法与数据流

原传感器、RedNet/在线建图、固定的 `PFExpSigmaD`、相同加权 SNN 波前 `poni.wavefront_neuron.propagate`、目标图、障碍膨胀、代价权重、5 cm 地图、STOP 与 no-path 语义保持不变。只新建：

- `poni/path_tiebreak_neuron.py`：在波前已有到达时间图上作局部等代价回溯。
- `poni/planner_adapter_mp3d_tiebreak_neuron.py`：独立 MP3D 规划器，建立与旧严格版本同一张加权自由图，保持原 STOP/no-path，输出新路径第五格点。
- `hlab/utils/planners_mp3d_tiebreak_neuron.py`：独立 worker/审计安装，不覆盖已有 worker。
- `hlab/eval_poni_sigma_d_mp3d_tiebreak_neuron.py`：独立评估入口，不覆盖原评估器。
- `experiment_scripts/snn_wavefront_tiebreak_20261005/*.sh`：分片和并行启动。
- `scripts/validate_path_tiebreak_neuron.py`、`scripts/validate_mp3d_tiebreak_neuron.py`、`scripts/screen_mp3d_tiebreak_neuron.py`：合成与冻结地图验证。

设波前从起点 `s` 发出，`T(v)` 为格点 `v` 的首次发放/到达时间，`w(v)` 为近障目的格权重。只允许前驱 `u` 满足 Bellman 等式：

`|T(u) + ||v-u||_2·w(v) - T(v)| ≤ 1e-8 + 1e-10|T(v)|`。

沿目标到起点回溯，每格最多看 8 个邻格；目标点先保留原前驱，随后在等代价候选里最小化该格的转角，转角相同时优先原前驱。传播并不重复 T 次，也不额外跑反向波；代价主排序完全由原 SNN 到达时间决定。返回路径、第五格点 STG；STOP 沿原路径按 `len<5 且几何路径≤0.25m` 判定，以隔离路径平局读出对 STOP 的影响。若目标不可达，委托旧严格 SNN 的原 no-path 随机 replan，**不调用 A***。在每次规划里断言新旧路径加权总代价相差不超过 `1e-5`，否则直接报错，不能静默使用新路径。

芯片意义上仅增加回溯时的局部邻居比较，不增加脉冲传播轮数、神经网络或全图矩阵状态；这不是硬件加速证明。它会改变部分 STG/动作，效果仍须看导航评测。

## 训练、数据、权重与目录

本轮**不训练**，无新 checkpoint。MP3D Sigma-D seed123 固定权重：`dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`，SHA256 `eb6b4b55a15f808088b5187b4cd5d4d2fac3394aa215d6c2a5df9325532a2c9b`。数据集 `data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz`，11 个 val part，每分片前 10 集，`TASK_CONFIG.SEED=100`，与旧严格 SNN pilot 同 cohort。全部新产物位于 `/mnt/jfzn/shensj/PONI/dairy/snn_wavefront_tiebreak_20261005/`；每版独立目录，含 `config.txt`、`processes.tsv`、`exit_codes.tsv`、`driver_val_part_*.log`、`logs_seed_*.txt`、`tb_seed_100_val_part_*/stats.json`、`audit_val_part_*/*.jsonl`、抽样地图 `*.npz`、`summary_waveopt.json`。无新训练日志与权重文件。

## 验证记录与启动命令

Python/Bash 语法检查通过。在 node1 `poni` 环境：`scripts/validate_path_tiebreak_neuron.py` 输出 `PATH_TIEBREAK_UNIT_OK`；`scripts/validate_mp3d_tiebreak_neuron.py` 输出 `MP3D_TIEBREAK_UNIT_OK`，覆盖空地图、近目标 STOP、障碍和无路径重规划，检查到达代价、路径代价、脉冲数及 STOP 不变量。冻结旧 SNN pilot 的 330 个快照，327 个可用于本算法，3 个不可达等情况跳过；新回溯在 294 张减少折弯、1 张增加折弯，平均折弯数差 -16.06；131 张 STG 不同、61 张即时动作不同，最大加权代价差 `7.96e-13`。输出 `dairy/snn_wavefront_refine_20261004/mp3d_tiebreak_offline_screen.json`；这只是离线算法筛查，不是效果证据。

真实链路一集冒烟命令及产物：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_tiebreak_20261005/run_mp3d_tiebreak_part_neuron.sh \
  0 0 1 /mnt/jfzn/shensj/PONI/dairy/snn_wavefront_tiebreak_20261005/smoke_part0_one
```

已完成，`smoke_part0_one/tb_seed_100_val_part_0/stats.json` 存在；审计首条记录 `readout_mode=equal_arrival_straight_tie`、`used=neuron`、`fallback=false`，`plan_failure=0`。

已启动的 110 集正式小样本命令（服务器后台，启动日志 `launcher_pilot110.log`）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_tiebreak_20261005/launch_mp3d_tiebreak_neuron.sh \
  10 pilot mp3d_tiebreak_pilot110
```

固定对照：旧严格 SNN `dairy/joint_sigma_d_strict_neuron_20261004/mp3d_seed123_best2k`、同批逐步轨迹 `dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110`；原 Sigma-D+加权 A* `dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu`。旧 SNN 同批 Success **0.37272727**、SPL **0.15155956**。只有候选两项同时不低于旧 SNN、至少一项严格提升，才允许启动独立全量目录；全量 2195 集再检验原 A* 的 Success≥0.30523918、SPL≥0.11818924。若小样本不达标，**不启动全量**，保留失败图和审计，不声称效果优于 PONI。

原 PONI MP3D 加权 A* 完整复现命令（本轮不重跑，标签必须新建）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh \
  dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt \
  4 5 6 7 mp3d_seed123_best2k_4gpu_repro_newlabel
```

旧严格 SNN 的独立追踪复现命令（本轮不重跑）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_trace_neuron.sh \
  10 mp3d_old_trace_pilot110_repro_newlabel
```

## 110 集导航结果与筛选结论（已完成）

`mp3d_tiebreak_pilot110/` 已产出 11/11 个分片 `stats.json`，`exit_codes.tsv` 11 个退出码均为 0。`summary_waveopt.json` 按相同 110 个 episode ID 核对旧严格 SNN 与原加权 A*，两者 `cohort_match=true`。结果：

| 相同 110 集 | Success | SPL |
|---|---:|---:|
| 旧严格 SNN | 0.37272727 | 0.15155956 |
| 本次等代价低转弯 SNN | **0.34545455** | **0.13414990** |
| 原 Sigma-D + 加权 A* | 0.34545455 | 0.12453876 |

相对旧严格 SNN，本次 Success 低 **2.73 个百分点**、SPL 低 **1.74 个百分点**；Success 配对标准误 0.03972、SPL 配对标准误 0.02045。8 集由败转胜、11 集由胜转败；该小样本结果不能证明真实泛化差异的精确大小，但**明确未通过预设双指标门槛**，故不启动 2195 集全量，也不将其替换为默认 MP3D SNN 规划器。相对同批 A* 的 Success 持平、SPL 高 0.96 个百分点，不足以替代旧 SNN 门槛，更不能外推到 A* 2195 集完整验收。

`tiebreak_usage.json` 逐条检查 32154 次规划：`plan_failures=0`、`invalid_strict_calls=0`、无 A* 回退；其中 31495 次有可达的等代价路径，13831 次（43.92%）改变 STG；28906 次减少折弯、1 次增加折弯，平均折弯差 -20.27，最大加权代价差 `1.02e-12`。这证明“等代价、少折弯”实现正确，却没有带来导航增益。相对旧 SNN，平均每集转向 131.86 vs 145.62；no-path 5.99 vs 6.19；前进后位移不足 3 cm 的**停滞代理** 3.07 vs 2.41，不能当成真实碰撞数。地图快照的即时动作分歧 60/321。少转弯没有抵消一些关键方向选择的损失。

同一服务器、50 张冻结地图的交替执行 CPU 计时 `tiebreak_matched_latency50.json`：旧严格 SNN 规划中位 **15.09 ms**、新版本 **19.07 ms**，配对中位额外 **4.39 ms**；P95 分别 48.66 和 66.27 ms。这是共享服务器上的 Python/CPU 实现耗时，不是类脑芯片延迟。由于效果未达标，当前也没有理由为此投入底层 kernel 移植。

### 失败与改善案例、图

`behavior_comparison.json` 保留逐集翻转和 SPL 差；`figures/map_01.png` 等 4 张图叠加障碍、SNN 前驱路径、旧/新/A* 短期目标；`figures/paired_behavior.png`、`figures/success_flips.png`、`figures/trajectory_40_steps.png` 和正反 40 步轨迹可视化均已生成。最严重的退步是 `part5/episode5`（`X7HyMhZNoso_124`），旧 SNN 成功且 SPL 0.79556，新版失败；专门生成 `figures/part5_ep5_full_pair.png`，并用 `part5_ep5_{new,old}_audit.json` 检查。新版本第一步从旧版的 forward 动作改为转向，轨迹随后走向另一侧，10 次规划后产生停止动作；旧版用 112 次规划到达远处目标。两边都无 no-path，且新版的 STOP 判定代码仍沿用旧原路径规则。因此这更像早期等代价路径方向变化引起导航过程分叉，不能简单归因于 STOP 公式错误。另一个 40 步图的退步例 `part9/episode4`（整集 SPL -0.7096），改善例 `part7/episode2`（SPL +0.6422）。图的前 40 步与整集指标严格分开解读。

分析命令（在 `poni` 环境及工程根目录运行，产物写到候选目录）：

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/analyze_mp3d_tiebreak_audit_neuron.py \
  --run-root dairy/snn_wavefront_tiebreak_20261005/mp3d_tiebreak_pilot110
python scripts/analyze_waveopt_pilot_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_tiebreak_20261005/mp3d_tiebreak_pilot110 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
python scripts/plot_mp3d_tiebreak_neuron.py \
  --run-root dairy/snn_wavefront_tiebreak_20261005/mp3d_tiebreak_pilot110 --max-maps 4
python scripts/plot_paired_trace_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_tiebreak_20261005/mp3d_tiebreak_pilot110 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 --steps 40
python scripts/benchmark_mp3d_tiebreak_neuron.py \
  --snapshot-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 \
  --output dairy/snn_wavefront_tiebreak_20261005/tiebreak_matched_latency50.json \
  --max-snapshots 50
```

结论：本轮“等代价路径更直”假设在算法层成立，在导航层失败。保留独立入口与完整产物，默认 MP3D 仍用已验证的旧严格 SNN。下一轮不应继续凭同一 110 集反复调平局参数；应先针对首次动作分歧及目标附近停止的完整状态做更密集、可配对的诊断，再选单一规则并用新 cohort 确认。

## 后续动作分歧诊断（2026-10-05）

新增 `scripts/analyze_mp3d_first_action_neuron.py` 逐集比对旧严格 SNN 与本版首个实际动作：110 集中有 17 集首步不同，**全部**为旧版前进、本版转向；这 17 集里本版独赢 0、旧版独赢 3。文件 `first_action_outcomes.json` 保留逐集 ID 与结果。重新运行 `scripts/screen_mp3d_tiebreak_neuron.py` 的 327 张可达冻结地图，61 次动作差异中 41 次旧前进→新转向、19 次旧转向→新前进、1 次相反转向；文件 `mp3d_tiebreak_action_screen.json`。这只是决定下一小试验规则的依据，不证明动作因果或泛化。后续独立试验见 `dairy/snn_wavefront_forward_guard_20261005/`。本目录 `code_sha256.txt` 保留**本版试跑当时**哈希；之后为了新版本在适配器里增加默认关闭的选项，当前该单个源码的哈希不再相同，不能拿旧 manifest 当作新版本源码哈希。
