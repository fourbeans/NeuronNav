# MP3D SNN 波前：只接受等代价路径带来的转向→前进

## 目的、接口与算法

此前低转弯 tie-break 与“旧前进保护”小样本均低于旧严格 SNN。前进保护版虽然 110 集首步动作已经与旧版全部相同，仍有 103 集后续动作分叉，35 集前 10 步内分叉；同批 Success/SPL `0.35454545/0.14198104` vs 旧严格 SNN `0.37272727/0.15155956`。因此本轮再缩小改动范围：**只在原 SNN 第五格点要求转向、等代价低转弯路径第五格点允许前进时，才采用新路径；其余仍采用原 SNN 前驱路径**。冻结地图中此种即时动作转换 19/327 张可达快照；需通过真实导航验证，不可凭此推断收益。

输入仍是 Sigma-D 势函数目标、局部栅格地图 `grid/start/goal/collision/visited`、机器人朝向/转角。相同目标、障碍膨胀、近障权重、加权脉冲波前到达时间 `T`、原路径 `P_old`。在 Bellman 等到达代价前驱中可回溯平直路径 `P_tie`；记第五格点动作 `a_old`、`a_tie`：

`P_selected = P_tie` 当且仅当 `a_old∈{TURN_LEFT,TURN_RIGHT}` 且 `a_tie=MOVE_FORWARD`，否则 `P_selected=P_old`。

每次可达规划断言 `|C(P_tie)-C(P_old)|≤1e-5`；STOP 始终由原路径长度决定；no-path 委托旧严格 SNN 原处理，**没有 A* 回退**。不重复跑波前、不改传播权重、不训练新模型。此规则比前版更保守，预期减少后续行动分叉，但是否改善 Success/SPL 必须以整集结果确认。

独立代码边界：动作级适配器最终固化在新文件 `poni/planner_adapter_mp3d_action_readout_neuron.py`；新 worker `hlab/utils/planners_mp3d_forward_gain_neuron.py` 复用前进保护版可传动作上下文的基类，入口 `hlab/eval_poni_sigma_d_mp3d_forward_gain_neuron.py`、运行脚本 `experiment_scripts/snn_wavefront_forward_gain_20261005/*.sh` 与既有实验分开。`poni/planner_adapter_mp3d_tiebreak_neuron.py` 已恢复原哈希。原 PONI、旧严格 SNN、低转弯、前进保护入口均可独立运行，检查点不变。

## 数据、训练、验证、目录

本轮无训练、无新 checkpoint。固定 MP3D Sigma-D seed123 `dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`，SHA256 `eb6b4b55a15f808088b5187b4cd5d4d2fac3394aa215d6c2a5df9325532a2c9b`。数据 `data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz`，11 分片各前 10 集、seed=100，同前几版/旧严格 SNN 的 110 集。用同 cohort 便于逐集归因，但会有重复使用小样本的选择偏差。

Python 编译/Bash 语法检查通过。冻结 330 个旧版地图快照及合成地图通过 `scripts/validate_mp3d_forward_guard_neuron.py`，其中可达快照只接受 19 次转向→前进。检查了原/新到达代价、脉冲数、路径代价、STOP/replan 和所选 STG/路径的逐值规则；不可达 no-path 有随机 STG，不作独立调用间的逐值比较。一集真实 MP3D 冒烟 `smoke_part0_one/` 完成，`stats.json` 已写，首条审计确认 `readout_mode=equal_arrival_forward_gain_only`、`used=neuron`、`fallback=false`。

验证与启动：

```bash
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=$PWD:$PWD/hlab:$PWD/dependencies/astar_pycpp \
  /mnt/jfzn/miniconda3/envs/poni/bin/python scripts/validate_mp3d_forward_guard_neuron.py \
  --snapshot-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
bash experiment_scripts/snn_wavefront_forward_gain_20261005/run_mp3d_forward_gain_part_neuron.sh \
  0 0 1 /mnt/jfzn/shensj/PONI/dairy/snn_wavefront_forward_gain_20261005/smoke_part0_one_newlabel
bash experiment_scripts/snn_wavefront_forward_gain_20261005/launch_mp3d_forward_gain_neuron.sh \
  10 mp3d_forward_gain_pilot110
```

原严格 SNN 复现：`bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_trace_neuron.sh 10 mp3d_old_trace_pilot110_repro_newlabel`。原 Sigma-D + 加权 A* 完整复现：`bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt 4 5 6 7 mp3d_seed123_best2k_4gpu_repro_newlabel`。均未在本轮重跑。

主产物服务器目录 `/mnt/jfzn/shensj/PONI/dairy/snn_wavefront_forward_gain_20261005/mp3d_forward_gain_pilot110/`；标签存在则拒绝覆盖，含 `config.txt`、`processes.tsv`、`exit_codes.tsv`、`driver_val_part_*.log`、`logs_seed_*.txt`、`tb_seed_100_val_part_*/stats.json`、`audit_val_part_*/*.jsonl`/快照、`summary_waveopt.json` 与分析图。父目录 `launcher_pilot110.log`；`code_sha256.txt` 记录**当时试跑代码**和原权重哈希，`repro_code_sha256.txt` 记录收尾重构后的可独立复现源码。无训练产物目录与新 ckpt。

110 集闸门：Success/SPL 同时不低于旧严格 SNN `0.37272727/0.15155956`，且至少一项严格提高；未通过不启动 2195 集。通过也只表明该重复使用的小样本可晋级，需再用更大独立 MP3D 完整评测确认原 A* 的 `0.30523918/0.11818924` 验收目标。

## 110 集运行结果与判定

11/11 分片结束、11 个退出码全为 0；与原加权 A*、旧严格 SNN 的 episode ID 完全对齐。`summary_waveopt.json` 结果：

| 相同 110 集 | Success | SPL |
|---|---:|---:|
| 原 Sigma-D + 加权 A* | 0.34545455 | 0.12453876 |
| 旧严格 SNN | **0.37272727** | **0.15155956** |
| 前进保护 SNN | 0.35454545 | 0.14198104 |
| 本版仅转向→前进 SNN | 0.34545455 | 0.13618402 |

相对旧严格 SNN，Success **-2.73**、SPL **-1.54 个百分点**，配对标准误分别 `0.03972/0.01824`；8 集本版独赢、11 集旧版独赢。相对前进保护版也更差。本轮**未通过小样本双指标门槛**，明确不启动 2195 集，也不替换默认 MP3D 旧严格 SNN。反复使用同一 110 集会带来选择偏差；不能继续凭它筛更多读出小规则。

`forward_guard_usage.json`（文件名沿用分析工具，在 `--mode gain` 下记录本版）：30220 次规划、0 次规划失败、0 次非法回退；648 次 no-path、29572 次可达等代价回溯；3669 次被允许的“转向→前进”改变 STG，最大路径代价差 `8.53e-13`。`behavior_comparison.json`：平均转向 120.91 vs 旧严格 145.62 次/集，前进停滞代理 2.66 vs 2.41（不是碰撞数）；转向更少仍未转化为导航成功。`action_divergence.json`：110 集首步动作全部与旧版相同，103 集后续出现动作分叉，30 集在前 10 步内分叉；新旧成功翻转 8/11。图 `figures/paired_behavior.png` 保留配对 SPL/转向/no-path 分布。以上结果均在 `mp3d_forward_gain_pilot110/`，本地镜像在 `result_mirror/mp3d_forward_gain_pilot110/`。

收尾重构后，对本版 330 张历史审计快照重放：323 张可达图的 STG、STOP/replan、脉冲数、路径代价、规则触发与当时结果逐项一致；7 张随机 no-path 分支不作 STG 逐值比较。命令 `python scripts/validate_mp3d_forward_guard_neuron.py --replay-run-root dairy/snn_wavefront_forward_gain_20261005/mp3d_forward_gain_pilot110`，结果 `MP3D_ACTION_READOUT_REPLAY_OK`。原 tie-break 源码 SHA256 已恢复为 `fb2a8578d6db16e7066223aec7133883f5547e76854fccf0c6a4144389024d79`。

重构后独立 worker 入口另跑一集冒烟：`bash experiment_scripts/snn_wavefront_forward_gain_20261005/run_mp3d_forward_gain_part_neuron.sh 0 1 1 /mnt/jfzn/shensj/PONI/dairy/snn_wavefront_forward_gain_20261005/repro_smoke_part0_one`，`stats.json` 已生成，审计首条仍是 `used=neuron`、`fallback=false` 和 `equal_arrival_forward_gain_only`。

后处理复现：

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/analyze_mp3d_forward_guard_neuron.py --mode gain \
  --run-root dairy/snn_wavefront_forward_gain_20261005/mp3d_forward_gain_pilot110 \
  --old-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
python scripts/analyze_waveopt_pilot_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_forward_gain_20261005/mp3d_forward_gain_pilot110 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
```

## 回查完整 2195 集旧基线，确定下一处诊断

用 `scripts/compare_mp3d_full_baselines_neuron.py` 按 2195 个相同 episode ID、相同 `goal_distance` 比对旧严格 SNN 与原加权 A*，产物 `full_old_snn_vs_astar.json`。旧严格 SNN Success **0.29931663**、SPL **0.12177395**；原 A* 分别 **0.30523918/0.11818924**。故真正的全量差距是 SNN Success **-0.592 个百分点**，但 SPL **+0.358 个百分点**；配对标准误分别 `0.903/0.409` 个百分点，不能把这点差距说成已证明规划器显著更差。旧 SNN 独赢 190 集，A* 独赢 203 集，净差仅 13 集。

按初始目标距离分四等份，最短一档（1.00–3.24m）SNN Success 48.82% vs A* 46.81%，SPL 17.68% vs 15.86%；最长一档（10.83–29.70m）SNN Success 8.93% vs A* 11.11%，SPL 4.45% vs 5.25%。这提示较长导航回合的累积决策更值得诊断，而非继续让每条等代价栅格路径尽量直。但距离分组只是相关性，不能归因到单个模块。

在 A* 成功而 SNN 失败的 203 集中，SNN 终点到目标中位距离 2.06m；只有 11 集小于 0.1m、30 集小于 0.25m。MP3D 配置 `transfer_configs/objectnav_mp3d.yaml` 的成功距离是 0.1m，且要执行 STOP，因此不能把“小于 1m”的 61 集直接称为几乎成功，也不能简单调大 STOP 阈值。下一步应选这些长距离/成功翻转回合做带 episode ID 的严格 SNN 与 A* 轨迹配对，分类实际是目标选择、no-path、障碍边界还是停止行为；得到机制证据后再改**一条**规划规则，并用未参与选择的 episode 检验。当前没有证据支持继续叠加等代价读出启发式。

比对命令：

```bash
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=$PWD:$PWD/hlab:$PWD/dependencies/astar_pycpp \
  /mnt/jfzn/miniconda3/envs/poni/bin/python scripts/compare_mp3d_full_baselines_neuron.py \
  --output dairy/snn_wavefront_forward_gain_20261005/full_old_snn_vs_astar.json
```
