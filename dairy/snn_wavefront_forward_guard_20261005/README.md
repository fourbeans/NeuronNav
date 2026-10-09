# MP3D SNN 波前：等代价路径的前进动作保护

## 目的与先前证据

冻结 Sigma-D seed123 权重、MP3D 加权波前、目标选择、障碍/近障权重、STOP 和 no-path 规则。本轮只检查等到达代价路径在转换成机器人动作时，是否不该把原本的 `MOVE_FORWARD` 改成转向。

前一版低转弯 tie-break 的相同 110 集 Success/SPL 为 `0.34545455/0.13414990`，旧严格 SNN 为 `0.37272727/0.15155956`，因此前版未通过 pilot 闸门。首步配对 110 集中有 17 集动作不同，全部是“旧前进→新版转向”；其中 0 集新版单独成功、3 集旧版单独成功。冻结旧版 330 个快照中，327 张可达地图上低转弯版本有 61 次即时动作变化：41 次“旧前进→新转向”、19 次“旧转向→新前进”、1 次相反转向。这个证据支持尝试动作保护，但快照和同一 110 集仍可能过拟合，不能当成泛化证明。原始诊断产物在 `dairy/snn_wavefront_tiebreak_20261005/first_action_outcomes.json` 和 `mp3d_tiebreak_action_screen.json`。

## 接口、数据流、规则

`PFExpSigmaD` 从语义地图预测势函数/目标，在线建图和局部规划窗口保持原样。新 worker 把 `grid/start/goal/collision/visited`、机器人 `heading/turn_angle` 传给 SNN 规划适配器。加权脉冲波前一次传播，得到达时间 `T`、原前驱路径 `P_old`；低转弯回溯仅在满足等代价条件的前驱间得到 `P_tie`。路径第五格点分别为 `g_old`、`g_tie`，经原动作规则 `a(g, heading, turn_angle)` 转为前进/左右转：

`P_selected = P_old`，当 `a(g_old)=MOVE_FORWARD` 且 `a(g_tie)∈{TURN_LEFT,TURN_RIGHT}`；其他情况 `P_selected = P_tie`。

因此不增加新的网络、训练、波前传播、成本重标定或 A* 后备。原波前到达代价、同代价路径总权重、STOP 判定保持不变。仅增加两个候选短期目标的动作比较。与前版同样在每次可达规划中断言 `|C(P_tie)-C(P_old)|≤1e-5`；不可达时继续委托旧**严格 SNN** 的原 no-path 处理。这个保护规则是导航启发式，不是芯片时延或效果提升的保证。

代码边界：动作级规则最终固化在独立 `poni/planner_adapter_mp3d_action_readout_neuron.py`；新入口 `hlab/eval_poni_sigma_d_mp3d_forward_guard_neuron.py`、worker `hlab/utils/planners_mp3d_forward_guard_neuron.py`、脚本 `experiment_scripts/snn_wavefront_forward_guard_20261005/*.sh` 单独运行。原 `poni/planner_adapter_mp3d_tiebreak_neuron.py` 已恢复到试跑前原哈希，原版 PONI、旧严格 SNN、低转弯 tie-break、此前可视路径读出版本均可独立运行；没有覆盖其入口或检查点。

## 数据、训练、产物

本轮**无需训练**，不产生新 checkpoint；固定 `/mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`，SHA256 `eb6b4b55a15f808088b5187b4cd5d4d2fac3394aa215d6c2a5df9325532a2c9b`。数据 `data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz`，11 分片各前 10 集，评测 seed=100，与前版和旧严格 SNN 同一 110 集。全部日志/配置/审计/地图快照/统计位于服务器 `/mnt/jfzn/shensj/PONI/dairy/snn_wavefront_forward_guard_20261005/mp3d_forward_guard_pilot110/`：`config.txt`、`processes.tsv`、`exit_codes.tsv`、`driver_val_part_*.log`、`logs_seed_*.txt`、`tb_seed_100_val_part_*/stats.json`、`audit_val_part_*/*.jsonl` 和 `*.npz`；总启动日志为父目录 `launcher_pilot110.log`。后续结果摘要也写同一实验目录。无训练输出目录、无新 `ckpt`。

## 启动与验证

服务器 `xty-node1-direct`，工程 `/mnt/jfzn/shensj/PONI`，`poni` 环境。合成/冻结地图验证命令：

```bash
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=$PWD:$PWD/hlab:$PWD/dependencies/astar_pycpp \
  /mnt/jfzn/miniconda3/envs/poni/bin/python scripts/validate_mp3d_forward_guard_neuron.py \
  --snapshot-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
```

已通过 `MP3D_FORWARD_GUARD_UNIT_OK`，可达/不可达快照共 330 张全部通过；其中 41 次触发保护。检查了原与新波前脉冲数、到达代价、路径代价、STOP/replan、保护触发时实际 STG/路径等同旧版；no-path 的随机 STG 不作跨独立调用逐值比较。Python 编译和 Bash 语法检查也通过。

110 集启动命令（新目录存在时拒绝覆盖）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_forward_guard_20261005/launch_mp3d_forward_guard_neuron.sh \
  10 mp3d_forward_guard_pilot110
```

原严格 SNN 同 cohort 复现命令，需另起不冲突标签（本轮不重跑）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/snn_wavefront_opt_20261004/launch_mp3d_trace_neuron.sh \
  10 mp3d_old_trace_pilot110_repro_newlabel
```

原 Sigma-D + MP3D 加权 A* 完整版独立复现命令（本轮不重跑）：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh \
  dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt \
  4 5 6 7 mp3d_seed123_best2k_4gpu_repro_newlabel
```

筛选门槛仍是**同 cohort**旧严格 SNN Success≥`0.37272727`、SPL≥`0.15155956`，且至少一项严格提高。未同时通过就不启动 MP3D 2195 集全量；通过仍要在全量对原 A* Success≥`0.30523918`、SPL≥`0.11818924` 进行独立验收。小样本上的选择偏差和误差必须保留在结论里。

## 110 集运行结果与判定

11/11 个分片正常结束，`exit_codes.tsv` 全为 0；统计按相同 episode ID 与原加权 A*、旧严格 SNN 对齐，`cohort_match=true`。完整统计见 `mp3d_forward_guard_pilot110/summary_waveopt.json`（本地镜像 `result_mirror/mp3d_forward_guard_pilot110/`）。

| 相同 110 集 | Success | SPL |
|---|---:|---:|
| 原 Sigma-D + 加权 A* | 0.34545455 | 0.12453876 |
| 旧严格 SNN | **0.37272727** | **0.15155956** |
| 前版低转弯 SNN | 0.34545455 | 0.13414990 |
| 本版前进保护 SNN | 0.35454545 | 0.14198104 |

相对前版，本版 Success +0.91、SPL +0.78 个百分点；相对旧严格 SNN 仍分别 **-1.82、-0.96 个百分点**，未通过预设双指标闸门，不启动 2195 集全量。相对旧 SNN 逐集为 7 集本版独赢、9 集旧版独赢；Success/SPL 的配对标准误分别 0.03649/0.01789，小样本不能证明真实差距精确大小。相对原 A* 本版两项稍高，但这也不能替代对旧 SNN 的先验门槛或 2195 集最终验收。

严格审计 `forward_guard_usage.json`：30450 次规划、0 次规划失败、0 次非法回退；634 次 no-path，29816 次可达等代价回溯；前进保护触发 2020 次，最终 STG 相对旧严格路径变化 10969 次，最大路径代价差 `1.02e-12`。`behavior_comparison.json` 与 `figures/paired_behavior.png`：本版平均转向 122.40 次/集、旧版 145.62；无路径平均 5.76 vs 6.19；前进后位移不足 3 cm 的**停滞代理** 2.34 vs 2.41，不是真实碰撞数。转向减少仍未带来总导航提升。

`first_action_outcomes.json` 表明 110 集首步动作与旧严格 SNN 完全一致，说明原先那 17 集“首步前进→转向”确实被消除；但 `action_divergence.json` 显示 103 集在后续动作上分叉，35 集在前 10 步内分叉，且仍有 7 集本版独赢、9 集旧版独赢。因而问题不是单一首步规则，而是大量后续等价路径读出会使轨迹累积偏移。基于此，下一候选只允许“旧转向→新前进”，其他等价路径都保留旧版；独立实验目录见 `dairy/snn_wavefront_forward_gain_20261005/`。该候选仍需小样本与更大样本检验，不能由本版直接推定有效。

后处理命令：

```bash
cd /mnt/jfzn/shensj/PONI
python scripts/analyze_mp3d_forward_guard_neuron.py --mode guard \
  --run-root dairy/snn_wavefront_forward_guard_20261005/mp3d_forward_guard_pilot110 \
  --old-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
python scripts/analyze_waveopt_pilot_neuron.py --dataset mp3d \
  --run-root dairy/snn_wavefront_forward_guard_20261005/mp3d_forward_guard_pilot110 \
  --old-trace-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110
python scripts/analyze_mp3d_first_action_neuron.py \
  --new-root dairy/snn_wavefront_forward_guard_20261005/mp3d_forward_guard_pilot110 \
  --old-root dairy/snn_wavefront_opt_20261004/mp3d_old_trace_pilot110 \
  --output dairy/snn_wavefront_forward_guard_20261005/mp3d_forward_guard_pilot110/first_action_outcomes.json
```

`code_sha256.txt` 保存**本版 pilot 当时**实现的哈希。收尾时将动作读出抽到独立模块并恢复原 tie-break 源码；新模块在本版 330 张审计快照中有 323 张可达图逐项重放一致（STG、STOP/replan、脉冲数、路径代价、规则触发），7 张随机 no-path 分支跳过逐值比较。当前可独立复现的源码哈希见 `repro_code_sha256.txt`；旧 manifest 仅用于审计当时试跑，不表示当前文件哈希。

重构后另用 `run_mp3d_forward_guard_part_neuron.sh 0 0 1 /mnt/jfzn/shensj/PONI/dairy/snn_wavefront_forward_guard_20261005/repro_smoke_part0_one` 完成一集 worker 端到端冒烟，`stats.json` 已生成。一次并行冒烟启动命令的另一支因 shell 工作目录未继承而报脚本路径不存在，随后已单独正确运行；该启动错误没有产生导航评测结果，也没有进入 110 集汇总。
