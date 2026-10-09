# node1：PONI / SNN 波前规划 48 小时运行记录

## 1. 目的与边界

- 主机：`sh-xty-gpua100-node001`，SSH 别名 `xty-node1-direct`。
- 工程：`/mnt/jfzn/shensj/PONI`，启动时 HEAD `30682c2bdcd820eec8f72043b2579eb045d547bf`。
- 环境：`/mnt/jfzn/miniconda3/envs/poni`，Python 3.8.20，PyTorch 1.9.1+cu111，8×A100-SXM4-80GB。
- 正式目的：分别比较 Gibson 的原 FMM 与 SNN 波前、MP3D 的原加权 A* 与 SNN 波前。
- 资源目的：至少 48 小时维持每张卡显存占用不低于 50%。
- `sustain` 任务只做原 PONI 双头 U-Net 的合成语义图/PF 前向、反向、Adam 更新和数值稳定性检查，不是科学训练，不输出可用于导航的 checkpoint，不参与 SR/SPL 结论。

已有 dirty worktree 在启动前已经存在。本批次没有清理或覆盖既有修改。

## 2. 启动前验证

运行：

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
python scripts/validate_planner_neuron.py \
  --output /tmp/poni_node1_wave_validate_20260923
```

验证通过内容：

- 60 组随机图与 SciPy 图最短路 oracle 一致。
- unit-tick IF oracle、状态重置、无路径、起点即目标和事件预算检查通过。
- 80 组 MP3D weighted A* 代价对照通过。
- Gibson、MP3D 控制器接口和双 worker IPC 检查通过。

结果复制为 `planner_validation.json`。

## 3. 正式导航矩阵

### 3.1 Gibson Box v1.1

每个批次 5 个 val 场景、每场景 200 episodes，总计 1,000 episodes；评测 seed 均为 100。

| GPU | checkpoint | 模式 | 目录 |
|---:|---:|---|---|
| 0 | 123 | legacy neuron audit | `pilot_wave_gpu0/` |
| 1 | 123 | reference FMM | `gibson_ref123_gpu1/` |
| 2 | 234 | legacy neuron audit | `gibson_wave234_gpu2/` |
| 3 | 234 | reference FMM | `gibson_ref234_gpu3/` |
| 4 | 345 | legacy neuron audit | `gibson_wave345_gpu4/` |
| 5 | 345 | reference FMM | `gibson_ref345_gpu5/` |

`reference` 只执行原 FMM。旧 `neuron` 入口正常路径执行 SNN 波前，但 `NoPathNeuron` 时会回退原规划器；因此这三条旧 neuron 批次只作为候选鲁棒性诊断，不能自动当作独立 SNN 结果。

### 3.2 MP3D val

完整 11 个 `val_part_0..10`，每个分片约 200 episodes：

| GPU | checkpoint | 模式 | 目录 |
|---:|---:|---|---|
| 6 | 123 | reference weighted A* | `mp3d_ref123_gpu6/` |
| 7 | 123 | legacy neuron audit | `mp3d_wave123_gpu7/` |

每个分片分别保存：

- `stdout.log`
- `eval.log`
- `audit/*.jsonl`
- `tb/stats.json`（完成后）

入口脚本为 `run_mp3d_wave_node1.sh`，逐一等待 11 个子进程并将退出码写入 `exit_codes.tsv`，避免原脚本只 `wait` 而不记录单个失败。

## 4. 严格 SNN 入口

审计发现旧 `neuron` 入口确实发生回退：

- Gibson：`NoPathNeuron: disconnected_start`。
- MP3D：`NoPathNeuron: disconnected_goal` 或 `empty_or_blocked_goal_mask`。

因此新增以下文件，均不覆盖旧入口：

- `poni/runtime_strict_neuron.py`
- `poni/planner_adapter_strict_neuron.py`
- `semexp/envs/habitat/sem_exp_strict_neuron.py`
- `semexp/eval_poni_strict_neuron.py`
- `hlab/utils/planners_strict_neuron.py`
- `hlab/eval_poni_strict_neuron.py`

严格模式中：

- 正常路径只执行 SNN 波前。
- Gibson 不可达时，用波前到达图的不可达常数和原环形短期点解码规则返回 `replan` 行为，不调用 `skfmm.distance`。
- MP3D 不可达时，复现上游控制器的随机短期点和 `replan=True`，不调用 weighted A*。
- 其他异常直接写 `plan_failure` 并终止，不回退。

单元验证确认：候选成功和候选异常两种情况下，reference planner 调用次数均为 0；封闭障碍图下 Gibson/MP3D 的 no-path 路径均能独立返回控制结果。

严格 Gibson 启动模板：

```bash
cd /mnt/jfzn/shensj/PONI/semexp
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/jfzn/shensj/PONI \
python eval_poni_strict_neuron.py \
  --neuron-mode neuron \
  --neuron-log-dir <new-output>/audit \
  --neuron-capture-limit 0 \
  --neuron-scene Collierville \
  --split val --seed 100 --eval 1 \
  --task_config tasks/objectnav_gibson_box.yaml --version v1.1 \
  --auto_gpu_config 0 --num_processes 1 \
  --num_processes_on_first_gpu 1 --num_processes_per_gpu 1 --sim_gpu_id 0 \
  --pf_model_path ../pretrained_models/gibson_models/poni_seed_123.ckpt \
  -d <new-output> --num_eval_episodes 200 \
  --num_local_steps 1 --exp_name seed_100 --global_downscaling 1 \
  --mask_nearest_locations --pf_masking_opt unexplored --seg_interval 3
```

严格 MP3D 启动时将原 `hlab/eval_poni.py` 换成 `hlab/eval_poni_strict_neuron.py`，保留原 PFExp、split、checkpoint 和地图设置，并增加 `--neuron-mode neuron --neuron-log-dir <new-dir>`。

### 4.1 严格入口冒烟验证

Gibson（Collierville，1 episode）完成，Success=1、SPL=1；共 15 条规划记录，全部为 `used=neuron`、`reference=null`、`fallback=false`，无异常。

MP3D（`val_part_0`，1 episode）完成。该 episode 跑满 500 步，Success=0、SPL=0，因此不能单独用来判断总体导航效果；其接口和独立性检查通过：500 条规划记录全部为 `mode=neuron_strict`、`used=neuron`、`reference=null`、`fallback=false`，无异常。完整 val 结果用于判断最终 SR/SPL。

### 4.2 严格完整实验

Gibson 严格实验使用 checkpoint 123、评测 seed 100、每场景 200 episodes。为了与原版结果逐场景比较并控制瞬时显存，三个 driver 串行覆盖五个场景：

| GPU | 场景顺序 | 目录 |
|---:|---|---|
| 0 | Collierville → Corozal | `gibson_strict123_gpu0/` |
| 2 | Darden → Markleeville | `gibson_strict123_gpu2/` |
| 4 | Wiconisco | `gibson_strict123_gpu4/` |

MP3D 严格实验同样使用 checkpoint 123、评测 seed 100。每张卡同一时刻只新增一个严格 MP3D 进程；有两个分片的卡在前一分片正常退出后才启动下一分片：

| GPU | 分片顺序 | 目录 |
|---:|---|---|
| 0 | 0 | `mp3d_strict123_gpu0/` |
| 1 | 1 → 8 | `mp3d_strict123_gpu1/` |
| 2 | 2 | `mp3d_strict123_gpu2/` |
| 3 | 3 → 9 | `mp3d_strict123_gpu3/` |
| 4 | 4 | `mp3d_strict123_gpu4/` |
| 5 | 5 → 10 | `mp3d_strict123_gpu5/` |
| 6 | 6 | `mp3d_strict123_gpu6/` |
| 7 | 7 | `mp3d_strict123_gpu7/` |

完整 MP3D 严格版启动器的通用命令为：

```bash
bash dairy/node1_48h_20260923/run_mp3d_strict_node1.sh \
  <gpu> \
  /mnt/jfzn/shensj/PONI/pretrained_models/mp3d_models/poni_seed_123.ckpt \
  /mnt/jfzn/shensj/PONI/dairy/node1_48h_20260923/<new-output> \
  <part-id> [<next-part-id> ...]
```

原版 MP3D 对照仍由 `hlab/eval_poni.py` 启动；严格 SNN 版由新建的 `hlab/eval_poni_strict_neuron.py` 启动，二者输出目录完全分离。原版 Gibson 对照仍由 `semexp/eval_poni.py` 启动；严格 SNN 版由新建的 `semexp/eval_poni_strict_neuron.py` 启动。

## 5. 49 小时 PONI sustain 任务

脚本：`scripts/poni_gpu_sustain_neuron.py`。

每张卡：

1. 加载与正式导航相同的 PONI checkpoint。
2. 构建原编码器、Object PF 解码器和 Area PF 解码器。
3. 生成 PONI 形状的累计语义图、Object PF 和 Area PF 合成 batch。
4. 持续执行两个监督 MSE、反向传播和 Adam 更新。
5. 通过 `nvidia-smi` 读取整张物理卡显存，而不是只看本进程。
6. 动态增减 GPU 常驻 batch，使整卡目标占用约 50,000MiB。
7. 训练激活缓存后实际稳态约 60–62GB，每张卡仍保留约 19–20GB。

初始稳定状态：

| GPU | 实测显存 |
|---:|---:|
| 0 | 61,535MiB |
| 1 | 61,137MiB |
| 2 | 61,493MiB |
| 3 | 61,393MiB |
| 4 | 61,393MiB |
| 5 | 60,827MiB |
| 6 | 61,056MiB |
| 7 | 61,620MiB |

严格 Gibson 与严格 MP3D 同时启动后的采样仍为每卡约 60.9–65.9GB，8 卡全部高于 40,960MiB；启动扫描未发现 CUDA OOM 或 Python traceback。

每个 `sustain_gpuN/metrics.jsonl` 记录：loss、step、耗时、常驻 batch、进程分配/保留显存、整卡占用和空闲显存。

## 6. 看门狗

`watchdog_node1.sh` 运行 49 小时：

- 每分钟把 8 卡显存、利用率和温度追加到 `gpu_watch.csv`。
- sustain 进程退出时，用全新的 `sustain_gpuN_restart_<timestamp>/` 重启。
- 某卡连续 3 次低于 40,960MiB 时，只重启该卡 sustain，不终止正式导航实验。
- 每次重启写入 `watchdog.log`。

监控命令：

```bash
nvidia-smi
tail -f /mnt/jfzn/shensj/PONI/dairy/node1_48h_20260923/gpu_watch.csv
tail -f /mnt/jfzn/shensj/PONI/dairy/node1_48h_20260923/watchdog.log
tail -f /mnt/jfzn/shensj/PONI/dairy/node1_48h_20260923/sustain_gpu0/metrics.jsonl
```

## 7. 完成后的结果位置

Gibson：

```text
<run>/dump/seed_100/final_eval_stats.json
<run>/audit/gibson_*.jsonl
```

MP3D：

```text
<run>/val_part_*/tb/stats.json
<run>/val_part_*/audit/mp3d_*.jsonl
<run>/exit_codes.tsv
```

MP3D 汇总：

```bash
cd /mnt/jfzn/shensj/PONI
python hlab/merge_results.py \
  --path_format 'dairy/node1_48h_20260923/mp3d_ref123_gpu6/val_part_*/tb/stats.json'
```

所有 SNN 结果还必须统计 `used=neuron`、`used=reference`、`fallback`、`error` 和 `no_path`。只有严格入口结果，或旧入口中逐条证明 reference 使用次数为 0 的结果，才可标成“独立 SNN 波前”。

## 8. 文件哈希与异常记录

启动时哈希：

```text
cc85df19317b2267ec8b022effc1bb310c18a60042d094467a0b1ddcd1a5a673  scripts/poni_gpu_sustain_neuron.py
e0fe116979e4189ab20248a02ff05309b53336c7c7d8c00381a72c2d8034e96d  run_mp3d_wave_node1.sh
33446ca85ff70b23022a85e60d3e6be641823e4ae2ce67b4c388de3d1639552b  watchdog_node1.sh
e050de376b78918d042a699cfa787eb80efb110107229472d05908c130768a35  poni/runtime_strict_neuron.py
8e647cb15aa78024487fddda94851cc5d35aa250a49497bbcbe5cd1deda8c6c0  poni/planner_adapter_strict_neuron.py
08a5c5ebfcf2985cf5d464946b3b9f5a74259d67433cc6fa9bd6a66cd0487bf9  semexp/envs/habitat/sem_exp_strict_neuron.py
ab01a3764c2c9b3a64ad3aede2f5439483c5ec8b7b4ccf123b679d93745544bb  semexp/eval_poni_strict_neuron.py
56c64f1b2081cdd2a271f2acb12b456765904c758eb820fdfd1d072fb2116add  hlab/utils/planners_strict_neuron.py
d2a24312f55ab287f5f9b5482cd95a37242827bcc59e8ca616aa18be3e0bd1b7  hlab/eval_poni_strict_neuron.py
83358e7239b4bc5f6a037f533e5d24f966ba7d205cfb7914230eb2d68b868f66  run_gibson_strict_node1.sh
2bfcbddfd185d00eda1dd32a4c8212a76b65cf11acdfa5eef970ab954a2cab81  run_mp3d_strict_node1.sh
```

第一次 sustain 启动因 PyTorch 1.9 没有 `torch.cuda.mem_get_info()` 而退出。按用户要求，8 组该次失败目录和 driver log 已删除，不可恢复；随后改用 `nvidia-smi` 查询整卡显存，兼容短测完成 21 个训练步后重新启动正式 49 小时任务。

第一次批量启动严格 MP3D driver 时，后台 shell 的相对日志路径只让 GPU0 成功进入任务；GPU1–7 没有创建实验目录或结果。随后改为绝对路径并用独立 session 启动，8 个 driver 均已核实存在，对应首批 `val_part_0..7` 均已进入 `eval_poni_strict_neuron.py`。遗留的两个空等 shell 已终止，GPU0 的正式评测子进程保持运行。

## 9. 完整结果分析

严格独立 SNN 波前实验已完成，所有正式退出码均为 0。详细的完整性检查、逐场景/逐分片指标、配对统计、no-path 归因、事件量和下一步优先级见：

```text
independent_snn_wavefront_results.md
independent_snn_wavefront_summary.json
```

核心结论：MP3D Success 30.57% → 30.66%、SPL 11.61% → 12.58%，达到本阶段替换 weighted A* 的效果目标；Gibson Success 73.10% → 70.60%、SPL 39.69% → 36.62%，尚未达到“不弱于 FMM”，下一步需优先修复严格 SNN 的断连/no-path 分支并改进 8 邻域距离场。

结果分析产物哈希：

```text
dfdcf6cf80a285b5bbaaeda3ca9ef65abd3cc9f1f90315c0d0563bc33f8bd571  scripts/summarize_wavefront_results_neuron.py
3553d55f577c81a35a7f451ae8e079b8551ffc39e861b0104990364296234be8  independent_snn_wavefront_results.md
fcd31c7ec5bdc36f13eb45fac6a74bb7282a030baefafc9e40c239884bfd2a1e  independent_snn_wavefront_summary.json
```
