# 完整增量脉冲 U-Net（Sigma–Delta）实验记录

> 实验根目录：`/mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/`
>
> 本文同时是设计说明、运行手册和逐次实验登记。任何正式结果都必须能由本文命令和对应 run 目录复现。

## 1. 目标、命名与不可破坏边界

本实验把 PONI 的双头 U-Net 改为跨真实导航步保存状态的增量 Sigma–Delta 脉冲网络。空间主干、skip connection、object PF head、area PF head 和原监督标签保持兼容；局部规划器仍使用各数据集原规划器，以单独测量 PONI 网络改造的影响。

命名规则：

- Python 文件使用合法模块名后缀 `_sigma_d.py`。
- 实验名、日志字段和文档显示名使用 `sigma-d`。
- 不再给本实验新增 `_neuron.py` 文件。

边界：

1. 不覆盖原 `poni/model.py`、`train.py`、`semexp/model_pf.py` 或已有 `_neuron.py`。
2. 原版 PONI、严格 SNN 波前和 Sigma–Delta PONI 必须可以分别启动。
3. 不把同一张地图复制成 `[x,x,...,x]` 后重复计算 T 次。
4. 不使用原 PONI 输出做蒸馏标签；监督只来自 GT semantic map、FMM distance 和 unexplored-area 标签。
5. Sigma–Delta 导航入口不得在异常或低置信度时调用原 PONI。
6. 不引入 Softmax、RMSNorm、QK 外积、注意力矩阵或 GRU/LSTM 状态。
7. runtime state 不写入 checkpoint；checkpoint 只保存权重、阈值、优化器、调度器和配置。
8. 先保证效果，再逐步加事件稀疏和量化约束；不能为了稀疏率掩盖导航掉点。

## 2. 原 PONI 结构与逐层对应

原 map 模式输入为 `B×C×480×480`：Gibson `C=17`，MP3D `C=23`。

编码器：

```text
input
  -> 1x1 embed Conv: C -> 64                         480x480
  -> double Conv+BN+ReLU: 64 -> 32 -> 32       x1  480x480
  -> MaxPool + double Conv: 32 -> 64 -> 64     x2  240x240
  -> MaxPool + double Conv: 64 -> 128 -> 128   x3  120x120
  -> MaxPool + double Conv: 128 -> 256 -> 256  x4   60x60
  -> MaxPool + double Conv: 256 -> 256 -> 256  x5   30x30
```

object decoder 与 area decoder 完全独立，只共享 encoder：

```text
x5 --bilinear--> concat(x4) -> double Conv 512 -> 128
   --bilinear--> concat(x3) -> double Conv 256 -> 64
   --bilinear--> concat(x2) -> double Conv 128 -> 32
   --bilinear--> concat(x1) -> double Conv  64 -> 32
   -> 1x1 Conv -> object C channels / area 1 channel -> sigmoid
```

首版 Sigma–Delta 保留相同通道、卷积核、双 decoder、bilinear、channel concat 和 sigmoid。只改变每层跨导航步的计算方式，避免同时改变空间表达能力。

## 3. 算法原理

### 3.1 输入事件

导航步 `t` 的累计语义地图只输入一次：

\[
\Delta x_t=x_t-x_{t-1}.
\]

必须是 signed ON/OFF event，因为在线语义会修正，local-map 窗口也可能移动；只允许正事件会产生不可恢复的累计漂移。

### 3.2 卷积累加状态

对第 `l` 层，首帧或 reset 时执行一次完整初始化。原 PONI checkpoint
加载时先把 `Conv+BN` 精确融合成带 bias 的卷积；Sigma–Delta 模型自身不再
保留训练态 BN：

\[
u_0^l=W'^l*\hat a_0^{l-1}+b'^l,
\qquad
a_0^l=\operatorname{ReLU}(u_0^l).
\]

后续导航步只对上游事件做卷积：

\[
u_t^l=u_{t-1}^l+W'^l*\Delta\hat a_t^{l-1},
\qquad
a_t^l=\operatorname{ReLU}(u_t^l).
\]

融合公式为：

\[
W'_c=\frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}}W_c,
\qquad
b'_c=\beta_c+\frac{\gamma_c}{\sqrt{\sigma_c^2+\epsilon}}(b_c-\mu_c).
\]

### 3.3 Sigma–Delta 神经元

当前连续激活与上次已发送重构激活的误差进入 Sigma–Delta 量化器：

\[
e_t^l=a_t^l-\hat a_{t-1}^l,
\]

\[
k_t^l=\operatorname{clip}\left(
\operatorname{round}_{STE}\left(\frac{e_t^l}{\theta_l}\right),
-K,K\right),
\]

\[
q_t^l=\theta_l k_t^l,
\qquad
\hat a_t^l=\hat a_{t-1}^l+q_t^l.
\]

未发送误差自然保留在下一步的
`a_t-\hat a_t` 中，因此代码不另外保存一个会重复计数的 residual state。
`k=3` 表示三个单位脉冲或一个带符号整数事件包，不表示把网络运行三遍。
`round` 反向使用 straight-through estimator。

第一帧默认 exact warm-start：直接用连续激活初始化重构状态，保证权重迁移后的首帧尽可能接近原 PONI。后续帧才统计增量事件率。

### 3.4 U-Net 非线性算子

MaxPool 不能写成 `old_pool + pool(delta)`。首版缓存重构输入并重新计算正确池化值：

\[
p_t=\operatorname{MaxPool}(\hat a_t),
\qquad
\Delta p_t=p_t-p_{t-1}.
\]

这保证数值正确；部署优化阶段再改为只更新受事件影响的局部 2×2 窗口。

Bilinear upsample 是线性算子：

\[
\Delta U_t=U(\Delta\hat a_t).
\]

Skip connection 保持原 channel concat：

\[
\Delta c_t=\operatorname{concat}(\Delta s_t,U(\Delta d_t)).
\]

### 3.5 状态与 reset

每个并行环境单独持有：上一输入、每层卷积累积值、重构激活、pool 缓存和输出状态。
当前模型 state 尚不内置坐标 metadata，因此部署 adapter 必须依据 episode、local-map
origin/boundary 变化明确传入 `reset_mask`；不能依赖模型猜测坐标是否变化。

以下情况必须只 reset 对应环境槽位：

- 新 episode 或 `done`；
- local-map boundary/origin 改变；
- shape、resolution 或 channel schema 改变；
- checkpoint/threshold 版本改变。

reset 后仍运行同一个 Sigma–Delta 模型的首帧初始化，不是调用原 PONI。

Gibson `global_downscaling=1` 时输入是固定全局坐标系，`g_masks=0` 直接触发
对应环境的 dense reset。MP3D HLab 会使用 smart local boundaries，但原接口
不把 `lmb` 传给 policy；新 adapter 因此跟踪 `extra_maps["agent_locations"]`。
重定中 local-map 原点会使 agent 局部网格位置发生超过地图宽度 10% 的跳变，
对该环境显式 reset。多 reset 只会运行精确 dense init，不改变预测；未 reset
才会破坏累积状态。

## 4. 数据流

训练序列来自同一 scene/floor、同一 shortest path：

```text
GT semantic map + per-class FMM distance
  -> sample one sufficiently long shortest path
  -> select K strictly increasing path prefixes
  -> one shared crop center and rotation for all K frames
  -> cumulative visible semantic maps x_0...x_{K-1}
  -> per-step frontier/loss mask
  -> per-step object PF and unexplored-area PF labels
  -> Sigma–Delta U-Net first-frame init + K-1 incremental steps
```

默认 `K=4`。序列中每帧都是不同导航时刻的累计地图，不允许用一张静态地图复制四次。

为避免 MP3D 极小楼层上反复重算 FMM 却只得到长度 1–2 的路径，
Sigma-D 数据层从一个随机起点计算一次 FMM，再在可达距离较远的 50%
网格中采样终点。这等价于对“路径长度至少为 K”条件采样，不是把短路径
补帧。如当前 scene-floor 仍无法产生严格增长序列，数据层按当前 epoch
的确定性场景顺序前进，并在 metadata 中记录 `source_sample_index` 和
`scene_fallback_hops`。

磁盘 `/mnt/jfzn` 当前使用率约 97%，余量约 3.1 TB。原 PONI 的 400k 大样本预计算会大量重复相同 full semantic/FMM map，因此本实验采用 on-the-fly 序列和紧凑 metadata，不生成 400k×K 张地图。

## 5. 直接监督训练

object PF 标签：

\[
Y_{obj}=\operatorname{clip}\left(
\frac{d_{cut}-D_{FMM}}{d_{cut}},0,1\right).
\]

Gibson `d_cut=10m`，MP3D `d_cut=5m`。

原 object loss 保持：忽略 free/wall 两个通道，在每步 frontier/visibility mask 内做归一化 MSE：

\[
\mathcal L_{obj}=
\operatorname{mean}_{b,t,c\ge2}
\frac{\sum_{h,w}M_{btchw}(\hat P_{btchw}-Y_{btchw})^2}
{\sum_{h,w}M_{btchw}+\epsilon}.
\]

Area head：

\[
\mathcal L_{area}=\operatorname{MSE}(\hat A,A_{GT}).
\]

可选时间变化监督不是强迫前后相同，而是要求预测变化接近标签变化；
当前实现使用 masked MSE，默认权重为 0：

\[
\mathcal L_{temp}=\operatorname{MSE}_{M_t\lor M_{t-1}}\left[
(\hat P_t-\hat P_{t-1})-(Y_t-Y_{t-1})\right].
\]

另有一个可选的输出 full-map GT MSE（不是内部层重构，也不是教师 loss），默认权重为 0：

\[
\mathcal L_{full}=\operatorname{mean}_{b,t,c\ge2,h,w}
(\hat P-Y)^2.
\]

事件量、active event 和饱和只作为 detached 诊断：

\[
R_{mag}=\frac{\sum_l\sum|k_t^l|}{\sum_lN_l},
\]

\[
R_{active}=\frac{\sum_l\sum\mathbf 1[k_t^l\ne0]}{\sum_lN_l},
\qquad
R_{sat}=\frac{\sum_l\sum\mathbf 1[|\operatorname{round}(e_t^l/\theta_l)|>K]}{\sum_lN_l}.
\]

总损失：

\[
\mathcal L=\lambda_{obj}\mathcal L_{obj}
+\lambda_{area}\mathcal L_{area}
+\lambda_{temp}\mathcal L_{temp}
+\lambda_{full}\mathcal L_{full}.
\]

当前首个效果基线只打开 object/area 直接监督。事件统计不进入反向传播，避免为了稀疏率掩盖任务掉点。

## 6. 权重初始化与“非蒸馏”说明

代码同时支持：

- `pretrained-init`：把原 PONI Conv/BN 权重映射到同构 Sigma–Delta 网络，再用 GT 标签训练；
- `random-init`：完全随机初始化、仍使用同一 GT 监督。

主线先用 `pretrained-init` 最大限度保留空间能力。它只是参数初始化，不使用教师输出、不包含 teacher loss，因此不是知识蒸馏。`random-init` 作为独立消融，不能用来替代主线效果门槛。

## 7. 分阶段实验

### S0：单元与接口验证

目的：先证明增量数学、state/reset 和原接口正确，不产生科学 checkpoint。

必须通过：

1. 首帧权重迁移后与原 PONI 输出误差在容许范围内；
2. exact/bypass 模式中，逐帧 dense 与 first-frame+delta 一致；
3. 相同帧第二次输入时事件数为 0，输出不漂移；
4. 像素增加、删除和类别修正产生 signed event；
5. MaxPool 原最大值下降时能切换到次大值；
6. batch 中单环境 reset 不污染其他环境；
7. state 不出现在 `state_dict`；
8. object/area shape 与 `[0,1]` 范围兼容；
9. STE 梯度有限且非零；
10. CPU/GPU forward 无 NaN、无 OOM。

### S1：小序列过拟合

目的：在固定 32–128 条 `K=4` Gibson 序列上检查训练闭环和可拟合性。
必须设置 `DATASET.freeze_train_samples=True`；否则每个 epoch 会确定性换一组路径，
只能称为 short-fit，不能称为过拟合实验。验证集始终固定为 epoch 0 样本。

### S2：Gibson 2k-update 门控实验

先并行比较 `theta=1/64,1/128,1/256,1/512`，再固定 `theta=1/128`
比较 `qmax=15,31,63,127,255,511,1023,4095,32767`。第一轮已证明
低 `qmax` 的饱和是主要保真瓶颈；当前进入训练门控的配置为 `qmax=4095`
（12-bit signed event packet，或拆成多个较小 event packet）。

门槛：相同数据上的 val task loss 不高于 stateless control 5%；进入 full train 前收紧到 2%。同时报告 frontier top-1/top-k、regret、input delta density、逐层事件率、饱和率和理论 SynOps。

### S3：Gibson full train 与导航

最佳两配置、至少 seed 123/234。训练后跑完整 5 场景 1,000 episodes，使用原 Gibson FMM 规划器。验收对照为既有原 PONI：Success 73.10%、SPL 39.69%。暂定非劣界为均不低于原版 1 pp，并报告配对 bootstrap CI。

### S4：MP3D

MP3D 先完成 semantic map 和 FMM 标签，再复用相同配置进行 2k 门控和 40k full train，最终跑 2,195 episodes，使用原 weighted A*。既有对照为 Success 30.57%、SPL 11.61%。

### S5：部署消融

只有效果达标后才逐项加入：power-of-two threshold、per-layer threshold、int8 weight QAT、nearest upsample、局部 event MaxPool 和更短状态精度。

## 8. 目录和产物约定

正式 run：

```text
dairy/sigma_d_unet_20260924/experiments/<run_name>/
  config.yaml
  command.sh
  source_manifest.json
  driver.log
  train.log
  metrics.jsonl
  event_stats.json
  tb/
  checkpoints/
    best.ckpt
    last.ckpt
  summary.md
```

`best.ckpt` 按离线 val task loss 选择；导航 SR/SPL 另行评测，不得把 best val loss 写成导航最优。

占卡任务放在 `occupancy/` 或 `/tmp`，明确标为非科学任务，不保存 checkpoint、不进入结果汇总。

## 9. 启动命令模板

原版 PONI 训练仍可独立运行：

```bash
cd <original-run-dir>
export PONI_ROOT=/mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
bash "$PONI_ROOT/experiment_scripts/gibson/train_poni.sh"
```

Sigma–Delta 单元验证：

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
python scripts/validate_sigma_d.py \
  --checkpoint pretrained_models/gibson_models/poni_seed_123.ckpt \
  --device cuda:0 \
  --output dairy/sigma_d_unet_20260924/tests/model_contract_gibson_seed123_gpu_fp32.json
```

Sigma–Delta 训练模板：

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
CUDA_VISIBLE_DEVICES=<gpu-list> \
torchrun --standalone --nproc_per_node=<N> \
  train_sigma_d.py --exp-config configs/sigma_d/gibson_sigma_d.yaml \
  LOGGING.output_dir <new-empty-run-dir>
```

实际命令必须固化到每个 run 的 `command.sh`，不能只保留 shell history。

## 10. GPU 资源规则

- 8 张 A100 在数据准备、单元测试和训练期间保持占用。
- 科学训练优先；启动科学训练前，占卡任务必须主动释放足够显存，避免科学实验 OOM。
- 暂时用不到的卡运行无 checkpoint 的 PONI/Sigma–Delta 计算占卡任务。

运行保障：`scripts/gpu_watchdog_sigma_d.py` 已启动 49 小时监控；只有某卡连续两次
低于 40960 MiB 且没有 occupancy companion 时，才在 `/tmp` 启动真实 PONI 计算
占卡。它不会 kill、暂停或修改任何科学训练。PID 与运行记录位于：

```text
dairy/sigma_d_unet_20260924/occupancy/watchdog.pid
dairy/sigma_d_unet_20260924/occupancy/watchdog.log
dairy/sigma_d_unet_20260924/occupancy/watchdog_extension_96h.pid
dairy/sigma_d_unet_20260924/occupancy/watchdog_extension_96h.log
```

原 49 小时进程到期前已启动 96 小时接力监控，以覆盖仍在运行的
Gibson seed234、MP3D full train 和导航评测。

## 11. 已实现模块与接口（2026-09-24）

所有文件都是新增文件，原文件未覆盖：

| 文件 | 责任 | 关键接口 |
|---|---|---|
| `poni/model_sigma_d.py` | 显式状态增量 U-Net、权重折叠与迁移 | `PONISigmaD.forward_step(frame,state,reset_mask)` |
| `poni/dataset_sigma_d.py` | 同 scene/floor/path 的真实累计序列 | `SigmaDTemporalSemanticMapDataset` |
| `default_sigma_d.py` | 独立配置扩展 | `get_cfg_sigma_d()` |
| `train_sigma_d.py` | GT 监督、DDP、AMP、断点、日志 | CLI，run 内自动写 `command.sh` |
| `scripts/validate_sigma_d.py` | 数值契约、reset、梯度测试 | JSON report |
| `scripts/compare_poni_sigma_d.py` | 相同序列离线非劣比较 | JSON report |
| `scripts/validate_online_sigma_d.py` | 在线重复帧、局部 reset、state_dict 契约 | JSON report |
| `semexp/model_pf_sigma_d.py` | 严格 checkpoint 加载、在线 state/reset、原 action 后处理 | `RL_PolicySigmaD` |
| `semexp/eval_poni_sigma_d.py` | Gibson 独立评测入口 | 不修改原 `eval_poni.py` |
| `hlab/policies/pf_exp_sigma_d.py` | MP3D HLab 独立 policy 注册 | `PFExpSigmaD` |
| `hlab/eval_poni_sigma_d.py` | MP3D 独立评测入口 | 只在本进程注册新 policy |
| `configs/sigma_d/gibson_sigma_d.yaml` | Gibson 正式配置 | `C=17,qmax=4095` |
| `configs/sigma_d/mp3d_sigma_d.yaml` | MP3D 正式配置 | `C=23,qmax=4095` |

模型不保存隐藏 episode state；`PONISigmaDState` 由调用方显式传入、显式返回。
训练 adapter 在一次 DDP forward 内按真实 `K` 帧依次调用 `forward_step`，随后丢弃
batch-local state。这一组织方式既保留跨帧计算图，又避免 K 次 DDP forward 后才
backward 的 reducer 生命周期错误。

### 11.1 数据接口验证

Gibson FMM 标签已生成：

```text
data/semantic_maps/gibson/fmm_dists_123/  70 files
dairy/sigma_d_unet_20260924/data_prep/gibson_fmm.log
dairy/sigma_d_unet_20260924/data_prep/gibson_fmm.exit = 0
```

真数据样本验证结果：

```text
frames       (4,17,480,480) float32
object_pfs   (4,17,480,480) float32
area_pfs     (4,1,480,480)  float32
loss_masks   (4,17,480,480) float32
prefix lengths = [1,2,4,6]
adjacent changed semantic elements = [214,482,356]
```

这里修复过一个关键 dtype 兼容点：原 `create_poni_dataset.py` 会在旋转后用
`>0.5` 把 input/GT semantic map 二值化。增量数据线现在执行相同阈值，再调用原
`get_masks_and_labels`；既避免 area 标签的 float bitwise 错误，也保证监督分布一致。

MP3D semantic map 创建已完成，退出码为 0：

```text
dairy/sigma_d_unet_20260924/data_prep/mp3d_semantic_maps.log
dairy/sigma_d_unet_20260924/data_prep/mp3d_semantic_maps.exit
```

MP3D FMM 已生成完成，退出码 0，共 129 个 scene-floor 文件（val 21，train 108）：

```text
dairy/sigma_d_unet_20260924/data_prep/mp3d_fmm.log
dairy/sigma_d_unet_20260924/data_prep/mp3d_fmm.pid
dairy/sigma_d_unet_20260924/data_prep/mp3d_fmm.exit = 0
```

MP3D 真样本为 `frames=(4,23,480,480)`，测试样本相邻帧新增语义元素
`[1705,1915,1860]`。

### 11.2 S0 数值契约结果

报告：

- `tests/model_contract_gibson_seed123_cpu.json`
- `tests/model_contract_gibson_seed123_gpu_fp32.json`

全部检查通过。CPU/GPU FP32 相对原 PONI 的首帧 object 最大绝对误差分别为
`5.84e-6` 和 `1.24e-5`；重复同一帧的 active/event 都为 0，预测差为 0；
changed-map 能产生事件；mixed reset 的 keep/reset 行分别与独立计算一致；58 个
梯度张量均有限且总绝对梯度非零；runtime state 没有进入 `state_dict`。

GPU 契约测试明确关闭 TF32。原因是 Conv+BN 融合前后采用不同 CUDA 累加路径时，
TF32 的微小差异可能跨过量化 round 边界；性能测试若打开 TF32，必须单独记录，
不能混用作数值等价证明。

MP3D 23 通道的 CPU/GPU 契约也全部通过，首帧 object 最大绝对误差分别为
`2.87e-5` 和 `2.86e-5`：

- `tests/model_contract_mp3d_seed123_cpu.json`
- `tests/model_contract_mp3d_seed123_gpu_fp32.json`

### 11.3 烟测与 DDP

单卡 480×480、K=4、两更新烟测成功，完成 forward/backward/validation/checkpoint：

```text
experiments/gibson_sigma_d_smoke/
update 1 train task_loss = 0.014645
update 2 train task_loss = 0.011009
best validation task_loss = 0.016808
```

两卡 DDP 两更新烟测成功，无 reducer、state 或 checkpoint 错误：

```text
experiments/gibson_sigma_d_ddp_smoke/
world_size=2, effective_batch=2
best validation task_loss = 0.006895
```

### 11.4 阈值与动态范围门控

所有配置在同一 32 条 Gibson val 序列、K=4、同一原 checkpoint 上比较。
原 PONI task loss 为 `0.00840025`。

固定 `qmax=15` 的阈值扫描：

| theta | Sigma–Delta task loss | / 原 PONI | saturation rate |
|---:|---:|---:|---:|
| 1/64 | 0.01728462 | 2.0576 | 0.003675 |
| 1/128 | 0.01727061 | 2.0560 | 0.004248 |
| 1/256 | 0.01727531 | 2.0565 | 0.004843 |
| 1/512 | 0.01727446 | 2.0564 | 0.005152 |

结论：降低 theta 不解决问题，反而在固定 qmax 下缩小可表达动态范围。

固定 `theta=1/128` 的 qmax 扫描：

| qmax | Sigma–Delta / 原 PONI task loss | object MAE vs PONI | active rate | saturation rate |
|---:|---:|---:|---:|---:|
| 31 | 2.0574 | 0.0040591 | 0.05690 | 0.003695 |
| 63 | 2.0576 | 0.0040364 | 0.06607 | 0.002844 |
| 127 | 1.9604 | 0.0038916 | 0.07187 | 0.001875 |
| 255 | 1.6702 | 0.0032671 | 0.07555 | 0.000843 |
| 511 | 1.2828 | 0.0022906 | 0.07785 | 0.000206 |
| 1023 | 1.0658 | 0.0015201 | 0.07800 | 0.0000266 |
| 4095 | **0.99907** | 0.0009608 | 0.07138 | 0.00000059 |
| 32767 | **0.99897** | 0.0009584 | 0.07136 | 0 |

报告目录：`experiments/gibson_threshold_sweep/` 与
`experiments/gibson_qmax_sweep/`。

据此主配置改为 `qmax=4095`。这不是“每层运行 4095 次”；它是 12-bit signed
event packet 上限，也可以在硬件上拆成若干较小 packet。32 样本统计中每个 active
位置平均约 41.3 个量化单位，active 位置仅约 7.14%。后续仍需按层统计长尾并研究
首层/少数层较大动态范围，而不能把 `qmax=15` 的明显掉点带入长训练。

保持近似相同动态范围并同步细化内部/输出步长的结果：

| theta / qmax / output theta | task loss / PONI | object MAE | object max abs | active rate | event magnitude/value |
|---|---:|---:|---:|---:|---:|
| 1/128 / 4095 / 1/256 | 0.999067 | 0.0009608 | 0.1984 | 0.0714 | 2.951 |
| 1/256 / 8191 / 1/512 | 0.999538 | 0.0009493 | 0.1782 | 0.0758 | 5.895 |
| 1/512 / 16383 / 1/1024 | 0.999469 | 0.0009454 | 0.1860 | 0.0797 | 11.794 |
| 1/1024 / 32767 / 1/2048 | 0.999545 | 0.0009439 | 0.1860 | 0.0831 | 23.584 |

更细步长只小幅改善 MAE，却让 event magnitude 约按 2 倍递增；所以当前 Pareto
选择仍为 `1/128 + qmax4095 + output 1/256`，而不是盲目追求更细量化。

### 11.5 short-fit、固定样本过拟合与正式门控

早期 `qmax=15` 的 200-update short-fit：

```text
experiments/gibson_sigma_d_overfit128/
val task_loss: 0.015921 @50 -> 0.013567 @200
```

这个 run 跨 epoch 更新了 on-the-fly 样本，因此准确名称是 short-fit，不作为“固定
128 条过拟合 90%”证据；同时它仍受 qmax=15 饱和限制。代码现已增加
`DATASET.freeze_train_samples`，验证 cohort 永远固定。

修正后的 fixed-64 过拟合线已完成 2,000 updates：

```text
experiments/gibson_overfit64_q4095/
first logged train task loss = 0.006815
update 2000 train task loss = 0.001022
relative reduction = 85.0%
best held-out val task loss = 0.007532
```

这证明从 GT label 到 Sigma-D state/STE 再到参数更新的链路可以显著拟合。
held-out validation 没有同比例下降是小样本过拟合实验的预期现象，不用该 run
挑选正式模型。

Gibson 2k 两条线的完整固定 128-sequence validation：

| 配置 | update | val object | val area | val task | event magnitude/value | active rate | saturation rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr=1e-5 | 500 | 0.006050 | 0.001327 | 0.007377 | 3.434 | 0.07597 | 3.65e-7 |
| lr=1e-5 | 1000 | 0.006006 | 0.001353 | 0.007359 | 3.456 | 0.07716 | 4.46e-7 |
| lr=1e-5 | 1500 | 0.006044 | 0.001345 | 0.007389 | 3.627 | 0.07739 | 4.83e-7 |
| lr=1e-5 | **2000** | **0.005934** | **0.001287** | **0.007221** | 3.630 | 0.07587 | 4.06e-7 |
| lr=1e-4 | 500 | 0.006359 | 0.001445 | 0.007805 | 2.384 | 0.06082 | 1.85e-8 |
| lr=1e-4 | **1000** | **0.006035** | **0.001378** | **0.007412** | 2.854 | 0.06257 | 2.49e-8 |
| lr=1e-4 | 1500 | 0.006263 | 0.001456 | 0.007719 | 3.719 | 0.06677 | 2.35e-8 |
| lr=1e-4 | 2000 | 0.006236 | 0.001337 | 0.007572 | 2.833 | 0.06602 | 2.80e-8 |

同一 128 序列的原 PONI task loss=`0.00867524`，未训练 Sigma-D warm-start
为 `0.00866430`（0.99874×）。最终 best checkpoint 的独立比较为：

| checkpoint | Sigma-D task | / 原 PONI | active rate | saturation rate |
|---|---:|---:|---:|---:|
| lr=1e-5, best@2000 | **0.00722150** | **0.83243** | 0.07586 | 4.06e-7 |
| lr=1e-4, best@1000 | 0.00741243 | 0.85444 | 0.06257 | 2.49e-8 |

因此 full-train 主线优先 lr=1e-5（任务 loss 更低），lr=1e-4 保留为更低事件量的
Pareto 对照。这个结论仍是 offline PF，不提前等同为导航 SR/SPL 提升。

### 11.6 MP3D 门控

MP3D 的 32 条固定 K=4 序列结果：原 PONI task loss=`0.00340676`，未经训练
Sigma-D qmax4095=`0.00340452`（0.99934×），object MAE=`8.64e-4`，active
rate=`0.08624`，saturation rate=`4.75e-9`。报告：
`experiments/mp3d_q4095_val32/result.json`。

MP3D 两更新真数据 smoke 已完成，update1/2 train task loss 为 0.002044/0.000992，
best val task loss=0.002904，best/last checkpoint 均成功写出。

首次两条 2k 运行在同一个 `B6ByNegPMKs_0`、index 536 中止：该楼层采样的
最短路长度为 2，无法生成 K=4 序列。这两个失败目录保留原始 traceback，
没有 checkpoint，不计入实验结果：

```text
experiments/mp3d_2k_q4095_lr1e5/
experiments/mp3d_2k_q4095_lr1e4/
```

修复后对原失败 index 的回归测试返回
`frames=(4,23,480,480)`，`scene_fallback_hops=1`，真实路径长度 37，
四帧变化元素数为 `[2868,4070,3758,2674]`。

第一次修复后的 `_rerun1` 已越过原失败点，但在首个 checkpoint 前发现
原 `plan_path()` 偶尔到达 10,000 步保护上限。严格增长检查会拒绝这些
序列，但会浪费时间。`_rerun2` 虽在返回后拒绝了长轨迹，原 helper
仍会先跑完 10,000 步并打印 warning。最终修复是 Sigma-D 数据层内部的
无重复节点、有界 FMM 回溯：一旦 STG 重访网格就立即换终点，不再调用
会返回病态 partial trace 的原 helper。`_rerun1/2` 仅保留诊断日志，
正式 2k 门控目录：

```text
experiments/mp3d_2k_q4095_lr1e5_rerun3/
experiments/mp3d_2k_q4095_lr1e4_rerun3/
```

`rerun3` 已完成。固定 128 序列上的最优结果：

| checkpoint | Sigma-D task | 原 PONI task | Sigma/original | active rate |
|---|---:|---:|---:|---:|
| lr=1e-5, best@500 | **0.00378424** | 0.00394820 | **0.95847** | 0.07475 |
| lr=1e-4, best@1500 | 0.00401614 | 0.00394820 | 1.01721 | **0.06549** |

因此 MP3D full-train 主线选择 `lr=1e-5`；离线 PF task loss 相对原 PONI
改善 4.15%。`lr=1e-4` 只保留为更低事件率的 Pareto 对照。详细曲线、
公平比较和边界说明见 `progress_20260924_2135.md`。

### 11.7 Gibson full train 运行状态

2k gate 确定 `qmax=4095, lr=1e-5` 后，已启动两个独立 seed：

```text
experiments/gibson_full_q4095_lr1e5_seed123/  # GPU 0,1,4,5; effective batch 8
experiments/gibson_full_q4095_lr1e5_seed234/  # GPU 6,7; effective batch 8
```

21:35 检查时，seed 123 已到 `10900/40000`，best validation 为
update 1000 的 `0.00705672`；seed 234 已到 `5340/40000`，best validation 为
update 2000 的 `0.00788331`。两条均无异常，best checkpoint 已保留。学习率
在 20k/32k 降低，所以按计划继续至 40k，不因 1k 后 validation 小幅回升
而提前中止。

MP3D gate 完成后已在 GPU 2/3 启动
`experiments/mp3d_full_q4095_lr1e5_seed123/`，effective batch=8，40,000 updates。

### 11.8 2026-09-26 full-train 结果更新

Gibson seed 123 已完成 40k，best 为 update 1000，固定 1000-sequence
validation task loss=`0.00705672`；40k last=`0.00742554`。长训练降低了 train loss，
但没有超过早期 val best，所以导航使用 `best.ckpt@1000`。

三个当前 best 在各自同 128 序列上的独立复核：

| 模型 | 原 PONI task | Sigma-D task | 改善 | active rate |
|---|---:|---:|---:|---:|
| Gibson seed123 best@1000 | 0.00867524 | **0.00723842** | **16.56%** | 0.07915 |
| Gibson seed234 best@2000 | 0.00817661 | **0.00718096** | **12.18%** | 0.07012 |
| MP3D seed123 best@2000 | 0.00394820 | **0.00376056** | **4.75%** | 0.09380 |

Gibson seed234 当前 `23360/40000`，MP3D seed123 当前 `10760/40000`。
详细曲线、checkpoint 和风险见 `progress_20260926_0003.md`。

Gibson seed123 best 的原协议 1,000-episode 导航评测已启动：

```text
dairy/sigma_d_unet_20260924/navigation/gibson_seed123_full_best/
```

评测使用新 Sigma-D online adapter 与原 Gibson FMM，不回退原 PONI。

## 12. 可复现启动命令

### 12.1 原版 PONI Gibson

原版依赖静态预计算目录，与 Sigma–Delta on-the-fly 目录不要混用：

```bash
mkdir -p /path/to/original_gibson_run
cd /path/to/original_gibson_run
export PONI_ROOT=/mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
bash "$PONI_ROOT/experiment_scripts/gibson/train_poni.sh"
```

实际输入为
`data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square`，原脚本使用
2 GPU、40k updates。

### 12.2 原版 PONI MP3D

```bash
mkdir -p /path/to/original_mp3d_run
cd /path/to/original_mp3d_run
export PONI_ROOT=/mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
bash "$PONI_ROOT/experiment_scripts/mp3d/train_poni.sh"
```

### 12.3 Sigma–Delta Gibson 单卡/多卡

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni

# 单卡
CUDA_VISIBLE_DEVICES=0 python train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d.yaml \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/<run_name>

# 八卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone \
  --nproc_per_node=8 train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d.yaml \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/<run_name>
```

### 12.4 Sigma–Delta MP3D

必须等待 `data_prep/mp3d_fmm.exit=0`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone \
  --nproc_per_node=8 train_sigma_d.py \
  --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/<run_name>
```

每个新 trainer run 自动生成 `config.yaml`、`command.sh`、`source_manifest.json`、
`train.log`、`metrics.jsonl`、TensorBoard 与 `best/last.ckpt`。shell 的外层
`driver.log` 也必须保留。

### 12.5 Gibson Sigma–Delta 完整导航评测

不覆盖原 `semexp/eval_poni.py`，使用新入口与新启动脚本：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/gibson/eval_poni_sigma_d.sh \
  dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed123/checkpoints/best.ckpt \
  gibson_seed123_full
```

输出只写入
`dairy/sigma_d_unet_20260924/navigation/gibson_seed123_full/`。如同名目录已存在，
脚本拒绝覆盖。

### 12.6 MP3D Sigma–Delta 完整导航评测

MP3D 继续使用原 weighted A* 局部规划器，只把 PF policy 切换为
`PFExpSigmaD`：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d.sh \
  dairy/sigma_d_unet_20260924/experiments/<mp3d-full-run>/checkpoints/best.ckpt \
  0 1 mp3d_seed123_full
```

输出只写入
`dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_full/`，11 个 validation part 分别
保存 driver 与 Habitat 日志。

## 13. 当前限制（不能误写成芯片加速结果）

1. PyTorch 首版虽然只把 activation delta 送入下一卷积，但 `conv(delta)` 仍调用
   dense CUDA Conv；它验证算法与训练，不代表 GPU 延迟或能耗已经下降。
2. MaxPool 当前重算完整 pooled map，只保证数值正确；芯片导出阶段再实现受事件影响
   的局部 2×2 窗口更新。
3. `qmax=4095` 已解决当前离线保真，但需要 12-bit packet 或 packet decomposition；
   仍要做逐层动态范围和 SynOps 统计。
4. 当前结果是 offline PF loss 非劣，不等于完整导航 Success/SPL 非劣。导航 adapter
   与 reset 契约已实现，但 Gibson/MP3D episode 评测要等 full checkpoint 后才能作结论。
5. 当前训练没有蒸馏、没有异常时回退原 PONI；原权重仅作为同构 warm-start。
- 占卡任务的 loss、速度或输出不得混入正式实验结论。
- 每分钟采样显存和利用率；进程异常时自动补位，但不能自动重启失败的正式科学实验。

## 14. 已执行记录

### 2026-09-24：环境与数据审计

- 仓库 HEAD：`30682c2bdcd820eec8f72043b2579eb045d547bf`；启动前已有 dirty worktree，未清理已有修改。
- 8 张 A100 均由上一批 49 小时 PONI sustain 占用，实测约 50–59GB/卡；该任务不是科学训练。
- Gibson 原始 semantic maps 齐全：30 scenes、70 个有效 floor maps。
- 原 PONI 预计算训练样本不存在；未启动会占用大量磁盘的 400k static `.pbz2` 生成。
- 运行 `ACTIVE_DATASET=gibson python scripts/precompute_fmm_dists.py`，输出 `data/semantic_maps/gibson/fmm_dists_123/`，共 70 文件，退出码 0。日志：`data_prep/gibson_fmm.log`。
- MP3D 的 90 套 `.glb/.house/_semantic.ply` 齐全；新增兼容链接 `data/scene_datasets/mp3d_uncompressed -> mp3d`。semantic map 与 129 个 FMM scene-floor 文件均已生成，退出码为 0。
- 完成 Gibson/MP3D CPU+GPU 数值契约、单卡烟测、2 卡 DDP 烟测、阈值/动态范围扫描、Gibson 2k gate 和 fixed-64 过拟合。
- 完成在线 adapter 烟测：相同帧二次输入输出 bit-exact，batch 局部 reset 未污染 keep 行，object/area shape 分别为 `(2,17,64,64)`/`(2,1,64,64)`，runtime state 未进入 `state_dict`。
- 上述在线契约的可复现报告为 `tests/online_adapter_gibson_fixed64_cpu.json`，`passed=true`。
- 启动 Gibson seed 123/234 full train 和 MP3D 两个 2k gate rerun3。运行命令分别保存在每个 run 的 `command.sh`、根目录 `launch_pending_sigma_d.sh` 及 `launch_mp3d_rerun3_sigma_d.sh`。

## 15. 当前阻塞和风险

1. PyTorch 中的 `conv(delta)` 仍是 dense CUDA kernel；本阶段验证算法正确性和事件统计，不把 wall-time 当成芯片速度。
2. MP3D 原 HLab policy 不传 `lmb`，当前以 agent local-cell 的大跳变检测 smart-boundary 重定中并按 env reset。多 reset 不改变数值结果，但 full navigation 必须记录 reset 次数和边界前后事件率，并评估后续是否给 HLab 增加精确 frame-id hook。
3. MaxPool 首版为了正确性会重算整张 pool；部署阶段才替换局部事件更新。
4. 当前 GPU 结果不是导航 SR/SPL；必须在 full checkpoint 产生后用新的在线 adapter 显式处理 episode/local-map reset，再和原 PONI 做同 episode 配对评测。
5. MP3D 远端点采样会条件化路径长度分布；这是 K=4 时序监督的必要条件，但需在结果中保留 `path_length` 与 fallback 次数以便审计。

## 16. 2026-09-26 多卡提速记录

MP3D 已从 update 10000 的 epoch 边界切换为 GPU0–3 四卡续跑，保持
`effective_batch=8`；Gibson seed234 将在 update 30000 边界且 GPU4 导航评测完成后，
由条件接力器无损切换为 GPU4–7 四卡续跑。原计算型占位任务已替换为零计算的弹性
显存保留器。完整原因、命令、断点语义、吞吐与 ETA 见：

```text
dairy/sigma_d_unet_20260924/progress_20260926_acceleration.md
```

## 17. 2026-09-27 完整训练与首轮导航结果

Gibson seed234和MP3D seed123均已完成40k训练。首个Gibson seed123 Sigma-D
1,000-episode结果为Success 69.2%、SPL 36.64%，目前低于先前记录的原PONI
73.1%/39.69%，所以尚不能作“导航非劣”结论。同脚本原PONI、seed234 best@32k
和best@2k三条最终导航对照已并行启动。完整checkpoint、曲线、事件率和实验目录见：

```text
dairy/sigma_d_unet_20260924/progress_20260927_final_training.md
```

## 18. MP3D best@2000完整导航评估

MP3D验证最好的`best@2000`已在GPU4–7启动11个scene part、共2,195 episodes的
完整ObjectNav评估。只替换PF网络为Sigma-D，原weighted A*、seed100和Habitat配置
保持不变；最终按逐episode样本加权汇总。启动命令、分片映射和输出说明见：

```text
dairy/sigma_d_unet_20260924/progress_20260927_mp3d_eval.md
```
