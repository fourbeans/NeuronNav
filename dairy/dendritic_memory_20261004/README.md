# 简化树突记忆实现日记：2026-10-04

## 1. 本次目标与状态定义

在原 Sigma-D U-Net 上加一个小型、四分支、四时间尺度的脉冲记忆模块，仅修正长期目标候选分数。
空间网络冻结，用真实导航序列直接强化学习训练树突模块，不使用原 PONI 教师，不重复输入地图 T 次。
先 Gibson 小规模跑通和 50k 步试训，再根据开发集结果决定长训练；MP3D 接入在 Gibson 门槛后进行。

**已实现并不代表效果已验证。** 本文中的公式是本版实际定义，试训结果另存 `STATUS.md`。
核心单测、小样本训练和最终全量导航是不同证据等级，不能相互替代。

## 2. 对原工程接口的分析

1. `semexp/model_pf_sigma_d.py::RL_PolicySigmaD` 已封装增量空间推理和 episode reset，继续使用，不复制 U-Net。
2. `semexp/model_pf.py::RL_Policy.act` 在目标势函数、面积项以及原掩码后调用 `get_action`。新子类只在这一选点接口加入候选集和树突残差。
3. Gibson `eval_poni.main(args)` 接收参数对象；新 runner 在**自己的进程内**安装 policy/env factory，退出后恢复原引用。磁盘上原源码不变。
4. `num_local_steps=1` 才能每个实际导航步更新；`global_downscaling=1` 保持 Gibson 地图原点固定。新入口对不支持的配置直接报错，不能静默错用坐标。
5. 原 `VectorEnv` 在 done 后会马上 reset，并用 reset 的 info 替换终止 info。新环境子类在 reset 前捕获终止转移；reset 后单独返回该转移和上一 episode 的评测指标，防止奖励跨 episode 串接。
6. 原环境 `wait` / planner action < 0 不执行物理动作。此时撤销本次预计算的树突状态和地图差分缓存，不写 replay、不增加环境步。
7. 原目标可见接管保持不变。此时记录真实观测和奖励、推进记忆，但把树突候选 action 标记为 -1；该步不能被当成已执行的树突动作训练。
8. 原动作坐标 `float32(row/H)` 再乘 H 取整可能向下偏一个像素。候选 0 直接复用原返回的浮点动作；新增候选采用 `(row+.25)/H`，避免编码舍入落到邻格。

## 3. 神经单元及公式

输入 `x_t ∈ R^(4×32)`；b=0..3 分别表示语义证据、探索进度、运动反馈、决策历史。
每条输入分支具有 128 个输出通道，固定随机连接 mask，每个输出 fan-in=16。

`u[b] = (W[b] ⊙ M[b]) x[b]`，同一分支的投影由四隔室共享，每隔室有独立可学习 gain。

对隔室 k：

```text
λ[k] = 1 - 2^(-k),  k ∈ {2,4,6,8}
d_pre[b,k,j] = λ[k] d_prev[b,k,j] + gain[b,k,j] u[b,j]
z[b,k,j] = clip(round(d_pre[b,k,j]), -7, 7)
d_new[b,k,j] = d_pre[b,k,j] - z[b,k,j]
```

阈值为 1，正负整数发放；泄漏半衰期约 2.4、10.7、44、177 个真实导航步。
胞体先聚合分支事件，再进行同样阈值与减法复位：

```text
s_pre[j] = (1-2^-5) s_prev[j] + Σ[b,k] mix[b,k,j] z[b,k,j]
z_s[j] = clip(round(s_pre[j]), -7, 7)
s_new[j] = s_pre[j] - z_s[j]
h_new[j] = (1-2^-5) h_prev[j] + z_s[j]/32
```

状态量：4×4×128=2048 个树突状态，加胞体 128、读出 128，共 2304 标量/环境。
运行状态不保存在模型 `state_dict` 中；新 episode 清零，不会把某个训练 episode 的记忆当模型参数部署。

训练采用 straight-through estimator（round/clip 的反向近似导数为 1）。
减法复位项在反向 detach，否则 `pre - STE(pre)` 的导数抵消为 0，会人为截断时间梯度。
前向仍然是真实整数发放。梯度通过窗口内的泄漏和读出跨时间传播。

**硬件边界**：泄漏可用定点移位实现；本版是 PyTorch 浮点训练参考，mask 线性仍使用 dense GEMM。
不含 Softmax、RMSNorm、QK 外积、矩阵记忆、自定义 CUDA kernel，但不宣称已经实现芯片部署或事件稀疏 GPU 加速。

## 4. 显式空间历史与输入特征

地图继续保存空间语义。历史使用 25cm 全局坐标网格的稀疏字典，保存访问数、最近访问步、碰撞数、目标尝试数、停滞次数、最近结果。
`xy_global = origin_xy + resolution × [col,row]`，不是用小向量存整张地图的位置。
目标未发生至少 0.1m 距离改善且无新增目标证据，连续 10 个执行步视为一次停滞；同一停滞区间只记一次，不强制切换目标。

四分支固定 32 维，不足通道填零；特征最终裁剪 [-8,8]：

| 分支 | 主要信息 | 信息来源 |
|---|---|---|
| 语义 | 各类别证据增减、目标证据新增/撤销、目标可见、语义变化量 | 预测语义地图相邻真实步差分 |
| 探索 | 新增探索、frontier 数量/变化、当前位置访问/碰撞、障碍比例、时间 | 地图与显式历史 |
| 运动 | 实际 dx/dy/转角、上一动作 one-hot、碰撞、当前长期目标距离和停滞 | sensor_pose、动作反馈、预测地图坐标 |
| 决策 | 目标持续时间、切换次数、尝试及失败统计、相对上一目标位置 | 已执行的长期目标选择历史 |

GT 距离只进入 `learning/reward_dend.py`，不会进入上述特征、候选或评分。
窗口原点变化会清除地图差分比较基准而保留全局坐标历史；Gibson 本版禁用滑窗配置。MP3D 仍需单独适配空间缓存。

## 5. 候选与评分

候选最多 128 个：原后处理 argmax 1 个；0.5m NMS 空间峰值最多 31 个；frontier 最多 48 个；4×8 分区最高分最多 32 个；历史目标最多 16 个。
顺序固定、去重、有效掩码，候选不足不伪造；全零退化图仍保留原 argmax。
这里的“有效”继承原选点的未探索/近邻屏蔽，不擅自加入 GT 可达性过滤。

每候选 24 维特征，包括空间分数、相对位置、地图证据、访问/失败统计、来源类别和上次目标关系。

```text
e(c) = integer_spike(sparse_linear(candidate_features(c))) ∈ R^128
ΔS(c) = wᵀ [e(c), h, e(c)⊙h] + b
Q(c) = S_spatial(c) + ΔS(c)
```

最后一层 w,b 初始化为 0；前面权重正常随机初始化。
初始 argmax 与原空间行为一致；训练后梯度能从非零残差头进入输入和记忆。
Q 是用于 DDQN 的候选动作价值估计，S_spatial 是固定先验，不是教师标签。
执行时只有一次冻结空间前向和一次树突状态更新，没有额外原 PONI 前向。

## 6. 训练数据、损失与生命周期

### 数据

Gibson 使用已安装官方 train 场景做在线采样，共 25 场景；确定性 sha256 排序取约 10%（最少 2）作为内部 dev。
当前 dev 为 `Klickitat.glb`、`Shelbyville.glb`，其余 23 场景训练。官方 val 不用于训练或调参。
每个 actor 独立 episode 流，进程间只共享磁盘发布的只读树突权重和完整 episode 文件。
各 actor 的不同随机流不是多 seed 实验。

Replay 存 `features[T,4,32]`、`candidates[T,128,24]`、`spatial/valid[T,128]`、已执行候选编号、每真实步奖励、terminal，以及少量诊断数组。
不保存整张语义地图/RGB；不将 GT 距离保存在训练输入数组。
只接收自然终止的完整 episode；到达采集预算后让当前 episode 结束，不能把采集截止当环境终止。
容量 500k 活跃回放步，按完整 episode 逐出内存；磁盘原始轨迹保留便于审计。

### 奖励

```text
γ = 0.995
Φ(d) = -clip(d/20,0,1)
r_t = 5·success_terminal + SPL_terminal - 0.001 - 0.01·collision
      + γ Φ(d_(t+1)) - Φ(d_t)
```

真实终止状态 Φ=0；达到任务 500 步也是终止，不 bootstrap。
success/SPL 只在终止步加入，不把旧 episode 指标带入新 episode。
这些 GT 信息仅供训练奖励，不参与部署候选和记忆输入。

### 循环 Double DQN

从 episode 中均匀按环境步采样起点，B=16、学习窗口=32。
先用**当前 online 和 target 权重**从 episode 起点重建状态，前缀无梯度，再对 32 步窗口做 BPTT。
不使用旧 actor 缓存的 hidden state 当当前模型状态，避免权重更新后的状态失配。

```text
y_t = Σ[i=t..u-1] γ^(i-t) r_i
      + γ^(u-t) Q_target(s_u, argmax_c Q_online(s_u,c))
L = mean Huber(Q_online(s_t,a_t), stop_gradient(y_t))
```

u 是向后第三个真实树突决策转移的下一个决策状态；跨目标可见接管区间时按每个真实环境步折扣并累计奖励。
若此前 episode 已终止，bootstrap 项为 0。
仅对实际执行的候选动作计算 loss；目标可见接管步 action=-1，不训练一个未执行的虚构动作。

参数：Adam lr=1e-4；梯度范数裁剪 10；预热 10k 环境步；每新增 64 步一次优化；target 每 1000 更新硬同步。
epsilon 前 10k 为 0.1，随后 90k 线性降至 0.02。评估纯 argmax。
actor 只在 episode 边界加载发布权重；发布和 last ckpt 通过同目录临时文件原子替换，不读取半写文件。
checkpoint 保存 online、target、optimizer、RNG、环境步数、更新数和已见 replay 清单。

## 7. 运行目录和产物

根目录 `/mnt/jfzn/shensj/PONI`：

```text
dendritic_nav/                         新代码，与原 Sigma-D 平行
dairy/dendritic_memory_20261004/
  README.md                           本文件：原理/接口/命令/边界
  STATUS.md                           本轮实际状态和验证结果
  inventory/original_sha256.json       88 个原源码/配置/空间 checkpoint 哈希
  gibson/integration_v01/              端到端前向/终止接口小测，不是正式训练
  gibson/integration_v02/              真实训练轨迹 + 极短优化链路小测
  gibson/pilot_v01/                    50k 步正式试训（启动后才存在）
    scene_split.json                   场景划分
    spatial_checkpoint.json            冻结空间权重路径和 SHA256
    command.sh config.json manifest.json
    launch_train.json logs_train/      实际 PID/GPU/命令和驱动日志
    checkpoints/initial.ckpt           零残差初始模型
    checkpoints/published.ckpt         actor 在 episode 边界加载的版本
    actor_NNN/
      command.sh config.json manifest.json resolved_habitat_args.json
      replay/episode_NNNNNNN.npz       原子写入的完整 episode
      episodes.jsonl                  每 episode SR/SPL、步数、奖励和身份
      visualizations/episode_*.png    轨迹、分支活动、候选分数
      navigation/                     原 evaluator 的本运行专属输出
    learner/
      command.sh config.json manifest.json
      metrics.jsonl tensorboard/       loss/Q/target/梯度/吞吐
      checkpoints/last.ckpt            可续训完整状态
      checkpoints/pilot_final.ckpt     试训末尾，不冒充 best_dev
    dev_baseline_NNN/ dev_dend_NNN/     同场景、同随机流配对开发评估
    development_comparison.json        匹配 episode 的 SR/SPL、差值、bootstrap CI
```

冻结空间权重：

- Gibson：`dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt`
- MP3D 后续接入计划使用：`dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`

上述旧文件不会写入。`best_dev.ckpt` 必须开发评估后才产生；本版不得把 last 命名成最佳结果。

## 8. 启动命令

环境准备：

```bash
ssh xty-node1-direct
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH=/mnt/jfzn/shensj/PONI:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

创建新的正式试训（同名目录存在则拒绝）：

```bash
python -m dendritic_nav.runners.prepare_dend \
  --root /mnt/jfzn/shensj/PONI \
  --output dairy/dendritic_memory_20261004/gibson/pilot_v01 \
  --spatial-checkpoint dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt

python -m dendritic_nav.runners.launch_dend \
  --experiment dairy/dendritic_memory_20261004/gibson/pilot_v01 \
  --gpus 0,1,2,3,4,5,6,7 --actors 23 --learner-gpu 7 --steps 50000 \
  --with-dev-pipeline --dev-actors 20 --baseline-first
```

GPU 列表与数量以现场显存检查为准；启动器预留至少 10GB，按每 actor 14GB 预算准入，不停止其他进程。
所有具体启动命令以各运行 `launch_*.json` 和 `command.sh` 为准。
`--with-dev-pipeline` 自动管理配对 dev；`--baseline-first` 在训练 actor 完成首个 episode 后提前启动纯空间参考（确保显存已稳定），训练模型 dev 则等待训练完成。
因此启用流水线以后不要再手动重复启动下面的 dev 命令。
它只执行有限试训和 dev，完成后写 `pipeline_complete.json`，异常写 `pipeline_failure.json`；不会无条件进入长训练。
初始实验还保存 `source_snapshot.zip`，每个进程保存 `code_sha256.json`，代码版本可追溯。

正式 dev 评估（共 100 episodes，20 路×5，每个 dev 场景合计 50，固定末尾 checkpoint）：

```bash
python -m dendritic_nav.runners.launch_dend \
  --experiment dairy/dendritic_memory_20261004/gibson/pilot_v01 \
  --role dev --no-learner --actors 20 --gpus 0,1,2,3,4,5,6,7 --steps 100000 --episodes 100 \
  --checkpoint dairy/dendritic_memory_20261004/gibson/pilot_v01/learner/checkpoints/pilot_final.ckpt

python -m dendritic_nav.runners.launch_dend \
  --experiment dairy/dendritic_memory_20261004/gibson/pilot_v01 \
  --role dev --no-learner --baseline --actors 20 --gpus 0,1,2,3,4,5,6,7 --steps 100000 --episodes 100

python -m dendritic_nav.runners.report_dend \
  --experiment dairy/dendritic_memory_20261004/gibson/pilot_v01
```

原 Sigma-D（完全不加载本目录）的原入口，以下 dump 必须选未使用的新名字：

```bash
cd /mnt/jfzn/shensj/PONI/semexp
CUDA_VISIBLE_DEVICES=1 python -u eval_poni_sigma_d.py \
  --split val --seed 100 --eval 1 \
  --pf_model_path /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt \
  -d /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/reference_sigma_d_val_new \
  --num_local_steps 1 --exp_name seed_100 --global_downscaling 1 \
  --mask_nearest_locations --pf_masking_opt unexplored
```

原 PONI（完全不加载 Sigma-D 或树突）的独立入口：

```bash
cd /mnt/jfzn/shensj/PONI/semexp
CUDA_VISIBLE_DEVICES=1 python -u eval_poni.py \
  --split val --seed 100 --eval 1 \
  --pf_model_path /mnt/jfzn/shensj/PONI/pretrained_models/gibson_models/poni_seed_234.ckpt \
  -d /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/reference_original_val_new \
  --num_local_steps 1 --exp_name seed_100 --global_downscaling 1 \
  --mask_nearest_locations --pf_masking_opt unexplored
```

以上原版 val 命令用于可独立运行的复现；本次开发阶段不提前跑官方 val 调参。
旧 SNN 波前线仍使用原来的 `_neuron` 入口，未接入本次树突线，本次仍为原 FMM。

检查原代码未变：

```bash
cd /mnt/jfzn/shensj/PONI
python -m dendritic_nav.runners.inventory_dend --root /mnt/jfzn/shensj/PONI \
  --output dairy/dendritic_memory_20261004/inventory/original_sha256.json --verify
```

## 9. 验证标准与下一阶段

- 单测：状态数量/重置、整数事件与梯度、固定 fan-in、零输出头、候选稳定性/退化图/浮点坐标、坐标原点、停滞计数、奖励 telescoping、接管折扣、真实 DDQN backward。
- 集成：自然终止 episode、终止奖励保存、完整回放加载、优化后非零残差、无 NaN、原权重哈希不变。
- 50k 试训：检查 Q/TD/梯度、目标切换、候选分布和记忆事件；dev 中 SR 掉超过 5pp 或 SPL 超过 3pp 为明显回退，先查原因，不直接长跑。
- 不回退/提升的主张必须同 episode 比较 Success/SPL；报告 95% 配对 bootstrap 区间，不把零残差一致当训练后“保证”。
- 试训通过后才安排更长训练和 MP3D 的窗口/原 weighted-A* 接入；暂不做消融、多 seed。
