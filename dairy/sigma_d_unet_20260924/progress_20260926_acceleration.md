# Sigma-D 实验提速与运行状态记录（2026-09-26）

## 1. 本次操作目的

此前根据实时吞吐估算，Gibson seed 234 还需约 24–25 小时，MP3D 还需约 70 小时。排查发现并非 A100 显存不足，而是资源组织存在两个问题：

1. 两条正式训练均仅使用 2 个 DDP rank；
2. 为满足“每卡至少占用 50% 显存”而运行的 PONI 占位任务并非静态占显存，而是在同卡持续执行完整 U-Net 前向、反向与 Adam 更新，因而会与正式训练竞争 SM、显存带宽和功耗预算。

本次目标是在不改变模型、监督标签、loss、有效 batch、优化器状态和学习率 update 语义的前提下缩短墙钟时间，并继续维持每张卡 50% 以上显存占用。

## 2. 2026-09-26 00:48 的运行任务

### 2.1 MP3D Sigma-D 完整训练：4 卡续跑

- GPU：物理 GPU 0,1,2,3
- 输出：`dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/`
- 源断点：`dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123/checkpoints/last.ckpt`
- 派生断点：`dairy/sigma_d_unet_20260924/handoff/mp3d_update10000_epoch4_w4.ckpt`
- 当前状态：已从 update 10000 恢复，00:48 到达 update 10080。
- 配置不变量：
  - 原运行：`per_rank_batch=1 × world_size=2 × grad_accum=4 = effective_batch=8`
  - 新运行：`per_rank_batch=1 × world_size=4 × grad_accum=2 = effective_batch=8`
  - `global_update`、Adam 状态、AMP scaler、scheduler、best metric 全部由源断点恢复。

实际启动命令由输出目录下 `command.sh` 自动保存。等价命令为：

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc_per_node=4 train_sigma_d.py \
  --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
  --resume dairy/sigma_d_unet_20260924/handoff/mp3d_update10000_epoch4_w4.ckpt \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k \
  LOGGING.auto_resume False \
  SEED 123 DATASET.seed 123 \
  MODEL.pretrained_path pretrained_models/mp3d_models/poni_seed_123.ckpt \
  MODEL.sigma_d.qmax 4095 \
  OPTIM.lr 0.00001 OPTIM.num_total_updates 40000 \
  DATASET.train_samples_per_epoch 20000 DATASET.val_samples_per_epoch 1000 \
  OPTIM.batch_size 1 OPTIM.grad_accum_steps 2 \
  LOGGING.log_interval 20 LOGGING.eval_interval 1000 LOGGING.ckpt_interval 1000
```

### 2.2 Gibson seed 234 Sigma-D 完整训练：当前 2 卡、计划在 30k 无损升 4 卡

- 当前 GPU：物理 GPU 6,7
- 当前输出：`dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234/`
- 00:48 状态：update 23980 / 40000。
- 当前有效 batch：`1 × 2 × 4 = 8`。
- 当前训练不会为了提速被立即中断。
- 条件接力器：`scripts/promote_gibson_w4_sigma_d.py`
- 接力监控日志：`dairy/sigma_d_unet_20260924/handoff/promote_gibson_w4.log`
- 接力条件必须同时满足：
  1. Gibson 保存 update 30000 且 `next_batch_in_epoch=10000` 的精确 epoch 边界；
  2. GPU4 上的完整导航评测自然结束。
- 接力产物：
  - 派生断点：`dairy/sigma_d_unet_20260924/handoff/gibson_update30000_epoch12_w4.ckpt`
  - 新输出：`dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/`
  - 新 GPU：4,5,6,7
  - 新有效 batch：`1 × 4 × 2 = 8`

最初考虑 update 25000 接力，但实测 Gibson 距 25k 仅约 1.3 小时，而完整导航约需 5–5.5 小时。若等待导航后再从 25k 恢复，会浪费数千个已计算 update。因此接力点调整为 30k，预计导航会先完成，训练到 30k 后可从精确边界无损切换。

### 2.3 Gibson 完整导航评测：GPU4

- GPU：物理 GPU4
- checkpoint：Gibson seed123 完整训练的 `best.ckpt`（update 1000）
- 新输出：`dairy/sigma_d_unet_20260924/navigation/gibson_seed123_full_best_gpu4_restart/`
- 协议：5 scenes，每场景 200 episodes，共 1000 episodes；Sigma-D PONI + 原 Gibson FMM planner。
- 00:48 状态：9 个 episode，FPS 约 10；样本过少，Success/SPL 暂不作为最终结论。
- 原 GPU0 运行保留在 `navigation/gibson_seed123_full_best/`，停止时完成 107 个 episode；它只作为中途观测，不与新运行拼接统计。

启动命令：

```bash
cd /mnt/jfzn/shensj/PONI
CUDA_VISIBLE_DEVICES=4 \
  bash experiment_scripts/gibson/eval_poni_sigma_d.sh \
  dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed123/checkpoints/best.ckpt \
  gibson_seed123_full_best_gpu4_restart
```

### 2.4 资源保障进程（不是科学实验）

- 新 watchdog：`scripts/gpu_watchdog_memory_only_sigma_d.py`
- 新显存保留器：`scripts/gpu_memory_reserve_sigma_d.py`
- watchdog 日志：`dairy/sigma_d_unet_20260924/occupancy/watchdog_memory_only_96h.log`
- 各 GPU 保留器日志：`dairy/sigma_d_unet_20260924/occupancy/memory_only_holders/gpuN.jsonl`
- 00:46 观测：GPU0–7 分别约占 42–45 GB / 80 GB，均超过 50%。
- 新保留器只持有 `uint8` CUDA tensor，不做卷积、反向或优化器更新；每 30 秒检查一次显存。
- 若正式任务增长导致空闲显存低于 30 GB，保留器自动按 256 MiB 块释放显存。
- 原 `poni_gpu_sustain_neuron.py` 计算型占位进程和旧 watchdog 已全部停止。

## 3. 为什么需要 epoch 边界派生断点

`train_sigma_d.py` 的 checkpoint 在 optimizer update 之后、epoch 循环复位之前保存。因此 MP3D update 10000 的源断点记录为：

```text
global_update=10000, epoch=3, next_batch_in_epoch=10000
```

这是 2-rank loader 的完整 epoch 边界。切换到 4 rank 后，每 rank 的 loader 长度变成 5000；若原样恢复，训练器会读取并跳过一个 loader epoch，虽然不会更新权重，但会浪费 I/O 时间。新工具 `scripts/prepare_epoch_boundary_resume_sigma_d.py` 验证 update 和旧 loader 的终止游标后，写出一个新文件：

```text
global_update=10000, epoch=4, next_batch_in_epoch=0
```

模型、optimizer、scheduler、AMP scaler、best value 均保持不变，源 checkpoint 绝不覆盖。Gibson 的 30k 接力使用相同规则。

## 4. DDP 初始化修复

4 卡首次尝试在早期 NCCL barrier 出现 rank-to-device 猜测警告和挂起。`train_sigma_d.py` 的 `_barrier` 已改为：

```python
dist.barrier(device_ids=[torch.cuda.current_device()])
```

该改动只显式声明每个 rank 已绑定的本地 CUDA 设备，不改变模型和训练数学。实测 GPU0–3 的 4-rank NCCL communicator 完成初始化，四个 rank 正常加载同一断点并开始 update。

## 5. 初步吞吐与新 ETA

### MP3D

- 原 2 卡：约 7.2–7.5 秒/update。
- 新 4 卡：update 10020–10080 的三个 20-update 区间分别约 91.5、94.5、87.3 秒，即约 4.55 秒/update。
- 训练步部分加速约 `7.3 / 4.55 ≈ 1.60×`。
- 剩余 29920 update 的纯训练约 37.8 小时；考虑每 1000 update 的验证，当前保守 ETA 约 40–42 小时，而不是原来的约 70 小时。

### Gibson seed 234

- 当前 2 卡约 4.4–4.6 秒/update。
- 预计到 30k 边界（含验证）约 8–9 小时。
- 30k 后切 4 卡，保守按 1.5–1.7× 加速估算，余下 10k 加验证约 8–9 小时。
- 当前总 ETA 约 16–18 小时，而不是原来的约 24–25 小时。最终 ETA 应在 30k 接力后用实际 4 卡吞吐重新计算。

## 6. 已完成但当前没有运行的实验

- Gibson seed123 Sigma-D 40k 完整训练：已完成；best 为 update 1000。
- MP3D 2k gate（lr 1e-4 / 1e-5）：已完成；正式线采用 lr 1e-5。
- 独立 SNN 波前规划小范围验证：已完成。
- 原 Gibson GPU0 导航评测：主动停止并迁移到 GPU4；旧目录不删除、不并入新统计。

## 7. 完整性与失败处理

- 原训练目录、原 checkpoint、原指标文件均未覆盖。
- MP3D 4 卡写入新的续跑目录；原 2 卡最后有效科学断点仍为 update 10000。
- 从 2 卡 live 进程切换时，update 10000 以后尚未落盘的约 960 update 被放弃并从边界重算。这避免了 world-size 改变后从 epoch 中部恢复造成样本分片歧义。
- 两次 4 卡初始化试探没有产生 optimizer update；最终运行从明确的派生断点启动。
- 所有自动接力均有明确 PID、checkpoint update、epoch 游标和导航完成状态门控，不使用模糊进程匹配。
