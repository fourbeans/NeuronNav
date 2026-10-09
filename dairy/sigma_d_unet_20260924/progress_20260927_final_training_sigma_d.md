# Sigma-D 完整训练与首轮导航结果（2026-09-27）

## 1. 当前结论

1. Gibson seed234 与 MP3D seed123 的 Sigma-D 完整训练均已完成 40,000 updates。
2. Gibson 在 update 30,000 从 2 卡无损接力到 4 卡；MP3D 从 update 10,000 的 epoch 边界转为 4 卡。两条训练均保持 `effective_batch=8`，没有修改模型、loss、optimizer update 或学习率里程碑的定义。
3. 全部训练、接力和首轮 1,000-episode Gibson 导航评测均未发现 traceback、OOM 或 NCCL error。
4. Gibson seed123 Sigma-D `best@1000` 的完整导航结果为 Success 69.2%、SPL 36.64%。相对先前记录的原 PONI 73.1% / 39.69%，分别低 3.90 和 3.05 个百分点。因此当前不能宣称导航效果不低于原 PONI。
5. 为排除历史基线口径和训练 seed 的影响，已并行启动同脚本原 PONI、Gibson seed234 `best@32000` 与 `best@2000` 三条 1,000-episode 对照。

## 2. 完整训练产物

### 2.1 Gibson seed234

两阶段目录：

```text
dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234/
dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/
```

接力过程：

```text
2026-09-26 09:26:58  2卡完成 update 30000 validation并保存断点
2026-09-26 09:27:27  生成epoch边界派生断点
2026-09-26 09:27:28  GPU4,5,6,7启动4卡续跑
2026-09-26 16:53:12  完成update 40000 validation并保存last.ckpt
```

关键验证结果：

| update | val task loss | active event rate | saturation rate | 备注 |
|---:|---:|---:|---:|---|
| 2,000 | 0.00788331 | 约0.070 | 1.65e-8 | 2卡阶段旧best |
| 30,000 | 0.00827995 | — | — | 接力边界 |
| 31,000 | 0.00757781 | 0.08116 | 1.99e-8 | 4卡续跑 |
| **32,000** | **0.00755412** | 0.08006 | 1.99e-8 | **最终best** |
| 40,000 | 0.00756525 | 0.07989 | 1.99e-8 | final/last |

最终选择：

```text
best checkpoint:
dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt

best update: 32000
best val task: 0.0075541154
```

相对旧 `best@2000`，新的 validation task loss 改善约 4.18%。32k 后继续训练没有进一步改善，40k 只比 best 高约 0.15%，属于稳定平台而非严重发散。

### 2.2 MP3D seed123

两阶段目录：

```text
dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123/
dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/
```

完成时间：

```text
2026-09-26 00:42:33  从update 10000派生断点恢复4卡训练
2026-09-27 12:59:33  达到update 40000
2026-09-27 13:07:01  完成validation并保存last.ckpt
```

关键验证结果：

| update | val task loss | active event rate | saturation rate | 备注 |
|---:|---:|---:|---:|---|
| **2,000** | **0.00385893** | — | — | **最终best，来自原2卡阶段** |
| 10,000 | 0.00398291 | — | — | 4卡接力源断点 |
| 18,000 | 0.00391328 | 0.09475 | 1.20e-8 | 4卡阶段局部最低，但未超过2k |
| 32,000 | 0.00398172 | 0.09564 | 1.27e-8 | 学习率降至1e-7 |
| 40,000 | 0.00397834 | 0.09738 | 1.27e-8 | final/last |

最终选择：

```text
best checkpoint:
dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt

best update: 2000
best val task: 0.0038589297
```

MP3D 在早期 2k 后没有继续改善；40k validation 比 best 高约 3.09%。因此后续导航必须使用 `best@2000`，不能使用 `last@40000`。

## 3. 多卡提速是否正常

### MP3D

- 原 2 卡阶段约 7.2–7.5 秒/update。
- 4 卡阶段常见约 3.7–4.6 秒/update。
- 30k续跑连同30次完整validation约36小时25分钟完成；明显短于原先约70小时估计。
- 有效 batch 始终为8：`1×2×4` 切换为 `1×4×2`。

### Gibson

- 30k后的10k updates与10次完整validation约7小时26分钟完成。
- 4卡训练区间约2.3秒/update，validation约4分50秒/次。
- 有效 batch 始终为8。

## 4. Gibson seed123 首轮完整导航

结果文件：

```text
dairy/sigma_d_unet_20260924/navigation/
  gibson_seed123_full_best_gpu4_restart/
  gibson_objectnav/dump/seed_100/final_eval_stats.json
```

协议：

- Gibson validation 5 scenes；
- 每 scene 200 episodes，共1,000 episodes；
- seed=100；
- Sigma-D checkpoint为Gibson seed123 `best@1000`；
- 低层规划器仍为原FMM；
- 不回退原PONI。

最终指标：

| 指标 | Sigma-D seed123 best@1k | 先前记录原PONI | 绝对差值 | 相对差值 |
|---|---:|---:|---:|---:|
| Success | 69.20% | 73.10% | -3.90 pp | -5.34% |
| SPL | 36.64% | 39.69% | -3.05 pp | -7.69% |
| DTS | 1.4374 | — | — | — |
| Goal distance | 4.8106 | — | — | — |

Success的二项分布Wilson 95%区间约为66.27%–71.98%。历史原PONI的73.1%在该区间之外，但历史值自身也有抽样误差，因此最终判断应以当前正在运行的同脚本原PONI 1,000-episode结果为准。

这个结果揭示了一个明确问题：固定128序列上Sigma-D seed123的平均PF loss曾比原PONI低16.56%，但导航SR/SPL反而更低。说明当前pixel-wise PF监督和候选目标峰值/排序、可达性以及闭环轨迹质量并不完全一致；只优化平均MSE不足以保证导航指标。

## 5. 2026-09-27启动的最终对照

### 5.1 原PONI同协议基线

```text
GPU0
dairy/sigma_d_unet_20260924/navigation/gibson_original_seed123_same_protocol/
checkpoint: pretrained_models/gibson_models/poni_seed_123.ckpt
seed: 100
episodes: 1000
```

### 5.2 Sigma-D seed234最终best

```text
GPU1
dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best32k/
checkpoint: gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt
best update: 32000
```

### 5.3 Sigma-D seed234早期best

```text
GPU2
dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best2k/
checkpoint: gibson_full_q4095_lr1e5_seed234/checkpoints/best.ckpt
best update: 2000
```

### 5.4 seed234 `best@32000` 固定128序列离线复核

已完成，结果文件：

```text
dairy/sigma_d_unet_20260924/experiments/
  gibson_full_q4095_lr1e5_seed234_w4_from30k/
  best_vs_original_val128.json
```

| 指标 | 原PONI seed234 | Sigma-D best@32k | Sigma/original |
|---|---:|---:|---:|
| task loss | 0.00817651 | 0.00753743 | 0.92184 |

即Sigma-D在这128条序列上改善7.82%，active event rate为0.07976，saturation
rate为2.20e-8。但旧`best@2000`在同类128序列复核中曾达到0.00718096，反而更低。
新32k checkpoint在完整1,000条validation上更好、在该128子集上更差，说明checkpoint
排序对样本子集敏感，也进一步说明必须以1,000-episode闭环导航而不是小规模PF loss决定
最终模型。

这三条对照均使用seed100、相同5 scenes、相同FMM和相同episode配置。预计约4.5小时完成。比较目标：

1. 原PONI本次实测是否复现73.1%/39.69%；
2. seed234是否优于已完成的seed123 Sigma-D；
3. 32k的offline改善是否转化为SR/SPL改善；
4. 若32k loss更低但导航更差，则下一轮loss必须加入候选点排序/峰值保持/可达性约束，而不是继续单纯延长训练。

## 6. 当前GPU与错误状态

- 在新对照启动前，8张GPU均只运行零计算显存保留器，约42.98GB/卡，GPU利用率0%。
- 新对照已使用GPU0–3；显存保留器会自动释放空间，不参与计算。
- 训练目录、导航目录和driver log中没有发现OOM、Traceback、NCCL error或被系统Killed记录。
- GPU4–7继续由零计算显存保留器维持50%以上显存，不生成科学结果。

## 7. 下一步判据

在三条1,000-episode对照完成前，不根据早期几十个episode调整模型。完成后按以下顺序判断：

1. 先比较同脚本原PONI与三个Sigma-D checkpoint的Success/SPL；
2. 再按episode做配对成功/失败翻转和SPL差值统计，而不只比较均值；
3. 若Sigma-D仍低于原版，保持当前代码与checkpoint不动，单独设计候选目标排序和峰值保持loss实验；
4. 不用`last.ckpt`替代`best.ckpt`，也不通过回退原PONI掩盖退化。

## 8. Gibson同协议最终结果（2026-09-28）

三条新增对照均已完成，每条都是相同seed=100、相同5个Gibson validation
scene、相同1,000 episodes、相同原FMM局部规划器和相同goal-distance序列。日志中未发现
Traceback、OOM、RuntimeError、数据缺失或被系统Killed。

| 模型 | Success | SPL | DTS | 相对本次原版Success | 相对本次原版SPL |
|---|---:|---:|---:|---:|---:|
| 原PONI seed123 | 71.70% | 37.56% | 1.4384 | — | — |
| Sigma-D seed234 best@2k | 72.50% | 40.60% | 1.0362 | +0.80 pp | +3.04 pp |
| **Sigma-D seed234 best@32k** | **72.90%** | **42.17%** | **1.1868** | **+1.20 pp** | **+4.61 pp** |
| Sigma-D seed123 best@1k | 69.20% | 36.64% | 1.4374 | -2.50 pp | -0.93 pp |

最终应选Sigma-D seed234 `best@32000`。它相对本次严格同协议原版不仅没有退化，
Success提高1.20个百分点，SPL提高4.61个百分点（相对提升约12.28%）。`best@32000`
又比同seed的`best@2000`提高0.40个百分点Success和1.57个百分点SPL，说明后段训练的
完整validation改善最终确实转化成了更好的闭环导航效率。

结果文件：

```text
dairy/sigma_d_unet_20260924/navigation/gibson_original_seed123_same_protocol/gibson_objectnav/dump/seed_100/final_eval_stats.json
dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best2k/gibson_objectnav/dump/seed_100/final_eval_stats.json
dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best32k/gibson_objectnav/dump/seed_100/final_eval_stats.json
```

注意：Success仅提高12个episode，逐索引比较中有122个原版失败/Sigma-D成功和110个
原版成功/Sigma-D失败；因此Success优势较小，不能单凭单seed宣称统计显著。SPL的提升
更大且工程意义更明确，下一阶段仍应补多个导航seed确认方差。
