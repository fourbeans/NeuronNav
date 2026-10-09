# MP3D Sigma-D best checkpoint导航评估启动记录（2026-09-27）

## 目的

优先评估MP3D训练中验证效果最好的Sigma-D checkpoint，而不是40k的最后模型。完整训练的最佳模型位于update 2000，validation task loss为0.0038589297；40k模型为0.0039783358，较best更差，因此本次只把`best@2000`作为首要候选。

## 保持不变的评估协议

- 数据：MP3D ObjectNav v1 validation，共11个scene part、2,195 episodes。
- 随机种子：100。
- 全局策略：`PFExpSigmaD`，只把势函数网络替换为Sigma-Delta U-Net。
- 局部规划器：保留原PONI MP3D的weighted A*，没有替换为SNN波前。
- `smart_local_boundaries=True`、`num_local_steps=1`、`pf_masking_opt=unexplored`。
- 不回退原PONI，不与Gibson FMM配置混用。

## checkpoint

```text
dairy/sigma_d_unet_20260924/experiments/
mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt

global_update=2000
best validation task loss=0.0038589297
```

## 四卡分片

为避免原脚本在两张卡上分别堆叠6/5个Habitat进程，新建四卡启动器：

```text
experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh
```

分配方式：

| GPU | validation parts | 进程数 |
|---:|---|---:|
| 4 | 0, 4, 8 | 3 |
| 5 | 1, 5, 9 | 3 |
| 6 | 2, 6, 10 | 3 |
| 7 | 3, 7 | 2 |

启动命令：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d_4gpu.sh \
  dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt \
  4 5 6 7 mp3d_seed123_best2k_4gpu
```

输出目录：

```text
dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu/
```

每个part分别保存：

- `driver_val_part_N.log`：标准输出与错误；
- `logs_seed_100_val_part_N.txt`：Habitat evaluator日志；
- `tb_seed_100_val_part_N/stats.json`：逐episode最终指标。

## 自动汇总

新增`scripts/summarize_mp3d_navigation_sigma_d.py`。它等待11个`stats.json`全部生成，检查总episode数必须等于2,195，然后按逐episode样本加权汇总Success、SPL和distance-to-goal，而不是对11个场景均值做不加权平均。

汇总产物：

```text
dairy/sigma_d_unet_20260924/navigation/
mp3d_seed123_best2k_4gpu/summary_2195.json
```

## 启动健康检查

启动后11个part进程全部存在，GPU4–7分别使用约16.5、15.8、16.6、11.3GB，GPU利用率约49%–79%。各日志已经进入`plan_and_act`循环，未发现OOM、缺失数据、RuntimeError或Traceback。

最终Sigma-D结果必须再与同seed、同2,195 episodes的原PONI MP3D checkpoint直接比较，不能只引用论文汇总值。

## 完成结果（2026-09-28）

11个part全部完成并通过episode计数检查，总计2,195 episodes。没有发现Traceback、
OOM、RuntimeError、数据缺失或被系统Killed。

| 指标 | 均值 | 标准误 |
|---|---:|---:|
| Success | 30.5239% | 0.9829 pp |
| SPL | 11.8189% | 0.4625 pp |
| Distance-to-goal | 5.2369 | 0.1427 |

汇总文件：

```text
dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu/summary_2195.json
```

这个结果只能说明Sigma-D `best@2000`的绝对导航水平。由于原PONI尚未在同一套
seed=100、2,195 episodes和weighted A*配置下重跑，目前不能据此判断MP3D是否达到
“不低于原版”。后续对比必须用本次原版实测值，不能把论文数字或其他episode口径直接
当作基线。

## 原PONI同协议基线启动（2026-09-28 18:20 CST）

为补齐严格对照，新增独立启动脚本：

```text
experiment_scripts/mp3d/eval_poni_original_8gpu.sh
```

启动命令：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_original_8gpu.sh \
  0 1 2 3 4 5 6 7 mp3d_original_seed123_8gpu
```

这条基线使用原版`hlab/eval_poni.py`、`PFExp`、官方
`pretrained_models/mp3d_models/poni_seed_123.ckpt`和原weighted A*；其余数据、seed、
masking和episode配置与Sigma-D评估一致。11个part按取模方式分到8张A100。启动健康
检查确认11个进程均已进入`plan_and_act`，GPU利用率已起来，且没有启动报错。

输出目录：

```text
dairy/sigma_d_unet_20260924/navigation/mp3d_original_seed123_8gpu/
```

完成后自动生成：

```text
dairy/sigma_d_unet_20260924/navigation/mp3d_original_seed123_8gpu/summary_2195.json
```

## 原版对照完成与最终比较（2026-09-29）

原版11个part于2026-09-28 21:43 CST全部完成，共2,195 episodes；其episode键与
Sigma-D完全一致，且未发现运行错误。

| 指标 | 原PONI | Sigma-D best@2k | 差值 |
|---|---:|---:|---:|
| Success | 30.5695% | 30.5239% | -0.0456 pp |
| SPL | 11.6061% | 11.8189% | +0.2128 pp |
| Distance-to-goal | 5.2764 | 5.2369 | -0.0395 |

Success只相差1个episode；配对McNemar精确检验`p=1.0`。SPL配对差值的近似95%
区间为[-0.568, +0.993] pp。因此当前结果支持“效果近似等价、未观察到实际退化”，
不支持“统计显著超过原版”。正式非劣结论仍需预设非劣界值并补导航seed。
