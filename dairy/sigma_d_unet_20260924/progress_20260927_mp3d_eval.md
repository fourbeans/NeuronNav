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
