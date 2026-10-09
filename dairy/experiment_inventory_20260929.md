# PONI神经形态导航实验总台账（截至2026-09-29 16:04 CST）

服务器与工程根目录：

```text
host: xty-node1-direct
repo: /mnt/jfzn/shensj/PONI
conda: /mnt/jfzn/miniconda3/envs/poni
```

本文中的相对目录均相对于`/mnt/jfzn/shensj/PONI/`。台账覆盖三条已开展的实验线：

1. 独立SNN波前规划，文件后缀`_neuron.py`；
2. 早期PONI候选目标排序SNN学生，文件后缀`_neuron.py`；
3. 完整Sigma-Delta增量脉冲U-Net，文件后缀`_sigma_d.py`。

## 1. 当前总结果

| 实验线 | 数据集 | 当前最好/正式结果 | 原版对照 | 结论 |
|---|---|---|---|---|
| Sigma-D U-Net + 原规划器 | Gibson | seed234 best@32k：Success 72.90%，SPL 42.17% | 当前同协议原PONI seed123：71.70% / 37.56% | 点估计很好，但训练seed不同；必须补原PONI seed234同协议基线 |
| Sigma-D U-Net + 原规划器 | MP3D | seed123 best@2k：Success 30.5239%，SPL 11.8189% | 原PONI seed123：30.5695% / 11.6061% | 2,195 episodes上近似等价；Success只少1条，SPL高0.2128 pp，无显著差异 |
| 严格独立SNN波前 + 原PONI | Gibson | Success 70.60%，SPL 36.62% | FMM：73.10% / 39.69% | 未通过替换门槛；no-path与8邻域距离是主要问题 |
| 严格独立SNN波前 + 原PONI | MP3D | Success 30.66%，SPL 12.58% | weighted A*：30.57% / 11.61% | 本阶段通过；无A*回退，SPL配对区间为正 |
| 候选排序SNN学生 | Gibson小样本 | 1 episode：Success 1，SPL 0.9154 | 原教师：Success 1，SPL 1.0 | 小样本已出现路径效率下降；不应替换PONI |

当前没有科学训练或评估进程运行。8张A100仅有显存保留器，约42.98 GiB/卡，GPU利用率0%。

## 2. 原版基线与导航栈权重

### 2.1 PONI势函数基线权重

| 数据集 | seed | checkpoint | 用途/状态 |
|---|---:|---|---|
| Gibson | 123 | `pretrained_models/gibson_models/poni_seed_123.ckpt` | 主要原版基线；波前和Sigma-D warm-start均使用过 |
| Gibson | 234 | `pretrained_models/gibson_models/poni_seed_234.ckpt` | 已用于波前旧入口/reference；尚缺当前Sigma-D同协议公平导航基线 |
| Gibson | 345 | `pretrained_models/gibson_models/poni_seed_345.ckpt` | 波前旧入口/reference诊断基线 |
| MP3D | 123 | `pretrained_models/mp3d_models/poni_seed_123.ckpt` | Sigma-D正式训练与波前正式对照基线 |
| MP3D | 234 | `pretrained_models/mp3d_models/poni_seed_234.ckpt` | 可用于后续多训练seed实验，当前未训练对应Sigma-D |
| MP3D | 345 | `pretrained_models/mp3d_models/poni_seed_345.ckpt` | 可用于后续多训练seed实验，当前未训练对应Sigma-D |

### 2.2 感知与辅助预测权重

| 模块 | 文件 |
|---|---|
| Gibson语义分割 | `pretrained_models/maskrcnn_gibson.pth` |
| MP3D语义分割 | `pretrained_models/rednet_mp3d.pth` |
| Gibson动作/位置/朝向辅助权重 | `pretrained_models/gibson_models/pred_act_seed_{123,234,345}.ckpt`、`pred_xy_seed_{123,234,345}.ckpt`、`pred_theta_seed_{123,234,345}.ckpt` |
| MP3D位置/朝向辅助权重 | `pretrained_models/mp3d_models/pred_xy_seed_{123,234,345}.ckpt`、`pred_theta_seed_{123,234,345}.ckpt` |
| MP3D动作辅助权重 | `pretrained_models/mp3d_models/pred_act_seed_123.ckpt` |

## 3. Sigma-D正式与门控checkpoint

共同结构：K=4真实累计地图；qmax=4095主配置；GT object/area直接监督；原PONI仅作参数初始化，不使用教师输出蒸馏；原空间U-Net双头与skip结构保持兼容。

### 3.1 Gibson正式/门控checkpoint

| 实验 | 目的 | best checkpoint | best update / val task | last checkpoint | last状态 | 是否建议使用 |
|---|---|---|---:|---|---|---|
| 2k，lr=1e-4 | 低事件率Pareto对照 | `dairy/sigma_d_unet_20260924/experiments/gibson_2k_q4095_lr1e4/checkpoints/best.ckpt` | 1,000 / 0.00741247 | 同目录`last.ckpt` | update 2,000 | 仅低事件率对照；不替代主线 |
| 2k，lr=1e-5 | full-train门控 | `dairy/sigma_d_unet_20260924/experiments/gibson_2k_q4095_lr1e5/checkpoints/best.ckpt` | 2,000 / 0.00722140 | 同目录`last.ckpt` | update 2,000 | 通过门控；作为主线配置依据 |
| full seed123 | 40k正式训练 | `dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed123/checkpoints/best.ckpt` | 1,000 / 0.00705672 | 同目录`last.ckpt` | update 40,000；val 0.00742554 | best做过完整导航；last存在过拟合，不使用 |
| full seed234，前段 | 2卡训练至30k | `dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234/checkpoints/best.ckpt` | 2,000 / 0.00788331 | 同目录`last.ckpt` | update 30,000；val约0.00827995 | best@2k做过完整导航；30k仅接力源 |
| full seed234，后段 | 从30k无损转4卡至40k | `dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt` | **32,000 / 0.00755412** | 同目录`last.ckpt` | update 40,000；val 0.00756525 | **当前Gibson首选Sigma-D checkpoint** |

Gibson相同128序列的独立离线比较：

| checkpoint | 原PONI task | Sigma-D task | 变化 | active rate |
|---|---:|---:|---:|---:|
| full seed123 best@1k | 0.00867524 | 0.00723842 | -16.56% | 0.07915 |
| full seed234 best@2k | 0.00817661 | 0.00718096 | -12.18% | 0.07012 |
| full seed234 best@32k | 0.00817651 | 0.00753743 | -7.82% | 0.07976 |

离线PF loss与导航排序并不完全一致，因此最终选择以完整导航为准。

### 3.2 MP3D正式/门控checkpoint

| 实验 | 目的 | best checkpoint | best update / val task | last checkpoint | last状态 | 是否建议使用 |
|---|---|---|---:|---|---|---|
| 2k，lr=1e-4 rerun3 | 低事件率Pareto对照 | `dairy/sigma_d_unet_20260924/experiments/mp3d_2k_q4095_lr1e4_rerun3/checkpoints/best.ckpt` | 1,500 / 0.00401622 | 同目录`last.ckpt` | update 2,000 / val 0.00407382 | 离线比原版差1.72%；不进入主线 |
| 2k，lr=1e-5 rerun3 | full-train门控 | `dairy/sigma_d_unet_20260924/experiments/mp3d_2k_q4095_lr1e5_rerun3/checkpoints/best.ckpt` | 500 / 0.00378473 | 同目录`last.ckpt` | update 2,000 / val 0.00386094 | 通过门控；离线比原版改善4.15% |
| full seed123，前段 | 2卡训练至10k | `dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123/checkpoints/best.ckpt` | 2,000 / 0.00385893 | 同目录`last.ckpt` | update 10,000 | best保留；10k仅接力源 |
| full seed123，后段 | 从10k无损转4卡至40k | `dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt` | **2,000 / 0.00385893** | 同目录`last.ckpt` | update 40,000 / val 0.00397834 | **当前MP3D首选Sigma-D checkpoint** |

MP3D完整训练中，2k之后没有超过早期best；40k比best高约3.09%，所以导航使用best@2k，不使用last@40k。

## 4. Sigma-D开发性checkpoint与接力checkpoint

这些文件用于验证训练闭环、DDP、过拟合能力或在线adapter，不用于正式SR/SPL结论。

| 实验 | checkpoint目录/文件 | update与结果 | 定位 |
|---|---|---|---|
| Gibson单卡smoke，历史qmax=15 | `dairy/sigma_d_unet_20260924/experiments/gibson_sigma_d_smoke/checkpoints/` | best@1，last@2；best val 0.01680795 | 验证真实480×480前反向、GT loss、TB和保存 |
| Gibson两卡DDP smoke | `dairy/sigma_d_unet_20260924/experiments/gibson_sigma_d_ddp_smoke/checkpoints/` | best@1，last@2；best val 0.00689522 | 验证K步单forward内DDP，无reducer错误 |
| Gibson历史short-fit 128 | `dairy/sigma_d_unet_20260924/experiments/gibson_sigma_d_overfit128/checkpoints/` | best/last@200；val 0.01356672 | qmax=15且旧采样协议；只保留开发历史 |
| Gibson fixed64 overfit/adapter模型 | `dairy/sigma_d_unet_20260924/experiments/gibson_overfit64_q4095/checkpoints/` | best@400，last@2,000；best val 0.00753169 | 用于在线adapter reset/state契约，不用于480导航 |
| MP3D单卡smoke | `dairy/sigma_d_unet_20260924/experiments/mp3d_sigma_d_smoke/checkpoints/` | best@1，last@2；best val 0.00290385 | 验证23通道、5m cutoff、40m²归一化和保存闭环 |
| MP3D四卡接力断点 | `dairy/sigma_d_unet_20260924/handoff/mp3d_update10000_epoch4_w4.ckpt` | update10k，epoch4，next batch=0 | 从2卡切4卡的派生断点；模型/优化器不变 |
| Gibson四卡接力断点 | `dairy/sigma_d_unet_20260924/handoff/gibson_update30000_epoch12_w4.ckpt` | update30k，epoch12，next batch=0 | 从2卡切4卡的派生断点；模型/优化器不变 |

## 5. MP3D失败后重跑目录：全部保留但无checkpoint

| 目录 | 状态/原因 |
|---|---|
| `experiments/mp3d_2k_q4095_lr1e4/` | 首次在短路径`length 2 < K=4`中止；无checkpoint |
| `experiments/mp3d_2k_q4095_lr1e5/` | 同上；无checkpoint |
| `experiments/mp3d_2k_q4095_lr1e4_rerun1/` | 修复短路径后暴露原`plan_path`可到10,000步病态轨迹；无checkpoint |
| `experiments/mp3d_2k_q4095_lr1e5_rerun1/` | 同上；无checkpoint |
| `experiments/mp3d_2k_q4095_lr1e4_rerun2/` | 增加返回后长度拒绝仍无法提前终止helper；在首checkpoint前停止 |
| `experiments/mp3d_2k_q4095_lr1e5_rerun2/` | 同上；在首checkpoint前停止 |
| `experiments/mp3d_2k_q4095_lr1e4_rerun3/` | 最终有界、无重复节点FMM回溯；正式完成并产生checkpoint |
| `experiments/mp3d_2k_q4095_lr1e5_rerun3/` | 最终有界、无重复节点FMM回溯；正式完成并产生checkpoint |

根因与修复记录：`dairy/sigma_d_unet_20260924/mp3d_sampler_fix_record.md`。

## 6. Sigma-D导航结果与目录

### 6.1 Gibson：Sigma-D U-Net，低层仍为原FMM

当前Sigma-D评估协议：5个validation scenes，共1,000 episodes，导航seed100，平均goal distance 4.81058388。

| 模型/用途 | Success | SPL | DTS↓ | 导航结果目录 | checkpoint |
|---|---:|---:|---:|---|---|
| 当前同协议原PONI seed123 | 71.70% | 37.5635% | 1.4384 | `dairy/sigma_d_unet_20260924/navigation/gibson_original_seed123_same_protocol/` | `pretrained_models/gibson_models/poni_seed_123.ckpt` |
| Sigma-D seed123 best@1k | 69.20% | 36.6361% | 1.4374 | `dairy/sigma_d_unet_20260924/navigation/gibson_seed123_full_best_gpu4_restart/` | `experiments/gibson_full_q4095_lr1e5_seed123/checkpoints/best.ckpt` |
| Sigma-D seed234 best@2k | 72.50% | 40.6028% | 1.0362 | `dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best2k/` | `experiments/gibson_full_q4095_lr1e5_seed234/checkpoints/best.ckpt` |
| **Sigma-D seed234 best@32k** | **72.90%** | **42.1746%** | **1.1868** | `dairy/sigma_d_unet_20260924/navigation/gibson_seed234_best32k/` | `experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt` |
| seed123首次导航，已主动迁移 | 不作为最终值 | 不作为最终值 | — | `dairy/sigma_d_unet_20260924/navigation/gibson_seed123_full_best/` | 同seed123 best；仅完成107 episodes，不与重启结果拼接 |

结果文件均位于相应目录的`gibson_objectnav/dump/seed_100/final_eval_stats.json`。当前最好seed234比当前同协议原版seed123高Success 1.20 pp、SPL 4.61 pp，但训练seed不同，尚缺原PONI seed234在同一4.8106 goal-distance cohort上的结果。

### 6.2 MP3D：Sigma-D U-Net，低层仍为原weighted A*

共同协议：11个val parts、2,195 episodes、导航seed100、训练seed123。

| 模型 | Success | SPL | Distance-to-goal↓ | 结果目录 | checkpoint |
|---|---:|---:|---:|---|---|
| 原PONI seed123 | 30.5695% | 11.6061% | 5.2764 | `dairy/sigma_d_unet_20260924/navigation/mp3d_original_seed123_8gpu/` | `pretrained_models/mp3d_models/poni_seed_123.ckpt` |
| Sigma-D seed123 best@2k | 30.5239% | 11.8189% | 5.2369 | `dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu/` | `experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt` |

汇总文件都是`summary_2195.json`。逐episode键完全一致：Sigma-D独有成功205条、原版独有成功206条，McNemar `p=1.0`；Success差值-0.0456 pp，SPL差值+0.2128 pp。结论为近似等价，不能表述为显著提升。

## 7. 独立SNN波前规划实验

这条线路没有学习得到的波前checkpoint；它使用原PONI checkpoint产生长期目标，只替换低层FMM/weighted A*。严格入口每次规划均为`used=neuron`、`reference=null`、`fallback=false`。

### 7.1 正式严格结果

| 数据集 | 原规划器 | 严格SNN波前 | 差值 | 目录/汇总 | 结论 |
|---|---|---|---|---|---|
| Gibson，1,000 episodes，PONI seed123 | Success 73.10%，SPL 39.6886%，DTS 1.3911 | Success 70.60%，SPL 36.6193%，DTS 1.4737 | Success -2.50 pp；SPL -3.07 pp | 原版`dairy/node1_48h_20260923/gibson_ref123_gpu1/`；严格版分布于`gibson_strict123_gpu{0,2,4}/` | 未通过；97 episodes出现no-path，SPL下降显著 |
| MP3D，2,195 episodes，PONI seed123 | Success 30.5695%，SPL 11.6061%，SoftSPL 17.3066% | Success 30.6606%，SPL 12.5846%，SoftSPL 18.8523% | Success +0.091 pp；SPL +0.978 pp | 原版`mp3d_ref123_gpu6/`；严格版分布于`mp3d_strict123_gpu0..7/` | 本阶段通过；无A*回退，SPL配对95%区间为正 |

统一汇总与详细分析：

```text
dairy/node1_48h_20260923/independent_snn_wavefront_summary.json
dairy/node1_48h_20260923/independent_snn_wavefront_results.md
```

严格Gibson场景目录：

| GPU目录 | 场景 |
|---|---|
| `gibson_strict123_gpu0/` | Collierville、Corozal |
| `gibson_strict123_gpu2/` | Darden、Markleeville |
| `gibson_strict123_gpu4/` | Wiconisco |
| `gibson_strict_smoke_gpu0/` | 1-episode严格入口smoke，Success/SPL=1，仅接口验证 |

严格MP3D目录：`mp3d_strict123_gpu0/`至`mp3d_strict123_gpu7/`，分片映射记录在`dairy/node1_48h_20260923/README.md`；`mp3d_strict_smoke_gpu7/`为1-episode接口smoke，Success/SPL=0，不能作为效果结论。

### 7.2 旧可回退波前与多PONI seed诊断

| PONI seed / 模式 | Success | SPL | DTS | 目录 | 解释 |
|---|---:|---:|---:|---|---|
| 123 原FMM | 73.10% | 39.6886% | 1.3911 | `gibson_ref123_gpu1/` | 波前实验正式reference cohort，goal distance 4.6773 |
| 123 旧neuron入口 | 69.70% | 36.5438% | 1.4111 | `pilot_wave_gpu0/neuron/` | 发生FMM回退，不能作为独立SNN证据 |
| 234 原FMM | 70.70% | 41.1669% | 1.3755 | `gibson_ref234_gpu3/` | 已跑，但cohort与当前Sigma-D导航不同，不能补当前公平基线 |
| 234 旧neuron入口 | 70.20% | 38.9305% | 1.2361 | `gibson_wave234_gpu2/` | 发生FMM回退，仅诊断 |
| 345 原FMM | 71.70% | 39.4061% | 1.4012 | `gibson_ref345_gpu5/` | 多PONI seed reference |
| 345 旧neuron入口 | 71.60% | 39.6914% | 1.3927 | `gibson_wave345_gpu4/` | 发生FMM回退，仅诊断 |

MP3D旧`mp3d_wave123_gpu7/`的最终episode指标与严格版逐字节相同（30.6606% / 12.5846%），但审计有12,380次A*回退，因此结论只能引用严格版。其原版目录是`mp3d_ref123_gpu6/`。

### 7.3 波前开发验证目录

| 目录 | 内容 |
|---|---|
| `dairy/neuron_wavefront_20260921/micro_v1/` | 首轮CPU/图搜索/控制接口验证 |
| `dairy/neuron_wavefront_20260921/micro_v2/` | 最终开发期验证；60/60 SciPy oracle、80/80 weighted A*可达性/代价、控制接口通过 |
| `dairy/neuron_wavefront_20260921/gibson_smoke_v1/` | 首次5环境因显存不足失败；保留日志，不计为SNN效果失败 |
| `dairy/neuron_wavefront_20260921/gibson_smoke_v2/` | 三场景、reference/guarded/neuron开发期smoke；后续完整正式实验已取代其结论 |
| `dairy/node1_48h_20260923/` | 波前完整正式实验根目录、审计、事件量、启动脚本和49小时资源记录 |

## 8. 早期PONI候选排序SNN学生

这条线路不是完整Sigma-D U-Net。它用固定多尺度地图/候选特征训练小型两层LIF排序器，教师后处理PF用于标签，属于蒸馏；低层仍为原FMM。

### 8.1 checkpoint

| checkpoint | 说明 | 状态 |
|---|---|---|
| `dairy/poni_rank_neuron_20260921/train_v1/student.pt` | 首次30轮小规模蒸馏，约1.01万参数 | 历史版本；类别覆盖声明加入前 |
| `dairy/poni_rank_neuron_20260921/train_v2/student.pt` | 同数据/seed/超参数重跑并保存supported categories | 该分支当前应使用的checkpoint，但不能独立替换PONI |

### 8.2 数据、离线与导航结果

| 阶段 | 结果 | 目录 |
|---|---|---|
| 教师采集v1 | 因原接口`dmap=None`在第二次调用退出；定位接口问题，不算性能失败 | `dairy/poni_rank_neuron_20260921/capture_v1/` |
| 教师采集v2 | Corozal 48张训练图；Darden 17张留出图；均为目标类别4 | `capture_v2/` |
| 离线排序 | PF regret从随机0.486005降至0.072254；精确目标格点一致率仍为0/17 | `train_v2/training.json` |
| 1-episode导航reference | Success 1，SPL 1.0，15动作 | `smoke_v1/reference/` |
| 1-episode导航guarded | Success 1，SPL 1.0；所有实际不一致均回退教师，学生采用0次 | `smoke_v1/guarded/` |
| 1-episode导航独立student | Success 1，SPL 0.915371，21动作；正常类别4步骤不调用教师 | `smoke_v1/neuron/` |

完整根目录与文档：

```text
dairy/poni_rank_neuron_20260921/
dairy/poni_rank_neuron_20260921.md
```

该分支已经出现路径效率下降的小样本反例，且训练只覆盖一个目标类别，不进入当前主线。

## 9. 无checkpoint的Sigma-D消融、接口和离线实验

| 实验 | 结果摘要 | 目录/文件 |
|---|---|---|
| Gibson阈值扫描，qmax=15 | theta 1/64至1/512均约为原task loss的2.056–2.058倍；证明主要瓶颈是动态范围，不是分辨率 | `experiments/gibson_threshold_sweep/` |
| Gibson qmax扫描 | qmax=31到32767；qmax1023仍劣化6.58%，qmax4095 ratio=0.999067，最终选4095 | `experiments/gibson_qmax_sweep/` |
| Gibson精度/范围Pareto | 更细theta只把object MAE 9.61e-4降到9.44e-4，但event magnitude/value从2.951增至23.584；选择theta=1/128、qmax4095 | `experiments/gibson_precision_range_sweep/` |
| Gibson q4095固定128序列 | 正式训练前的warm-start离线门控 | `experiments/gibson_q4095_val128/result.json` |
| MP3D q4095固定32序列 | 未训练warm-start task ratio=0.999343，active rate=0.08624 | `experiments/mp3d_q4095_val32/result.json` |
| 模型契约CPU/GPU FP32 | Gibson、MP3D权重映射、zero-delta、signed event、局部reset、梯度、state排除均通过 | `dairy/sigma_d_unet_20260924/tests/model_contract_*_{cpu,gpu_fp32}.json` |
| 初始默认GPU契约 | `model_contract_gibson_seed123.json`为失败历史：非FP32数值容差导致dense/mixed keep未通过；后续显式GPU FP32复验通过 | `tests/model_contract_gibson_seed123.json` |
| 在线adapter契约 | repeat、局部reset、shape、state不进checkpoint均通过 | `tests/online_adapter_gibson_fixed64_cpu.json`、`online_adapter_record.md` |

## 10. Sigma-D目录与产物索引

主根目录：`dairy/sigma_d_unet_20260924/`

| 子目录/文件 | 内容 |
|---|---|
| `README.md` | 算法公式、结构、数据流、训练、启动和实验阶段总说明 |
| `experiments/` | 所有训练、门控、扫参、smoke和失败重跑目录 |
| `navigation/` | Sigma-D与原PONI的完整导航评估 |
| `handoff/` | 2卡到4卡的epoch边界派生断点与接力日志 |
| `tests/` | 模型契约和在线adapter JSON |
| `data_prep/` | 数据准备记录 |
| `occupancy/` | watchdog和纯显存保留器；不是科学实验，不产出checkpoint |
| `mp3d_sampler_fix_record.md` | MP3D时序路径采样错误与三轮不覆盖重跑记录 |
| `online_adapter_record.md` | Gibson/MP3D在线接入、reset与启动说明 |
| `progress_20260927_final_training_sigma_d.md` | 40k训练和Gibson导航最终记录 |
| `progress_20260927_mp3d_eval_sigma_d.md` | MP3D Sigma-D与原版完整评估记录 |
| `progress_20260929_results_and_remaining_sigma_d.md` | 当前结果、统计解释和待补实验 |

每个成功训练目录一般包含：`config.yaml`、`command.sh`、`source_manifest.json`、`driver.log`、`train.log`、`metrics.jsonl`、`tb/`、`checkpoints/{best,last}.ckpt`。不要只复制checkpoint而丢失同目录配置与日志。

## 11. 当前应保留/部署候选

| 用途 | 应使用文件 | 不应使用 |
|---|---|---|
| Gibson Sigma-D候选 | `experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt` | 同目录`last.ckpt`、seed123 last、任何smoke/overfit checkpoint |
| MP3D Sigma-D候选 | `experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt` | 40k `last.ckpt`、lr1e-4 checkpoint |
| MP3D独立SNN波前 | 严格入口代码与`node1_48h_20260923/mp3d_strict123_gpu*/`审计；波前本身无学习checkpoint | 旧`mp3d_wave123_gpu7`不能作为无回退证据 |
| Gibson独立SNN波前 | 只保留研究分支，先修no-path与距离stencil | 当前严格版不能替代FMM |
| 早期排序学生 | 如复现实验用`train_v2/student.pt` | 不作为PONI正式替换模型 |

## 12. 尚缺实验

1. Gibson原PONI seed234在当前Sigma-D导航协议、同一goal-distance cohort上的1,000-episode公平基线。
2. Gibson和MP3D至少再补2个成对导航seed，并在运行前固定非劣界值。
3. 状态保留vs每步reset、增量事件vs稠密逐帧输入消融。
4. 逐层事件率、SynOps、状态内存、软件延迟和类脑芯片能耗代理。
5. MP3D组合“Sigma-D U-Net + 严格SNN波前”的2,195-episode端到端实验。
6. Gibson先修严格波前no-path、高阶stencil和事件早停，再做联合实验。
7. 论文级稳健性需要更多训练seed；简化树突记忆应在以上基线闭环后再单独引入。
