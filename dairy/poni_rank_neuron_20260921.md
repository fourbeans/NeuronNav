# PONI → SNN 候选目标排序：接口、实现、验证与启动记录

工程：`xty-node3-direct:/mnt/jfzn/shensj/PONI`，原 HEAD `30682c2`。
本次日志：`/mnt/jfzn/shensj/PONI/dairy/poni_rank_neuron_20260921/`。

## 1. 范围与效果承诺的边界

本次是“原 PONI 教师监督一个新的脉冲候选排序学生”，不是声称把 U-Net 所有卷积权重直接转成了 SNN，也不是替换 RGB-D 分割、语义建图或运动控制。学生从语义地图的固定多尺度统计和候选位置特征预测排序，不把教师 PF 分数作为输入。训练目标是列表排序与正负样本顺序，不以势函数像素 MSE 代替导航验证。

默认 `guarded` 对每次实际长期目标格点执行严格一致性检查，不同就回到教师；因此可以优先保护原版效果，但仍需教师计算，不能宣称节能/加速。`neuron` 是不做目标一致性保护的独立实验模式：成功计算候选时跳过教师前向；异常时回退且逐条记录。它没有统计不劣保证。任何小样本的成功、离线排序改善，都不证明全量 Gibson Success/SPL 不下降。

本次低层始终使用原 Gibson FMM，**不与上一轮 SNN 波前一起替换**，避免两个改动相互影响而无法定位原因。MP3D 没有安装场景，遵照用户要求不做 MP3D 数据集实验，也未声称这个 Gibson 接入已验证 MP3D PONI。

## 2. 原版接口分析

### 2.1 模型与输入

- `poni/model.py:99` 起：U-Net 编码器、跳连、上采样解码器；`train.py:90` 的 forward 产生物体势函数和可选面积势函数。
- `semexp/model_pf.py:RL_Policy.do_proc`：原全局策略输入包含本地/全局地图与语义通道；取本地 obstacle、explored 与 `inputs[:,8:-1]` 语义图。构造 free=`obstacle<0.5 && explored>=0.5`；最后 unknown 类不送入 PONI。
- 当前 Gibson 配置语义类别数参数16，处理后输入17通道：free、obstacle、15类语义。本次学生特征维数为 `4C + (C-2) + 8 = 91`，随通道数推导并由 checkpoint 约束。

### 2.2 教师目标并不等于原始物体 PF 的 argmax

`RL_Policy.act` 先运行 PONI，再按配置融合面积项、可选 agent-distance 项和坐标变换，最后调用 `get_action`。本次蒸馏记录的标签来自**同一次原版调用得到的后处理 PF**，不重新猜测这些组合。

`get_action` 选择 `goal_category+2` 通道，按 `pf_masking_opt` 乘 unexplored map，然后在 `mask_nearest_locations` 下将近身方形区域清零，最后按行优先 argmax。原代码没有在这里额外做碰撞/障碍掩码。新分支保留这个定义；不擅自加障碍过滤、改目标容差或修改成功距离来美化指标。

返回值为 `value, action, action_log_probs, rnn_hxs, pred_maps`。当前 value/log_probs 为零，action 是 `[row/H, col/W]`。`semexp/eval_poni.py:377` 和约696行再将 float32 坐标乘地图尺寸后 `int()`，所以存在舍入到邻格的可能。保护判定对照的是这个**最终解码格点**，不只是 PF 峰值坐标。新候选编码使用格内0.25偏移避免自身舍入落到前一格，但进入下游后仍是整数目标格点。

### 2.3 后续导航逻辑不改

- 原 evaluator 得到全局探索目标，构造 goal_map，再送原 FMM 求短期点及控制动作。
- 一旦目标语义已见，原 evaluator 会用已观测物体掩码覆盖探索目标；本分支不干预该逻辑。训练/离线验证剔除 `target_visible=True` 的样本，避免把实际已被覆盖的探索目标当作有意义的标签。
- `pred_maps` 在 reference/shadow/guarded 下完整保留教师原输出。独立 neuron 模式返回空 PF 调试字典，默认评测流程不依赖它；若请求 dense PF 可视化，则明确回退教师并记录，而非伪造稠密 PF 图。
- 关键可空接口：默认未启用距离项时，初始 `dmap` 存在，但后续设为 `None`。学生在此配置下一律用“距离不可用”的特征，连首帧偶然存在的 dmap 也忽略，避免分布切换。启用距离项时要求有效 dmap。

## 3. 新文件与修改原因

所有新增 Python 文件均以 `_neuron.py` 结尾；原文件不修改，上一轮波前分支也不修改。

| 新文件 | 用途 |
|---|---|
| `poni/rank_features_neuron.py` | 固定1/17/65窗口统计、全局语义均值、目标类别、坐标/相对位置/距离可用性；按原掩码生成粗网格和地图边界候选，再细化。无教师分数输入。|
| `poni/rank_model_neuron.py` | 两层可训练 LIF，每次前向重置；默认64隐藏单元、8步、泄漏0.9、阈值1。二值脉冲前向，替代梯度反向，线性发放率读出连续排序值。|
| `semexp/model_pf_rank_neuron.py` | 原 `RL_Policy` 子类，记录原教师标签、实现四种模式；检查教师哈希/特征版本/参数契约，拒绝把随机权重当学生部署。|
| `semexp/eval_poni_rank_neuron.py` | 新评测入口，只在当前进程接入新策略。复用上一轮新环境工厂的场景选择和审计，强制 `PONI_NEURON_MODE=reference`，低层仍为原 FMM。|
| `scripts/run_poni_rank_neuron.py` | 单环境、逐场景采集或 reference/guarded/neuron 对照；新目录、命令/状态/stdout 全记录，任一子任务失败则停止。|
| `scripts/train_poni_rank_neuron.py` | 场景隔离的少量数据蒸馏；默认CPU、固定30轮，不用留出场景反向训练/选择最佳轮次。保存checkpoint与完整训练/留出报告。|
| `scripts/validate_poni_rank_neuron.py` | 脉冲、梯度、状态重置、原掩码/坐标、默认None距离图、保护回退、跳过教师前向、场景泄漏检查。|
| `scripts/summarize_poni_rank_neuron.py` | 完成后比较 SR/SPL/DTS、实际动作和起点/目标，并报告教师调用、学生采用、回退数。未完成的批次不汇总。|

### 3.1 候选排序的具体实现与限制

1. 使用16格步长的全图粗候选和最多256个由地图自身得出的边界候选，过滤原掩码为零的区域；不按教师 PF 挑候选。
2. 学生在粗候选上排序，对前4个位置半径16格范围逐像素细化，再排序选择实际格点。最终候选数量与地图有关。
3. 训练额外加入教师峰值附近3格正样本/难负样本；这只用于训练示例，**验证/推理绝不注入教师峰值**。验证报告候选池 oracle regret，把“候选没覆盖好位置”和“学生排序错误”区分开。
4. 固定 box pooling 是预处理，线性发放率读出是模拟读出；未实现全脉冲传感器/地图流水线。相比原 U-Net，其远距离空间理解容量有限，需实际检验，不能因为使用了 SNN 就推定效果相同。
5. 当前教师权重仍驻留内存，以便异常回退和保留原配置接口；独立 neuron 模式仅在正常前向路径不调用教师。没有测芯片映射、量化、延迟或能耗。

### 3.2 模式

- `reference`：原 PONI 计算与选点，附加日志/可选样本采集。
- `shadow`：两者都算，只执行教师目标。
- `guarded`（默认）：两者都算，只有最终目标格点完全一致才采用学生，否则教师；异常也回退。教师其余输出、原控制链保留。在相同输入和确定性环境下，目标地图不变。
- `neuron`：正常时只跑学生；无候选、非法输出等异常回退教师并记录。这是必要的独立质量消融，不能将 guarded 的效果作为它的效果。

训练checkpoint同时保存实际覆盖的 `supported_categories`。本次仅目标类别4有训练数据，其他类别在两种学生模式均拒绝学生候选并回退教师；不把一个类别的小模型扩用到全部15类。学生初始化使用独立的可恢复Torch随机数作用域，不扰动原评测随机状态。

## 4. 已执行验证及运行记录

### 4.1 CPU回归 `unit_v1` → `unit_v2` → `unit_v3`

全部通过：

- 实际前向脉冲仅为0/1，替代梯度有限；LIF状态不会跨前向污染。
- 合成排序问题120步训练损失从4.636160降至1.393840，确认模型可训练；这是合成测试，不是导航结果。
- 默认两层初始发放率约0.176/0.167；这些只是软件活动统计，不是能耗测量。
- 32种原掩码/全零/边缘案例与原 `get_action` 完全对照；1033个坐标对（2066个坐标分量）完成实际编码/解码回归。
- 7组×2样本的原 `act` 返回接口测试：reference、shadow、guarded一致/不一致/异常、neuron无教师前向/异常回退。
- 验证默认 dmap 从首次有效变成 None 不改变特征语义；空候选拒绝；同场景训练/留出划分拒绝。
- `unit_v2` 新增未训练目标类别拒绝；`unit_v3` 再比较真实完整轨迹：Corozal 500步、Darden 171步，与上一轮原教师/FMM reference 的动作全部一致，episode 起点/朝向/目标也完全一致。这验证的是本次 reference/采集包装，不是学生独立能力。

### 4.2 教师采集

`capture_v1` 首次在 Corozal 第二次策略调用遇到默认 `dmap=None`，退出。根因定位与修复如2.3节所述，错误和后续 BrokenPipe 清理日志完整保留；不是学生性能失败，也没有算作成功实验。

`capture_v2` 已完成。采用两个场景各第1个 episode、seed100、原 seed123 PONI、原 Mask R-CNN、500步上限、480×480地图、5 cm分辨率、seg_interval=3、单GPU单环境。每10次策略调用采一次，每场景最多48张地图。Corozal采集48张，Darden采集18张；Darden剔除1张已见目标样本，最终48张训练、17张留出，全部目标类别4。

### 4.3 真实地图蒸馏结果

`train_v1` 完成最初30轮固定协议；随后增加类别覆盖声明，用完全相同数据/seed/超参数重跑 `train_v2`。这不是依据留出结果调参，两个版本的训练数值一致。当前使用 `train_v2/student.pt`，约1.01万可训练参数。

| 留出17张地图上的指标 | 随机初始化 | 训练后 |
|---|---:|---:|
| 平均 PF regret（教师全图最好分数－学生所选位置的教师分数，越低越好） | 0.486005 | 0.072254 |
| 精确最终目标格点一致率 | 0/17 | 0/17 |
| 候选池内精确最优排序一致率 | 0/17 | 0/17 |
| 候选池 oracle 平均 PF regret | 0.012967 | 0.015273 |

学生学到了一部分排序，但仍未复现教师选点；候选池最优位置也不能保证覆盖全图峰值。候选池会随学生粗排序改变，所以前后 oracle gap 略有差异。**这些结果不支持独立替换 PONI**；更不能拿严格保护下的原版效果来证明学生不劣。默认保护在这17张图上均需要教师目标。

当前可用小样本检查点：`/mnt/jfzn/shensj/PONI/dairy/poni_rank_neuron_20260921/train_v2/student.pt`。详细逐地图 regret、目标坐标、候选覆盖、训练损失与样本清单见 `train_v2/training.json`。

### 4.4 完整导航小样本

`smoke_v1` 已全部完成并生成 `smoke_v1/comparison.json`。单环境依次执行 Collierville 的 reference、guarded、neuron，各1个完整episode、上限500步；该场景未用于本次训练或离线留出。三个运行退出码均为0；各约72秒（含模型/模拟器启动），不用于推断 SNN 速度收益。原始 stdout、完整命令、批次状态均已保留。

| 模式 | Success | SPL | DTS | 实际动作数 | 说明 |
|---|---:|---:|---:|---:|---|
| 原教师 reference | 1 | 1.000000 | 0 | 15 | 原PONI＋原FMM |
| 严格保护 guarded | 1 | 1.000000 | 0 | 15 | 15个实际动作逐个与原版一致；目标不一致时全部回退教师 |
| 独立学生 neuron | 1 | 0.915371 | 0 | 21 | 5个共同时间步动作不同；多走了路，SPL下降0.084629 |

起点、朝向、任务目标一致。**这已经是首版独立学生未达到“效果不比原版差”的小样本反例，不能直接替换原PONI。** 一条成功episode不代表总体成功率100%；本测试只是小范围实现/效果检查。

策略审计包含 evaluator 自动 reset 后的一次额外策略调用，因此 reference/guarded 记录16次，neuron记录22次，不能与实际动作数混淆：

- guarded：前15次候选与教师格点不同，全部回退；第16次是未执行的下个episode（类别2），由未训练类别保护回退；学生实际采用次数为0。
- neuron：前21次正常使用学生、没有教师前向；第22次自动reset到未训练类别2，记录 `unseen_goal_category: 2` 并回退教师，随后评测结束，没有执行这个下一episode动作。
- 所以整个进程日志有1次教师回退；不能把整个运行笼统标为“零教师调用”。而已统计的类别4 episode 的21次策略执行确实使用了学生，不是隐藏教师回退造成的得分。

保护模式保留效果的证据来自动作/目标完全一致，不是学生学得和教师一样；这轮没有保护模式的推理成本收益。下一步若继续改进，应先扩训练数据与空间表示，再用未调参的独立episode验证，不对这个测试episode反复调参追分。

## 5. 完整启动命令

以下命令在 `xty-node3-direct` 执行。日志路径都放在 dairy，复跑时换一个新的末级目录名。不要复用已存在的输出目录。

### 5.1 真正原版：不加载本次任何排序模块

此命令保留原默认场景分配；单环境第1条为 Collierville。原入口没有 `--rank-scene` 参数。任意其他场景的严格配对，请使用下一节的 reference 包装，而不是声称原入口支持新参数。

```bash
cd /mnt/jfzn/shensj/PONI/semexp
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/jfzn/shensj/PONI \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/mnt/jfzn/miniconda3/envs/poni/bin/python eval_poni.py \
  --split val --seed 100 --eval 1 --task_config tasks/objectnav_gibson_box.yaml \
  --version v1.1 --auto_gpu_config 0 --num_processes 1 \
  --num_processes_on_first_gpu 1 --num_processes_per_gpu 1 --sim_gpu_id 0 \
  --pf_model_path ../pretrained_models/gibson_models/poni_seed_123.ckpt \
  -d ../dairy/poni_rank_neuron_20260921/original_new \
  --num_eval_episodes 1 --num_local_steps 1 --exp_name seed_100 \
  --global_downscaling 1 --mask_nearest_locations \
  --pf_masking_opt unexplored --seg_interval 3 \
  > ../dairy/poni_rank_neuron_20260921/original_new.log 2>&1
```

原入口本身不会拒绝旧输出路径，执行者必须手动确认 `original_new` 尚不存在；新分支启动脚本则会直接拒绝目录复用。

### 5.2 采集原教师的真实在线地图（原PONI＋原FMM）

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/run_poni_rank_neuron.py \
  --phase capture --output dairy/poni_rank_neuron_20260921/capture_new \
  --scenes Corozal Darden --episodes 1 --capture-limit 48 --capture-every 10 --gpu 0 \
  > dairy/poni_rank_neuron_20260921/capture_new_driver.log 2>&1
```

每个子目录 `command.json` 记录完整展开命令、cwd、GPU。`rank_audit/launch.json` 记录新增参数；`rank_*.jsonl` 是长期目标决策；`sample_*.npz` 保存语义图/掩码/教师标签和配置；`low_level_fmm/*.jsonl` 记录真实动作与 episode 起点/目标。

### 5.3 小规模CPU蒸馏（使用本次 capture_v2）

```bash
cd /mnt/jfzn/shensj/PONI
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/train_poni_rank_neuron.py \
  --train-glob 'dairy/poni_rank_neuron_20260921/capture_v2/reference/Corozal/rank_audit/sample_*.npz' \
  --val-glob 'dairy/poni_rank_neuron_20260921/capture_v2/reference/Darden/rank_audit/sample_*.npz' \
  --output dairy/poni_rank_neuron_20260921/train_new \
  --epochs 30 --steps 8 --hidden 64 --device cpu --seed 7 \
  > dairy/poni_rank_neuron_20260921/train_new.log 2>&1
```

### 5.4 新版三模式完整episode对照（不改FMM）

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/run_poni_rank_neuron.py \
  --phase smoke --output dairy/poni_rank_neuron_20260921/smoke_new \
  --checkpoint dairy/poni_rank_neuron_20260921/train_new/student.pt \
  --scenes Collierville --episodes 1 --gpu 0 \
  > dairy/poni_rank_neuron_20260921/smoke_new_driver.log 2>&1
```

依次运行 reference/guarded/neuron；每种模式完整episode最多500步。Collierville 不在本次训练或离线留出场景中，但仅1条episode也不能给出统计不劣结论。

已有原版命令要单独切换默认保护版本，入口改成 `eval_poni_rank_neuron.py`，保留所有原参数，额外添加：

```bash
--rank-mode guarded \
--rank-checkpoint /mnt/jfzn/shensj/PONI/dairy/poni_rank_neuron_20260921/train_new/student.pt \
--rank-log-dir /mnt/jfzn/shensj/PONI/dairy/poni_rank_neuron_20260921/guarded_new_audit \
--rank-scene Collierville
```

教师参数/权重、掩码参数不匹配 checkpoint 时会报错，不静默使用错配置学生。`--rank-mode neuron` 才是跳过正常教师前向的消融；不要把它当作已验证可直接部署的默认模式。

### 5.5 检查与汇总

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/validate_poni_rank_neuron.py \
  --output dairy/poni_rank_neuron_20260921/unit_new

/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/summarize_poni_rank_neuron.py \
  --run dairy/poni_rank_neuron_20260921/smoke_new \
  --output dairy/poni_rank_neuron_20260921/smoke_new/comparison.json
```

只有批次全部完成后写出批次 `run_status.json`；汇总脚本缺少完成标志会报错。重点检查 episode 身份、guarded 动作与指标一致性、neuron 指标与回退次数；不只看平均 PF regret。

## 6. 文件保全

本次修改前已记录1163个既有 `.py` 文件哈希（含上一轮新增文件及既存文件）。最终 `sha256sum -c ... --quiet` 返回0，全部内容未改变。只新增本次8个 `_neuron.py`；现有dirty工作树和数据保持原样。

8个新增文件均通过服务器 Python 3.8 `py_compile`。证据文件为 `original_py_sha256.txt`、`original_py_verify_final.log`（成功为空）、`git_status_before.txt`、`git_status_handoff.txt`、`new_files_sha256.txt`；后者包含新源码、本文档和最终小样本权重哈希。`unit_v3/validation.json`、`train_v2/training.json`、`smoke_v1/comparison.json` 分别是最终回归、训练/离线验证和完整导航对照记录。

## 7. 使用本次已训练版本的最短复现命令

不用重新训练即可运行（换新的输出目录名）：

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/run_poni_rank_neuron.py \
  --phase smoke --output dairy/poni_rank_neuron_20260921/smoke_reproduce \
  --checkpoint dairy/poni_rank_neuron_20260921/train_v2/student.pt \
  --scenes Collierville --episodes 1 --gpu 0
```

本轮已经生成 `smoke_v1/comparison.json`；若需要重算，另写新文件：

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/summarize_poni_rank_neuron.py \
  --run dairy/poni_rank_neuron_20260921/smoke_v1 \
  --output dairy/poni_rank_neuron_20260921/smoke_v1/comparison_recheck.json
```

下一阶段应先扩大教师数据的场景/目标类别覆盖，并单独改善学生的语义空间表示与候选覆盖；再做预先规定的多episode非劣检验。当前不应关闭保护并推广到完整导航任务，也不应把小模型参数少直接换算成芯片能耗收益。
