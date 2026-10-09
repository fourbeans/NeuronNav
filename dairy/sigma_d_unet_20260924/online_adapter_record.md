# Sigma-D 在线 PONI adapter 记录

## 目的与边界

把已训练的增量 Sigma-D U-Net 接到 Gibson Semantic Exploration 和 MP3D HLab，
同时保留原目标点后处理与各数据集的低层规划器。本阶段只验证接口契约，
不在 full checkpoint 训练完成前提前报告 SR/SPL。

严格边界：

- 只接受 `train_sigma_d.py` 写出且 `model_family="sigma_d"` 的 checkpoint。
- 不加载、不回退原 PONI。
- 不修改 `semexp/model_pf.py`、`semexp/eval_poni.py`、`hlab/policies/pf_exp.py`。
- runtime state 是 Python 成员，不注册为 parameter/buffer，不写入 checkpoint。

## 新模块

| 文件 | 作用 |
|---|---|
| `semexp/model_pf_sigma_d.py` | 严格 checkpoint 加载、跨导航步 state、逐环境 reset、原 PF 距离/目标点后处理 |
| `semexp/eval_poni_sigma_d.py` | 只在该进程把 Gibson evaluator 的 policy 换成 `RL_PolicySigmaD` |
| `hlab/policies/pf_exp_sigma_d.py` | 注册 MP3D `PFExpSigmaD` |
| `hlab/eval_poni_sigma_d.py` | 只在该进程导入新 HLab policy |
| `experiment_scripts/gibson/eval_poni_sigma_d.sh` | Gibson 独立启动、拒绝覆盖 run 目录 |
| `experiment_scripts/mp3d/eval_poni_sigma_d.sh` | MP3D 11-part 独立启动、拒绝覆盖 run 目录 |

## 数据流与 reset

```text
mapping global_input
  -> original do_proc: [free, obstacle, semantic categories]
  -> Sigma-D forward_step(current_map, previous_state, reset_mask)
  -> object PF + area PF
  -> original PONI distance weighting / unexplored masking / argmax goal
  -> Gibson FMM or MP3D weighted A*
```

episode `mask<=0` 会只 reset 对应 batch row。Gibson 当前命令使用
`global_downscaling=1`，坐标系固定。MP3D smart boundary 没有在原 policy 接口暴露
`lmb`，adapter 跟踪 agent local-cell；当单步 L-infinity 跳变超过
`max(8 cells, 0.1*map_width)` 时触发对应行 dense reset。false positive 只增加计算，
因为 dense init 是同一模型的精确首帧路径；它不改变权重或预测定义。

adapter 暴露 `last_reset_mask` 和 `reset_statistics`，分别累计 full request、manual row、
frame-jump row 和 episode row，供完整导航记录。

## 已做验证

用 Gibson fixed-64 best Sigma-D checkpoint、B=2、17 通道、64×64 输入做快速契约测试：

```text
repeat_exact = true
partial_reset_keep_exact = true
object_shape = (2,17,64,64)
area_shape = (2,1,64,64)
state_in_checkpoint = false
```

`PFExpSigmaD` 已在 `poni` 环境中通过 Python 编译与 HLab registry 导入。
可复现脚本和报告：

```text
scripts/validate_online_sigma_d.py
dairy/sigma_d_unet_20260924/tests/online_adapter_gibson_fixed64_cpu.json
passed = true
```

## 启动命令

Gibson：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/gibson/eval_poni_sigma_d.sh <sigma-d-best.ckpt> <new-run-name>
```

MP3D：

```bash
cd /mnt/jfzn/shensj/PONI
bash experiment_scripts/mp3d/eval_poni_sigma_d.sh \
  <sigma-d-best.ckpt> <gpu-a> <gpu-b> <new-run-name>
```

两个脚本均将日志写入 `dairy/sigma_d_unet_20260924/navigation/`，同名目录存在时
立即退出，不覆盖旧试验。
