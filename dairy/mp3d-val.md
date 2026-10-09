# MP3D val：Habitat-Lab ObjectNav 与 11 个分片的在线评测

MP3D val 同样分两层：离线 `precomputed_dataset_24.0_123_spath_square/val` 用于训练中的 loss/选 ckpt；这里讨论的是 Habitat ObjectNav 的 **11 个验证场景**。它与 Gibson 的 `semexp` 验证实现不同：MP3D 使用 `hlab` + Habitat-Lab `ObjectNav-v1` 数据集和内置 `SUCCESS/SPL/SOFT_SPL`，RedNet RGB-D 分割，以及按场景分片并行。

## 运行前要存在的输入

| 类型 | 目标路径 | 当前服务器状态 |
|---|---|---|
| 3D 场景 | `data/scene_datasets/mp3d/<scene>/<scene>.glb` | **缺失**。这是目前 MP3D 在线评测的关键阻断，需官方 Matterport Habitat 场景。|
| 11 场景验证 episodes | `data/datasets/objectnav/mp3d/v1/val_parts/val_part_{0..10}/{split}.json.gz` 和各自 `content/*.json.gz` | 已下载到 `/mnt/jfzn/shensj/data/datasets/objectnav/mp3d/v1` 并以 symlink 接入。每个 `val_part` 对应一个场景；现有 `val/content` 也有 11 个场景。|
| PONI checkpoint | `pretrained_models/mp3d_models/poni_seed_{123,234,345}.ckpt` | 已有；本章示例用 seed123。|
| 感知模型 | `pretrained_models/rednet_mp3d.pth` | 已有；在线 RGB+Depth → MP3D/ObjectNav 类别。|
| 在线输出 | 新实验目录 `mp3d_objectnav/logs_seed_100_val_part_*.txt`、`tb_seed_100_val_part_*/stats.json` | 尚未运行；每个分片单独记录，每条 episode 独立 metric。|

MP3D JSON 与 Gibson Box 不同：单条 episode 的 `object_category` 是单目标类别，例如 `cabinet`，`scene_id` 指向 `mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb`；episode 还含 `start_position`/`start_rotation`、`info.geodesic_distance`。场景 JSON **顶层**另有 `goals_by_category` 与类别映射表；单条 episode 的 `goals` 有时为空，由 Habitat ObjectNav dataset 按类别解析顶层目标。不能只看 `goals=[]` 就认定目标缺失。实测 11 个 `val_parts` 的 episode 数分别为 `200,201,198,197,200,200,200,201,200,200,198`（part 0 到 10），合计 **2195**；正式完成时应检查 11 个 `stats.json` 的唯一条目数合计是否也是 2195。

## 官方脚本与配置如何层层调用

```text
experiment_scripts/mp3d/eval_poni.sh GPU_A GPU_B
  └─ 每个 val_part 启动一个 CUDA_VISIBLE_DEVICES=... python hlab/eval_poni.py
       ├─ transfer_configs/default.py 合并 transfer_objectnav_mp3d.yaml 和命令行覆盖
       ├─ Habitat registry 找到 TransferEvaluator
       ├─ Habitat ObjectNav-v1 读取 val_parts/<split>/<split>.json.gz + content
       ├─ GlobalAgent(RedNet, Semantic_Mapping, PFExp, PlannerActor)
       └─ 每分片 stats.json + log/TensorBoard
  └─ hlab/merge_results.py 对 11 份 stats.json 汇总
```

[eval_poni.sh](../experiment_scripts/mp3d/eval_poni.sh#L1) 设置 `PONI_ROOT`、`PYTHONPATH`（包含 `dependencies/astar_pycpp`）、`SAVE_ROOT=$EXPT_ROOT/mp3d_objectnav`；part 0..5 背景跑在第一个 GPU，part 6..10 背景跑在第二个 GPU，最后 `wait`。脚本默认 GPU IDs 来自 `$1/$2`。每个进程设置：

```text
--exp-config transfer_configs/transfer_objectnav_mp3d.yaml
TASK_CONFIG.DATASET.DATA_PATH ../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz
EVAL.SPLIT val_part_i
TASK_CONFIG.SEED 100
GLOBAL_AGENT.name PFExp
GLOBAL_AGENT.smart_local_boundaries True
GLOBAL_AGENT.num_local_steps 1
PF_EXP_POLICY.pf_model_path ../pretrained_models/mp3d_models/poni_seed_123.ckpt
PF_EXP_POLICY.pf_masking_opt unexplored
```

`{split}` 由 [TransferEvaluator.eval](../hlab/transfer_evaluator.py#L197) 改为 `val_part_i` 后展开；每个分片用 1 个环境进程，避开 MP3D 单线程评测慢的问题。官方脚本的 6+5 并行模式约需第一卡 20 GB、第二卡 16 GB（[README.md](../README.md#L154) 提示），具体显存仍取决于场景和当前占用。11 个后台进程共用 `mp3d_objectnav` 根目录，但各分片日志/统计名不同；**重复使用同一个实验目录会覆盖同名 `stats.json`**，新实验应在新目录运行。

`hlab/eval_poni.py:main` 调 `run_exp` → [transfer_configs/default.py:get_config](../hlab/transfer_configs/default.py#L1) → [execute_exp](../hlab/eval_poni.py#L39) 通过 Habitat registry 找 `transfer_evaluator` → [TransferEvaluator.eval](../hlab/transfer_evaluator.py#L197)。`TransferEvaluator._synchronize_config` 把 Habitat 动作索引、相机尺寸/深度范围映射给 Planner、RedNet 和 GlobalAgent；[construct_envs](../hlab/utils/env_utils.py#L10) 将 dataset 场景分到环境，并把 `SIMULATOR_GPU_ID` 指向进程内 GPU 0。`TORCH_GPU_ID=0` 也指进程可见的那一张卡。

## 每一步精确数据流

1. Habitat `ObjectNav-v1` 从 val_part 读 scene id、起点、朝向、`object_category` 与 `goals_by_category`；加载 `<scene>.glb`，初始化 RGB/Depth、GPS/Compass、ObjectGoal sensor，任务动作 `[STOP,MOVE_FORWARD,TURN_LEFT,TURN_RIGHT]`。配置见 [objectnav_mp3d.yaml](../hlab/transfer_configs/objectnav_mp3d.yaml#L1)；`MAX_EPISODE_STEPS=500`、`SUCCESS_DISTANCE=0.1`，目标距离按 `VIEW_POINTS` 计算。MP3D 的成功/SPL 由 Habitat-Lab 任务测量，不走 Gibson `MultiObjectGoal_Env.get_metrics`。
2. [TransferEvaluator.eval](../hlab/transfer_evaluator.py#L197) 先 `envs.reset`、`batch_obs`，初始化 [GlobalAgent](../hlab/global_agent.py#L27) 的 `full_map/local_map/full_pose/local_pose/lmb/origins/global_goals`、各环境完成掩码；`TEST_EPISODE_COUNT=-1` 表示评完整分片。
3. [GlobalAgent.preprocess_obs](../hlab/global_agent.py#L433) 将 Depth 按 Habitat 传感器范围还原/清理，RGB+Depth 送 [SemanticPredRedNet.get_predictions](../hlab/utils/rednet_semantic_prediction.py#L438)。RedNet 做归一化、softmax、概率阈值 0.9、深度筛选、THDA→ObjectNav 类别映射，输出 21 类语义 + 1 背景。若 `SEMANTIC_MAPPING.use_gt_segmentation=True`，可切 GT 路径，但那是诊断上界，不是默认复现设置。
4. [GlobalAgent.act](../hlab/global_agent.py#L109) 把 RGB/Depth/语义缩到 160×120，结合 GPS/Compass 相对位姿，经 [Semantic_Mapping.from_config](../hlab/utils/semantic_mapping.py#L7) 包装的 [semexp/model.py:Semantic_Mapping.forward](../semexp/model.py#L230)，累计局部/全局 5 cm 栅格。默认 `GLOBAL_AGENT.map_size_cm=4800`、`global_downscaling=2`，所以 full map 48 m/960×960、local map 24 m/480×480，正好满足 PONI 训练的 24 m 输入。`smart_local_boundaries=True` 控制何时平移局部窗口。
5. `GlobalAgent` 拼 `global_input`（局部基础图、缩小的全局基础图、局部语义图），从 ObjectGoal sensor 取 `goal_cat_id`。策略 registry 选择 [PFExp](../hlab/policies/pf_exp.py#L11)，它继承 [semexp/model_pf.py:RL_Policy](../semexp/model_pf.py#L55)，从 MP3D checkpoint 的 `cfg`/权重构建 PONI U-Net，并按目标类别读 PF 通道。物体 PF 与 area PF 按默认 `area_weight_coef=0.5` 融合，`pf_masking_opt=unexplored` 过滤已探索区域，附近 1 m 可遮罩，取峰值为长程目标。`num_local_steps=1` 意味着每一步都更新该目标。
6. 如已在在线语义图看到目标类别，`GlobalAgent.act` 在 [L365 附近](../hlab/global_agent.py#L365) 用已见目标掩码覆盖长程 PF 目标，置 `found_goal=1`。构造 `planner_inputs={map_pred,pose_pred,goal,new_goal,found_goal,wait}` 交给 [PlannerActor](../hlab/utils/planners.py#L22)。其 `plan_and_act` 做碰撞记录，并由 [_get_stg](../hlab/utils/planners.py#L211) 调 `pyastar.multi_goal_weighted_astar_planner` 求短期目标，再按角度阈值转向/前进/STOP；源码中的 FMM 短期规划分支已注释，FMM 在该类的可达性图计算中另有使用。`move_as_close_as_possible=True` 对 STOP 有额外近距/重规划逻辑。若替换局部规划器，必须保持同一动作索引与地图坐标约定。
7. `TransferEvaluator` 将动作交给 [自定义 VectorEnv](../hlab/utils/vector_env.py#L29)，它支持 `wait_mask`（暂停已完成环境）。Habitat 返回下一观测、奖励、`done` 和指标。完成时按 `scene_id + episode_id` 存一条 `episode_stats`，从 `infos` 抽标量：如 success、spl、softspl、distance_to_goal，加起始 `goal_distance` 和 reward。
8. 进程结束时 [TransferEvaluator.eval](../hlab/transfer_evaluator.py#L505) 在 `TENSORBOARD_DIR/stats.json` 写 **episode id → 指标字典**，并写 TensorBoard/日志。它不像 Gibson 输出一个 `final_eval_stats.json`。对 11 个分片使用 [merge_results.py](../hlab/merge_results.py#L14) 汇总 `success`、`spl`、`distance_to_goal`；success/SPL 乘 100 后输出百分数，距离以米输出。脚本忽略不包含目标字段的 metric，合并时要检查 `# Paths: 11` 与每分片 episode 条数。

## 下载补齐后的启动模板

当前缺 MP3D Habitat 场景，所以以下命令是**场景就绪后的运行模板**，不能把缺场景时报错当成 checkpoint 问题。按 [DATASETS.md](../DATASETS.md) 用官方 Matterport 权限下载 `--task_data habitat`，把真实场景目录 symlink 到 `data/scene_datasets/mp3d`，先确认 11 个 val 场景 `.glb` 可见，再启动：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni
export PONI_ROOT=/mnt/jfzn/shensj/PONI
export PYTHONPATH="$PYTHONPATH:$PONI_ROOT:$PONI_ROOT/dependencies/astar_pycpp"
mkdir -p "$PONI_ROOT/experiments/mp3d_eval_seed123_new"
cd "$PONI_ROOT/experiments/mp3d_eval_seed123_new"
bash "$PONI_ROOT/experiment_scripts/mp3d/eval_poni.sh" 0 1
python "$PONI_ROOT/hlab/merge_results.py" \
  --path_format "$PWD/mp3d_objectnav/tb_seed_100_val_part_*/stats.json"
```

脚本当前 `MODEL_PATH` 写死 seed123；测 seed234/345 时需复制脚本到新的实验目录、修改 `MODEL_PATH`（同时保证 `EXPT_ROOT` 不复用）。脚本 11 个子进程后台并行，shell 最后的 `wait` 不逐个汇报 exit code；可靠复现还要检查 11 份 `stats.json`、11 个日志有无 Traceback/缺场景。

## 可替换模块与联调约束

- `GLOBAL_AGENT.name` 经 [policy_registry](../hlab/policies/policy_registry.py#L26) 选择 `PFExp`、`SemExp`、`NearestFrontierExp` 等长程策略；PONI checkpoint 属于 `PFExp`。XY/theta/action checkpoint 要配套相应评测脚本及 checkpoint `output_type`。
- RedNet 权重可独立训练/替换；改输出类别映射时必须同步 ObjectGoal sensor 类别 id、`GlobalAgent` 语义通道、PONI 23 通道 checkpoint。看起来能运行也可能选错语义通道。
- 5 cm 分辨率、相机高度/HFOV、Depth 单位、`map_size_cm=4800` 与 local 24 m、`PF_EXP_POLICY` 的 area 权重/掩码、Planner STOP/碰撞阈值属于跨模块接口。可单独扫在线超参，但每改一项要用同一 val_parts 与 checkpoint 比较；改网络输入/监督定义需重训 PONI。
- 调 MP3D 在线评测 **不需要** `mp3d_uncompressed` 或离线 `semantic_maps/mp3d`；只要 Habitat 3D 场景、val_parts、RedNet、PONI checkpoint。反过来，从头训练 PONI 需要未压缩语义场景/离线图，却不需要在线 train episodes。
