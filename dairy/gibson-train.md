# Gibson train：从语义场景到 PONI checkpoint

这份文档只讲 **Gibson 离线监督训练**，不把在线 Gibson `train` episodes 当成 PONI 的监督数据。读完应能回答：哪 25 个场景用于训练、30 张场景图从何而来、每条监督样本长什么样、loss 在哪里算、最终 checkpoint 如何喂给导航器。

## 数据清单与位置

| 阶段 | 当前/目标路径 | 内容与用途 |
|---|---|---|
| 3D 场景/语义 | `data/scene_datasets/gibson_semantic` → `/mnt/jfzn/shensj/data/datasets/gibson/habitat/gibson` | Habitat 用的 `.glb`、语义 `.ply` 等；依据 [DATASETS.md](../DATASETS.md) 转换语义。不是监督样本。|
| 场景划分 | [poni/constants.py:SPLIT_SCENES](../poni/constants.py#L5) | Gibson train 25 个、val 5 个。30 是提取 train+val 俯视图的**场景总数**，不是 30 条 episode 或 30 条训练样本。|
| 俯视图 | `data/semantic_maps/gibson/semantic_maps/<scene>.h5` | 当前已有 30 个 H5。每个楼层含 `mask`、`map_heights`、`map_instance`、`map_semantic`、可视化 RGB；旁边 `semmap_GT_info.json` 保存世界坐标偏移/分辨率/楼层高度。|
| FMM 距离 | `data/semantic_maps/gibson/fmm_dists_123/<scene>_<floor>.pbz2` | 每个类别到物体的地面最短距离。当前不存在，训练前需生成。|
| 离线样本 | `data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/{train,val}/<scene>_<floor>/sample_*.pbz2` | 当前不存在；`train.py` 读它，而不是直接读 H5。|
| 模型输出 | 自选实验目录 `checkpoints/{best,last}.ckpt`、`train.log`、`tb/train` | `best` 按离线 val loss，`last` 为最近一次保存。训练脚本默认同名目录的 `last.ckpt` 会自动 resume。|

容易混淆的一点：**Box ObjectNav 包的 `train/content` 只有 22 个场景文件**，比离线训练常量中的 25 个少 `Shelbyville`、`Stockman`、`Tolstoy`。这不代表 PONI 离线训练只用 22 个场景；训练脚本不读取 Box 的 train episodes，而是按 `SPLIT_SCENES['gibson']['train']` 从俯视 H5 选 25 个场景、再按有效楼层采样。当前 30 个 H5 包含上述三个场景。

### 场景到俯视图的转换

[create_semantic_maps.py](../scripts/create_semantic_maps.py#L1) 由环境变量 `ACTIVE_DATASET=gibson` 选 Gibson 常量：`SCENES_ROOT=data/scene_datasets/gibson_semantic`，楼层边界采样步长 `SAMPLING_RESOLUTION=0.10 m`、12 workers。主程序在 [L661](../scripts/create_semantic_maps.py#L661) 搜 `.glb`，要求配套 `_semantic.ply`，再按 `SPLIT_SCENES['gibson']['train'] + ['val']` 筛 30 个场景：

```text
GLB + semantic PLY/SCN
  -> get_scene_boundaries: scene_boundaries/*.json
  -> extract_scene_point_clouds: point_clouds/*.h5
  -> convert_point_cloud_to_semantic_map: semantic_maps/*.h5 + *_floor.png
  -> semmap_GT_info.json
```

`extract_scene_point_clouds` 将实例 id 映射到类别；`convert_point_cloud_to_semantic_map` 按楼层把最高可见语义点投影到 2D 网格，生成 5 cm 的语义图。实测 `Allensville.h5` 的楼层 `0/map_semantic` 是 `337×355 int32`；`Collierville.h5` 有楼层 `0/1/2`。**一个场景可有多个楼层图**，所以 `SemanticMapDataset` 的样本名是 `<scene>_<floor>`，样本预算按楼层图而不是场景名分配。H5 的 `map_semantic_rgb` 只是可视化，不能当训练标签。

### 从完整图到监督样本

1. [precompute_fmm_dists.py:precompute_fmm_dists](../scripts/precompute_fmm_dists.py#L25) 建 `SemanticMapDataset(split='val')` 与 `split='train'`。该 Dataset 只加载目标 split 的场景；楼层必须通过 `MIN_OBJECTS_THRESH=4` 的过滤。`compute_fmm_dists` 把 floor/物体 mask 变成每类 geodesic 距离场，存 `.pbz2`。**这一步虽然处理 val 楼层，但 val 距离只为离线选模，不进入训练梯度。**
2. [create_poni_dataset.py](../scripts/create_poni_dataset.py#L1) 设 `OUTPUT_MAP_SIZE=24.0`、`MASKING_MODE=spath`、`MASKING_SHAPE=square`、`SEED=123`；train 目标预算 400000、val 1000，实际每图 `budget//n_maps + 1`。`--map-id`/`--map-id-range` 可按图并行生成，注意不要让两个任务写同一 map 目录。
3. [SemanticMapDataset.__getitem__](../poni/dataset.py#L124) 在 `semmap[FLOOR_ID]` 可通行处抽随机最短路径，`get_visibility_map` 沿路径开可见方形窗口（`visibility_size=3 m`）；其余位置被遮住。随机选路径上的点为 agent 中心，再平移、旋转、裁为 24 m × 24 m（480×480 像素）。
4. [transform_input_output_pairs](../poni/dataset.py#L295) 对完整语义图、已见局部图和 FMM 距离一起做几何变换，建立 frontier/有效区域 mask；其坐标变换、5 cm 网格、物体边界距离都影响最终标签。
5. [precompute_dataset_for_map](../scripts/create_poni_dataset.py#L32) 把 `in_semmap`、完整 `semmap`、`fmm_dists`（米乘 100 转 int32）、`scene_name`、`world_xyz`、`world_heading` 写为 bz2 pickle。`world_xyz`/朝向用于可视化或诊断，训练主要使用地图与距离。

注意 `precomputed_dataset` 不是直接保存最终 PF；[SemanticMapPrecomputedDataset.__getitem__](../poni/dataset.py#L575) 读样本时才将厘米还原为米，再算 `object_pfs = clamp((cutoff - fmm_dist)/cutoff,0,1)`。Gibson 默认 cutoff=10 m；物体附近值高，太远或不可达为 0。`get_masks_and_labels` 还生成 frontier、`loss_masks`、可选的未探索面积 PF；`enable_unexp_area=True` 使 PONI 成为物体 PF + area PF 双头监督。这里的“哪里值得看”是地图预测任务，**不需要在 Habitat 中走训练 episode**。

## 神经网络与 loss 的确切去向

[train.py:main](../train.py#L715) 读 [poni/default.py](../poni/default.py#L6) 默认配置和脚本命令行覆盖，创建 `Trainer`。`SemanticMapperModule` 构造 [poni/model.py:get_semantic_encoder_decoder](../poni/model.py#L274)：输入语义 one-hot `[B,17,480,480]`，U-Net encoder；object decoder 输出每通道 PF；area decoder 在 `enable_unexp_area=True` 时输出 `[B,1,480,480]`。

`batch_step` 在 [train.py:L233](../train.py#L233) 对 `object_preds[:,2:]` 与 `object_pfs[:,2:]` 做带 `loss_masks` 的 L2 损失（默认 `MODEL.object_loss_type=l2`），再加 area 头的 L2 损失。`train_dataloader` 读样本根下 `train`，`val_dataloader` 读 `val`；`Trainer.train` 使用 Adam、默认 40000 updates，定期记录 train/val loss，定期写 `best.ckpt`/`last.ckpt`。`best` 是监督损失最低，并非线上 SPL 最好。

checkpoint 的 `cfg` 和 `state_dict` 会被 [semexp/model_pf.py:Potential_Function_Semantic_Policy](../semexp/model_pf.py#L14) 重新装载，剥掉 DataParallel 名称中的 `.module`。这条接口要求 Gibson 的 17 通道语义类别顺序稳定。`MODEL.num_categories=17`、`DATASET.dset_name=gibson` 都会保存在 checkpoint 的配置中。

## 启动顺序和命令

所有命令从仓库根目录运行；训练脚本要从**独立实验目录**运行，因为它将启动时的 `$PWD` 记为 `EXPT_ROOT`。当前已有俯视图，可从 FMM 开始：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni
export PONI_ROOT=/mnt/jfzn/shensj/PONI
export PYTHONPATH="$PYTHONPATH:$PONI_ROOT"
cd "$PONI_ROOT"
ACTIVE_DATASET=gibson python scripts/precompute_fmm_dists.py
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split val
mkdir -p "$PONI_ROOT/experiments/gibson_train_new"
cd "$PONI_ROOT/experiments/gibson_train_new"
bash "$PONI_ROOT/experiment_scripts/gibson/train_poni.sh"
```

若需要从场景重建俯视图，先在仓库根目录运行 `ACTIVE_DATASET=gibson python scripts/create_semantic_maps.py`。该步骤会用 12 个 worker、显著占用 CPU/内存；已有 H5 时无须重复。训练脚本见 [train_poni.sh](../experiment_scripts/gibson/train_poni.sh)，用 `torch.distributed.launch --nproc_per_node=2`；模型 checkpoint 在刚才实验目录的 `checkpoints/`。请在开跑前确认样本盘容量，400k 个 pbz2 不是轻量数据。

## 可改处与联调点

- `DATASET.output_map_size`、`map_resolution`、`visibility_size`、masking 参数决定输入分布；改后重生成监督样本并重训模型。在线 `semexp/arguments.py` 的 `map_size_cm=2400`、`map_resolution=5` 要对齐 24 m/5 cm。
- `DATASET.object_pf_cutoff_dist` 改的是监督 PF 数值；同一配置在 checkpoint 中保存，在线 `convert_object_pf_to_distance` 读它。只改单侧会让距离语义不一致。
- 可独立训练 `pred_xy`、`pred_theta`、`pred_act` 基线，脚本在 `experiment_scripts/gibson/train_pred-*.sh`；Dataset 会生成 XY/方向/动作标签，`SemanticMapperModule` 改 decoder 与 loss，在线 `semexp/model_pf.py` 再按 `output_type` 转成地图或动作。它们是**替代长程策略**，不是 PONI 的附加模块。
- 语义分割器可以独立训练，但它输出的类别顺序必须与 [poni/constants.py](../poni/constants.py#L117) 和 PONI checkpoint 对齐。更换分割器后首先检验地图语义准确度，再看在线 SR/SPL；离线 train loss 无法代表感知质量。
