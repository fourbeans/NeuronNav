# MP3D train：56 个训练场景的离线监督流

MP3D (Matterport3D) 的 PONI 训练**仍然不跑在线导航 episode**。核心数据流与 Gibson 相同：带语义的 3D 场景 → 俯视语义图 → 每类物体 FMM 距离 → 遮挡地图/势函数监督样本 → `train.py` → MP3D checkpoint。不同的是 MP3D 原始场景格式、56/11 场景划分、21 个目标物体类别以及 loss/PF 超参数。在线评测另见 [mp3d-val.md](mp3d-val.md)。

## 两套 MP3D 场景与三套“数据”

| 数据 | 目标位置 | 用在哪一步 |
|---|---|---|
| Habitat 压缩场景 | `data/scene_datasets/mp3d/<scene>/<scene>.glb` | **在线模拟器**渲染/碰撞；由 Matterport 官方 `download_mp.py --task_data habitat` 获取。|
| 未压缩语义场景 | `data/scene_datasets/mp3d_uncompressed/...` | **离线抽俯视语义图**；按 [DATASETS.md](../DATASETS.md) 下载 `matterport_mesh`，再转换为 GLB，并准备相应语义 PLY。仅下载上行 basis 压缩 GLB 不足以复现该提图脚本。|
| MP3D ObjectNav episodes | `data/datasets/objectnav/mp3d/v1/{train,val,val_parts}` | Habitat 的起点/目标任务数据；`train` 文件里可有每场景大量 episode，但本仓库 [train_poni.sh](../experiment_scripts/mp3d/train_poni.sh#L1) **不读取它们**。`val_parts` 供在线验证。|
| 俯视语义图 | `data/semantic_maps/mp3d/semantic_maps/*.h5` | 训练监督母图；当前服务器没有。|
| FMM 距离 | `data/semantic_maps/mp3d/fmm_dists_123/<scene>_<floor>.pbz2` | 为所有类别计算可达距离；当前没有。|
| 预计算样本 | `data/semantic_maps/mp3d/precomputed_dataset_24.0_123_spath_square/{train,val}/...` | `train.py` 的真正输入；当前没有。|
| 作者训练权重 | `pretrained_models/mp3d_models/poni_seed_{123,234,345}.ckpt` | 当前已有，可在 MP3D 场景补齐后直接评测，无需从头训练。|

本机 `/mnt/jfzn/shensj/data/datasets/objectnav/mp3d/v1/train/content` 当前有 56 个场景 JSON，`val/content` 有 11 个。第一份 train scene JSON 有 50000 条 Habitat ObjectNav episode；**它们不等于 PONI 的 400000 条离线地图样本**。离线训练的场景划分由 [SPLIT_SCENES['mp3d']](../poni/constants.py#L37) 控制：train 56 个、val 11 个；楼层数和最后有效地图数由 H5 内容及 `MIN_OBJECTS_THRESH=4` 决定。

## 原始 MP3D 场景到俯视图

[scripts/create_semantic_maps.py](../scripts/create_semantic_maps.py#L1) 读取环境变量 `ACTIVE_DATASET=mp3d` 后设 `SCENES_ROOT=data/scene_datasets/mp3d_uncompressed`、楼层边界采样步长 `SAMPLING_RESOLUTION=0.20 m`、8 workers；主程序搜有 `_semantic.ply` 的 `.glb`，按 [SPLIT_SCENES](../poni/constants.py#L5) 限定 56+11 个场景。`get_scene_boundaries` 取场景边界/楼层范围，`extract_scene_point_clouds` 取带实例与类别的语义点，[convert_point_cloud_to_semantic_map](../scripts/create_semantic_maps.py#L403) 将每楼层点投到 5 cm 俯视栅格，输出：

```text
data/semantic_maps/mp3d/
  scene_boundaries/<scene>.json
  point_clouds/<scene>.h5
  semantic_maps/<scene>.h5
  semantic_maps/<scene>_<floor>.png
  semantic_maps/semmap_GT_info.json
```

H5 的 `floor/map_semantic` 是整数类别格网；`floor/mask` 是有效区域；`map_instance` 保留实例图；`semmap_GT_info.json` 记录分辨率、世界坐标偏移和楼层高度。[SemanticMapDataset.__init__](../poni/dataset.py#L54) 会把每楼层 `map_semantic` 转 one-hot，并在选定 split 中只保留类别数足够的楼层。这里的 5 cm、坐标系、楼层高度会影响后续监督/在线模块的几何一致性。

MP3D [OBJECT_CATEGORIES](../poni/constants.py#L134) 列表有 `floor`、`wall` 加 21 种语义物体，故 `MODEL.num_categories=23`。注意 MP3D Habitat `objectgoal` 传感器有自己的类别 id；在线 `RedNet` 还做 THDA/MP3D 到 ObjectNav 的映射。不能直接把一种 id 当另一种 id。

## 俯视图到 24 m 样本

1. `ACTIVE_DATASET=mp3d python scripts/precompute_fmm_dists.py`：按 [precompute_fmm_dists](../scripts/precompute_fmm_dists.py#L25) 先 val 后 train 创建 `SemanticMapDataset`，其 [compute_fmm_dists](../poni/dataset.py#L197) 对 floor/每物体类算到目标的 FMM geodesic 距离。目标不在图上则距离为 `inf`。每个有效楼层写一份 `.pbz2`。
2. `create_poni_dataset.py --split train/val`：配置 `OUTPUT_MAP_SIZE=24.0`、`MASKING_MODE=spath`、`MASKING_SHAPE=square`、`visibility_size=3.0`。train 样本预算 400000、val 1000；每楼层 `budget//n_maps+1`，故最终文件数可能略大于预算。[SemanticMapDataset.get_random_shortest_path](../poni/dataset.py#L171) 在 floor 可行走区域抽路径。
3. [create_spath_based_input_output_pairs](../poni/dataset.py#L261) 沿路径揭示局部语义图，随机选其中位置为 agent，随机旋转，平移/裁到 480×480；形成未知区域为 0 的 `in_semmap`、完整 `semmap`、同坐标系 `fmm_dists`。保存脚本 [precompute_dataset_for_map](../scripts/create_poni_dataset.py#L32) 将距离乘 100 以 int32 存盘，`world_xyz`/`world_heading` 可追溯采样位姿。
4. 训练 DataLoader 使用 [SemanticMapPrecomputedDataset.__getitem__](../poni/dataset.py#L575) 还原距离、生成势函数及 `loss_masks`。MP3D 的 `object_pf_cutoff_dist=5.0m`，不同于 Gibson 默认 10 m；未探索面积头启用 `normalize_area_by_constant=True`、`max_unexp_area=40.0`，使不同大小场景中的面积目标有统一分母。此处改参数会改变监督标签，即使原始 pbz2 不变，也会要求重训 checkpoint。

`in_semmap` 的形状为 `[23,480,480]`；`object_pfs` 也是每类一张 480×480 图，前两类 floor/wall 不参与物体 PF loss。`area_pfs` 是 `[1,480,480]`。frontier 是已探索地面与未知区域的边界，监督 loss 对 frontier/可评区域加权；它不是物体的真实位置，监督目标利用完整 GT 地图给“从当前未知处看向物体”的线索。

## 模型、损失和 checkpoint

[experiment_scripts/mp3d/train_poni.sh](../experiment_scripts/mp3d/train_poni.sh#L1) 以 `torch.distributed.launch --nproc_per_node=2` 启动 [train.py:main](../train.py#L715)，覆盖 `DATASET.root`、`dset_name=mp3d`、`enable_unexp_area=True`、`object_pf_cutoff_dist=5.0`、面积归一参数、`MODEL.num_categories=23`、`OPTIM.batch_size=16`。每进程 batch 16、2 进程的有效 batch 约 32（按 [Trainer.train](../train.py#L477) 的计算）。默认 `OPTIM.num_total_updates=40000`、Adam lr `1e-3`，并按 milestones 衰减。

[SemanticMapperModule](../train.py#L27) 包含 23 通道 U-Net encoder、物体 PF decoder 和 1 通道 area decoder。`batch_step` 对通道 `2:` 的物体 PF 用 loss mask 算 L2，对 area PF 用 L2，两项相加；[Trainer.train](../train.py#L477) 从 `train` 子目录取样，周期性用 `val` 子目录算 val loss/选 `best.ckpt`，记录 `train.log`、TensorBoard，并保存 `best.ckpt`、`last.ckpt`。这只是离线 val loss；MP3D ObjectNav 的 Success/SPL 还要单独在线测。`last.ckpt` 若已存在，会自动 resume，同一路径重跑可能不是全新训练。

## 可执行步骤与当前阻断

下列是**所需顺序**，当前服务器缺 `mp3d_uncompressed`，不能直接从第二行开始：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni
export PONI_ROOT=/mnt/jfzn/shensj/PONI
export PYTHONPATH="$PYTHONPATH:$PONI_ROOT"
cd "$PONI_ROOT"
# 先按 DATASETS.md 放好 mp3d_uncompressed 场景、语义 PLY 和 GLB
ACTIVE_DATASET=mp3d python scripts/create_semantic_maps.py
ACTIVE_DATASET=mp3d python scripts/precompute_fmm_dists.py
ACTIVE_DATASET=mp3d python scripts/create_poni_dataset.py --split train
ACTIVE_DATASET=mp3d python scripts/create_poni_dataset.py --split val
mkdir -p "$PONI_ROOT/experiments/mp3d_train_new"
cd "$PONI_ROOT/experiments/mp3d_train_new"
bash "$PONI_ROOT/experiment_scripts/mp3d/train_poni.sh"
```

授权下载场景由 Matterport 官方脚本完成，见 [DATASETS.md](../DATASETS.md)；`matterport_mesh` 转 GLB 的具体批处理并没有在仓库提供一键脚本，不应把 `download_mp.py --type matterport_mesh` 误认作生成了脚本可读的全部 GLB/PLY。使用现有 `/mnt/jfzn/shensj/data/datasets` 共享目录可以通过 symlink 映射到 `data/scene_datasets/mp3d` 和 `mp3d_uncompressed`；只需保证源码中的相对路径落到预期文件，不必重复下载模型和 episodes。

## 单独训练/联合调优边界

MP3D 的 PONI、predict-XY、predict-theta、predict-action 是四种**可替代的长程输出头**，各自脚本在 `experiment_scripts/mp3d/train_*.sh`，共享样本根、encoder 框架，但标签和 loss 配置不同（位置回归、方向/动作分类）。分割器 RedNet 在 [hlab/utils/rednet_semantic_prediction.py](../hlab/utils/rednet_semantic_prediction.py#L401) 独立加载，可单独替换；PONI 只收在线语义地图。需要联调的是 RedNet 类别顺序 ↔ `GlobalAgent` 语义通道 ↔ PONI 的 23 通道输入，以及地图 5 cm/24 m ↔ 训练样本坐标系 ↔ 局部规划器的目标坐标。调 area 权重或 frontier mask 等在线参数只需重评测；调离线 PF cutoff/类别/监督图需重训。
