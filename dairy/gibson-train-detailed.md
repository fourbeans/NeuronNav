# Gibson PONI train：从已知语义地图到 checkpoint 的完整调用链

这份文档重新只讲一件事：**Gibson 上训练 PONI 势函数网络时，到底运行什么命令、训练什么、哪些东西训练前已经知道、离线和在线有什么区别、代码按什么顺序调用**。

先把最重要的结论放前面：

- Gibson 的 PONI 训练不是在 Habitat 里跑 ObjectNav episode 做强化学习。
- `experiment_scripts/gibson/train_poni.sh` 启动的是 [train.py](../train.py)，它只读已经预生成的 `.pbz2` 离线监督样本。
- 训练的模型是 [poni/model.py](../poni/model.py) 里的 U-Net：输入“当前已见的 top-down 语义地图”，输出“每个物体类别的 potential function”和“未探索面积 potential function”。
- 训练前已知的是完整 Gibson 语义场景、由场景生成的完整俯视语义图、每个位置到每类物体的 FMM 地面距离、随机遮挡/裁剪出来的监督样本。
- ObjectNav episodes、Habitat 在线环境、Mask R-CNN 语义分割器都不是 PONI 势函数网络的训练输入；它们在后面的在线评测阶段才进入。

## 1. Gibson train 会运行什么命令

README 里的训练方式是先进入一个实验目录，再复制并运行脚本。对 Gibson PONI 来说，实际入口是 [experiment_scripts/gibson/train_poni.sh](../experiment_scripts/gibson/train_poni.sh)：

```bash
mkdir -p /mnt/jfzn/shensj/PONI/experiments/gibson_poni_train_001
cd /mnt/jfzn/shensj/PONI/experiments/gibson_poni_train_001

export PONI_ROOT=/mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni

bash "$PONI_ROOT/experiment_scripts/gibson/train_poni.sh"
```

`train_poni.sh` 里面真正执行的是：

```bash
EXPT_ROOT=$PWD
conda activate poni
cd $PONI_ROOT

python -W ignore -u -m torch.distributed.launch \
  --use_env \
  --nproc_per_node=2 \
  train.py \
    LOGGING.ckpt_dir "$EXPT_ROOT/checkpoints" \
    LOGGING.tb_dir "$EXPT_ROOT/tb" \
    LOGGING.log_dir "$EXPT_ROOT" \
    DATASET.root $PONI_ROOT/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square \
    DATASET.enable_unexp_area True
```

几个细节必须注意：

- `EXPT_ROOT=$PWD` 发生在脚本刚启动时，所以你在哪个实验目录里运行脚本，日志和 checkpoint 就写到哪里。
- 脚本随后 `cd $PONI_ROOT`，所以 Python 入口总是仓库根目录下的 [train.py](../train.py)。
- `--nproc_per_node=2` 表示默认用 2 个 GPU / 2 个 DDP 进程训练。
- `DATASET.root` 指向的不是 `.h5` 语义地图，而是已经做好的离线 `.pbz2` 监督样本目录。
- `DATASET.enable_unexp_area True` 会打开第二个 head，也就是未探索面积势函数 head。

输出会在实验目录下：

```text
experiments/gibson_poni_train_001/
  train.log
  tb/train/
  checkpoints/
    last.ckpt
    best.ckpt
```

如果 `checkpoints/last.ckpt` 已经存在，[Trainer.train](../train.py) 会自动 resume。这对续训方便，但如果你想开一个全新实验，必须换一个空实验目录。

## 2. 训练主要训练什么

PONI 训练的是一个**地图到地图**的监督学习模型，不是端到端导航策略。

输入：

```text
in_semmap: [17, 480, 480]
```

含义是 24m x 24m 的局部 top-down 语义图，分辨率 5cm，所以 `24 / 0.05 = 480`。17 个通道来自 [poni/constants.py](../poni/constants.py) 的 Gibson 类别：

```text
0 floor
1 wall
2 chair
3 couch
4 potted plant
5 bed
6 toilet
7 tv
8 dining-table
9 oven
10 sink
11 refrigerator
12 book
13 clock
14 vase
15 cup
16 bottle
```

这里的 `in_semmap` 是“局部已观察地图”：完整语义图被随机路径可见区域遮掉，只保留 agent 沿随机 shortest path 走过时附近可见的区域，未知区域全 0。

监督目标 1：

```text
object_pfs: [17, 480, 480]
```

它不是语义分割标签，而是每个语义类别的 potential function。对每个像素、每个类别，先知道该像素沿可通行区域到最近该类物体的 FMM 距离 `d`，再转成：

```text
PF = clamp((cutoff - d) / cutoff, 0, 1)
```

Gibson 默认 `cutoff = 10m`。越靠近目标物体，PF 越高；超过 10m 或不可达，接近 0。训练 loss 会忽略 `floor/wall` 两个通道，只训练通道 `2:` 的物体 PF。

监督目标 2：

```text
area_pfs: [1, 480, 480]
```

这是 `DATASET.enable_unexp_area True` 打开的未探索面积势函数。它不是某个物体类别，而是估计某个 frontier 后面还连着多少未探索可通行区域。在线导航时，PONI 会把目标物体 PF 和 area PF 结合，用来决定“该去哪个 frontier 看一眼”。

模型：

- [SemanticMapperModule](../train.py) 初始化网络、loss、optimizer。
- [get_semantic_encoder_decoder](../poni/model.py) 构造 U-Net encoder、object decoder、area decoder。
- encoder 输入 17 通道语义地图。
- object decoder 输出 17 通道物体 PF。
- area decoder 输出 1 通道 area PF。
- 激活函数默认 sigmoid，因为标签是 0 到 1 的势函数。

所以训练问题可以一句话概括：

```text
给定“只看过一部分的语义地图”，预测“如果目标是某类物体，哪些 frontier / 区域最值得去看”。
```

## 3. 什么东西是已知的，分别从哪里来

PONI 的核心想法是 interaction-free learning：训练势函数时不需要在环境里反复试错。它把完整场景语义当成已知的离线监督来源。

### 3.1 Gibson 3D 场景和语义是已知的

来源是 Gibson 场景和 3DSceneGraph 语义标注，经过 Habitat 格式转换后放在项目的 scene dataset 目录。它们提供：

- 几何：`.glb`，Habitat 可以加载的 3D 场景。
- 语义：每个点/面属于什么语义类别。
- 坐标：场景坐标系、楼层高度、世界坐标。

这些东西不是训练网络直接读取的张量，而是生成俯视语义图的原料。

### 3.2 俯视语义图是已知的

当前本机已有：

```text
/mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/semantic_maps/
```

这个目录里有 30 个 Gibson 场景的 `.h5`，对应 [SPLIT_SCENES["gibson"]](../poni/constants.py) 里的 25 个 train scene 和 5 个 val scene。

如果从原始场景生成，命令是：

```bash
cd /mnt/jfzn/shensj/PONI
ACTIVE_DATASET=gibson python scripts/create_semantic_maps.py
```

这一步会把 3D 场景投影成每个楼层的 2D 语义图，输出大致是：

```text
data/semantic_maps/gibson/
  scene_boundaries/*.json
  point_clouds/*.h5
  semantic_maps/
    <scene>.h5
    <scene>_<floor>.png
    semmap_GT_info.json
```

`.h5` 里的核心数据是每层的 `map_semantic`，每个格子是一个类别 id。训练代码不会直接用 RGB 图，RGB 图只是给人看的可视化。

### 3.3 FMM 距离是已知的

PONI 的物体 PF 标签依赖“从每个位置沿可通行区域到最近物体的距离”。这个距离不是欧氏直线距离，而是 Fast Marching Method 在 floor/wall/obstacle 约束下算出的地面路径距离。

生成命令：

```bash
cd /mnt/jfzn/shensj/PONI
ACTIVE_DATASET=gibson python scripts/precompute_fmm_dists.py
```

脚本 [scripts/precompute_fmm_dists.py](../scripts/precompute_fmm_dists.py) 会：

1. 调 `get_cfg()` 读取默认配置。
2. 设置 `cfg.DATASET.dset_name = "gibson"`。
3. 设置 `cfg.DATASET.root = data/semantic_maps/gibson/semantic_maps`。
4. 设置 `cfg.DATASET.fmm_dists_saved_root = ""`，强制现场计算 FMM。
5. 对 `split in ["val", "train"]` 分别构造 [SemanticMapDataset](../poni/dataset.py)。
6. 保存每张 `<scene>_<floor>` 地图的 FMM 距离到：

```text
data/semantic_maps/gibson/fmm_dists_123/<scene>_<floor>.pbz2
```

这里的 `123` 是生成数据用的固定 seed 标识，不是 checkpoint seed 123/234/345 那种实验种子概念。

当前本机检查结果：这个目录下现在是 0 个 `.pbz2`，所以从头训练前要先跑这一步。

### 3.4 训练 `.pbz2` 样本是已知的

真正喂给 [train.py](../train.py) 的不是 `.h5`，而是 `create_poni_dataset.py` 预生成的样本：

```bash
cd /mnt/jfzn/shensj/PONI
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split val
```

输出目录固定为：

```text
data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/
  train/<scene>_<floor>/sample_00000.pbz2
  val/<scene>_<floor>/sample_00000.pbz2
```

[scripts/create_poni_dataset.py](../scripts/create_poni_dataset.py) 的固定参数是：

```text
OUTPUT_MAP_SIZE = 24.0
MASKING_MODE = spath
MASKING_SHAPE = square
SEED = 123
NUM_SAMPLES["train"] = 400000
NUM_SAMPLES["val"] = 1000
```

每条 sample 存的是：

```text
in_semmap      # 被可见区域遮罩后的输入语义地图
semmap         # 完整局部语义地图
fmm_dists      # 每类物体的 FMM 距离，保存时乘 100 转 int32
scene_name
world_xyz
world_heading
```

注意：sample 里保存的是 `fmm_dists`，不是最终 `object_pfs`。最终 PF 在训练读取 sample 时重新按 cutoff 算出来。

当前本机检查结果：`precomputed_dataset_24.0_123_spath_square` 下现在是 0 个 `.pbz2`，所以 `train_poni.sh` 现在直接开会没有训练数据。

## 4. 离线训练和在线评测到底有什么区别

这里是最容易混的一层。

### 离线训练

离线训练只发生在这些文件之间：

```text
Gibson 3D 场景 + 语义
  -> scripts/create_semantic_maps.py
  -> semantic_maps/*.h5
  -> scripts/precompute_fmm_dists.py
  -> fmm_dists_123/*.pbz2
  -> scripts/create_poni_dataset.py
  -> precomputed_dataset_24.0_123_spath_square/train,val/*.pbz2
  -> train.py
  -> best.ckpt / last.ckpt
```

它不启动 Habitat 环境，不读取 ObjectNav episode，不执行 `env.step()`，也不需要 Mask R-CNN。完整语义地图被当成“上帝视角已知标签”，训练网络学习从部分地图猜测目标位置价值。

### 在线评测

在线评测才是导航：

```text
ObjectNav episode
  -> Habitat reset 到某个 scene/start/goal
  -> RGB/depth 观测
  -> Mask R-CNN 预测语义
  -> semantic mapping 累积局部地图
  -> PONI checkpoint 预测 PF
  -> long-term goal / frontier
  -> FMM local planner
  -> Habitat env.step()
  -> success / SPL / DTS 等指标
```

所以：

- Gibson scene dataset 是环境和离线地图的空间基础。
- Gibson ObjectNav episodes 是在线任务定义：起点、目标类别、目标实例、最大步数等。
- PONI checkpoint 是离线训练得到的“去哪看”的模块。
- Mask R-CNN 是在线从 RGB 里估计语义的感知模块。

从头训练 PONI checkpoint 时不需要 episodes；复现导航指标时必须有 episodes。

## 5. 训练前需要准备什么

训练 `train_poni.sh` 前最少要满足：

```text
1. Python/torch 环境能运行 PONI
2. /mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/semantic_maps/*.h5
3. /mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/semantic_maps/semmap_GT_info.json
4. /mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/fmm_dists_123/*.pbz2
5. /mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/train/**/*.pbz2
6. /mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/val/**/*.pbz2
```

当前机器上第 2、3 项已有；第 4、5、6 项未生成。

推荐准备命令：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni
export PONI_ROOT=/mnt/jfzn/shensj/PONI
export PYTHONPATH="$PONI_ROOT:$PYTHONPATH"
cd "$PONI_ROOT"

# 1. 如果 semantic_maps 已存在，跳过这一行。
# ACTIVE_DATASET=gibson python scripts/create_semantic_maps.py

# 2. 生成每个楼层、每个类别的 FMM 距离。
ACTIVE_DATASET=gibson python scripts/precompute_fmm_dists.py

# 3. 生成 train/val 离线监督样本。
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split val
```

`create_poni_dataset.py` 支持并行拆 map：

```bash
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train --map-id 0
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train --map-id-range 0 9
```

但同一个 `map-id` 不要被两个进程同时写，否则会覆盖同一个 `<scene>_<floor>/sample_*.pbz2`。

## 6. 数据生成调用顺序：从 `.h5` 到一个训练样本

这一节按函数调用解释“一个 sample 是怎么来的”。

### 6.1 `precompute_fmm_dists.py`

入口：

```text
scripts/precompute_fmm_dists.py
  -> precompute_fmm_dists()
```

调用链：

```text
precompute_fmm_dists()
  -> get_cfg()
  -> SemanticMapDataset(cfg.DATASET, split="val")
  -> SemanticMapDataset.compute_fmm_dists()
  -> save fmm_dists_123/<name>.pbz2
  -> SemanticMapDataset(cfg.DATASET, split="train")
  -> SemanticMapDataset.compute_fmm_dists()
  -> save fmm_dists_123/<name>.pbz2
```

[SemanticMapDataset.__init__](../poni/dataset.py) 做的事情：

1. `glob(cfg.root/*.h5)` 找到 30 个 Gibson 场景语义地图。
2. 读取 `semmap_GT_info.json`。
3. 按 [SPLIT_SCENES](../poni/constants.py) 过滤 train 或 val scene。
4. 打开每个 `.h5`，遍历楼层 id。
5. 读每层 `map_semantic`。
6. 过滤掉语义类别太少的楼层：`len(unique(map_semantic)) >= MIN_OBJECTS_THRESH + 2`。
7. `convert_maps_to_oh()` 把单通道类别 id 图转成 `[17,H,W]` one-hot。
8. 如果 `fmm_dists_saved_root == ""`，调用 `compute_fmm_dists()` 现场算距离。

[compute_fmm_dists](../poni/dataset.py) 的核心逻辑：

```text
for 每个 <scene>_<floor>:
  navmap = semmap[FLOOR_ID]
  for 每个类别 catmap:
    if 该类别不存在:
      fmm_dist = inf
    else:
      把目标物体周围 1m 区域扩张成可到达目标区域
      cat_navmap[navmap > 0] = 1
      planner = FMMPlanner(cat_navmap)
      planner.set_multi_goal(catmap)
      fmm_dist = planner.fmm_dist
  stack 成 [17,H,W]
```

这里的 FMM 距离是 PONI 物体 PF 的根标签。为什么要用 FMM 而不是欧氏距离？因为导航不能穿墙，欧氏距离会把隔墙很近的物体当成“很容易到”，FMM 会沿可通行区域算 geodesic 距离。

### 6.2 `create_poni_dataset.py`

入口：

```text
scripts/create_poni_dataset.py
  -> precompute_dataset(args)
```

调用链：

```text
precompute_dataset(args)
  -> get_cfg()
  -> cfg.DATASET.root = data/semantic_maps/gibson/semantic_maps
  -> cfg.DATASET.fmm_dists_saved_root = data/semantic_maps/gibson/fmm_dists_123
  -> SemanticMapDataset(cfg.DATASET, split=args.split)
  -> 根据 NUM_SAMPLES / n_maps 算每张地图采多少条
  -> multiprocessing pool
  -> precompute_dataset_for_map(kwargs)
```

[precompute_dataset_for_map](../scripts/create_poni_dataset.py) 对每个 `<scene>_<floor>` 做：

```text
dataset = SemanticMapDataset(cfg.DATASET, split=split, scf_name=name, seed=123)
for i in range(n_samples_per_map):
  input, label = dataset.get_item_by_name(name)
  保存 sample_i.pbz2
```

继续进入 [SemanticMapDataset.__getitem__](../poni/dataset.py)：

```text
__getitem__(idx)
  -> name = self.names[idx]
  -> semmap = self.maps[name]
  -> fmm_dists = self.fmm_dists[name]
  -> nav_space = semmap[FLOOR_ID]
  -> nav_locs = self.nav_locs[name]
  -> get_random_shortest_path(nav_space, nav_locs)
  -> create_spath_based_input_output_pairs(...)
```

`get_random_shortest_path()` 做的是随机造一个“被动观察轨迹”：

1. 在可通行 floor 上随机选一个 start。
2. 用 FMM 找到从这个 start 可达的区域。
3. 在可达区域随机选一个 end。
4. `plan_path()` 沿 FMM shortest path 从 start 走到 end，得到 `spath`。

注意：这里的 `spath` 不是 ObjectNav episode。它没有目标物体类别，没有成功/失败，只是为了模拟“agent 走过一段路，所以看过地图中的一部分区域”。

`create_spath_based_input_output_pairs()` 做的是把完整地图变成一个监督样本：

1. `out_semmap = 完整 one-hot 语义图`。
2. `out_fmm_dists = 预计算 FMM 距离 * 0.05m`，把格子距离转成米。
3. `vis_map = get_visibility_map(in_semmap, spath)`，沿 path 上每个点打开一个 `visibility_size=3m` 的方形可见窗口。
4. `in_semmap = out_semmap * vis_map`，未知区域被清零。
5. 从 `spath` 上随机选一个点作为 agent 中心。
6. 随机旋转角 `rot in [-pi, pi]`，让模型不要记住固定朝向。
7. `transform_input_output_pairs()` 对 `in_semmap/out_semmap/fmm_dists` 做同样的平移、旋转、裁剪。
8. 裁出 `24m x 24m`，即 `[17,480,480]`。
9. 计算 `world_xyz/world_heading/scene_name`。
10. 返回输入地图和 label 字典。

`transform_input_output_pairs()` 里最关键的三步：

```text
先裁一张足够大的图：2 * output_map_size = 48m
旋转到随机朝向
再裁最终 24m x 24m
```

为什么要先裁 48m 再旋转再裁 24m？因为如果直接对 24m 图旋转，角落会丢信息。先裁大图再旋转可以保证中心周围 24m 内容更完整。

最后 sample 写盘时：

```text
in_semmap = bool [17,480,480]
semmap = bool [17,480,480]
fmm_dists = int32 [17,480,480]  # 米 * 100
```

乘 100 转 int32 是为了省空间。训练读取时会除以 100 还原为米。

## 7. 训练调用顺序：从 `train_poni.sh` 到一次参数更新

入口：

```text
experiment_scripts/gibson/train_poni.sh
  -> python -m torch.distributed.launch ... train.py ...
```

Python 入口：

```text
train.py
  -> main(args.exp_config, args.opts)
  -> cfg = get_cfg(exp_config, opts)
  -> trainer = Trainer(cfg)
  -> trainer.train()
```

### 7.1 配置合并

[poni/default.py](../poni/default.py) 给默认配置：

```text
SEED = 123
MODEL.num_categories = 17
MODEL.object_loss_type = l2
MODEL.area_loss_type = l2
MODEL.object_activation = sigmoid
MODEL.area_activation = sigmoid
MODEL.nsf = 32
OPTIM.lr = 1e-3
OPTIM.num_total_updates = 40000
OPTIM.batch_size = 20
OPTIM.num_workers = 8
DATASET.output_map_size = 24.0
DATASET.object_pf_cutoff_dist = 10.0
DATASET.potential_function_masking = True
```

`train_poni.sh` 再覆盖：

```text
LOGGING.ckpt_dir = $EXPT_ROOT/checkpoints
LOGGING.tb_dir = $EXPT_ROOT/tb
LOGGING.log_dir = $EXPT_ROOT
DATASET.root = $PONI_ROOT/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square
DATASET.enable_unexp_area = True
```

### 7.2 `Trainer.__init__`

[Trainer.__init__](../train.py) 做：

1. 从 `LOCAL_RANK/RANK/WORLD_SIZE` 判断是不是 DDP。
2. 如果是 DDP，`ddp_setup()` 建 NCCL process group。
3. 设置 seed：`cfg.SEED + world_rank`。
4. device 设成 `cuda:<local_rank>`。
5. 构造 [SemanticMapperModule](../train.py)。
6. 多卡时把 encoder/object_decoder/area_decoder 包成 DistributedDataParallel。

### 7.3 `SemanticMapperModule.__init__`

[SemanticMapperModule](../train.py) 做：

1. `get_loss_fn("l2")` 得到 MSE loss。
2. 检查是否启用 directions/locations/actions。PONI 正常训练都不启用这些替代 baseline。
3. 看到 `DATASET.enable_unexp_area=True`，把 `MODEL.enable_area_head=True`。
4. 调 [get_semantic_encoder_decoder](../poni/model.py) 建模型。
5. 建 Adam optimizer，学习率 `1e-3`。
6. 建 MultiStepLR，milestone 默认 `[2]`。
7. object/area 输出都用 sigmoid。

模型结构：

```text
input [B,17,480,480]
  -> UNetEncoder
  -> object_decoder: [B,17,480,480]
  -> area_decoder:   [B,1,480,480]
```

### 7.4 `Trainer.train()` 建 dataloader

[Trainer.train](../train.py) 进入后先建日志，然后：

```text
train_loader = self.model.train_dataloader(self.is_distributed)
val_loader = self.model.val_dataloader()
```

`train_dataloader()` 里实际创建：

```text
SMPrecompDataset(self.cfg.DATASET, split="train")
DataLoader(... batch_size=20, num_workers=8, collate_fn=collate_fn)
```

`SMPrecompDataset` 是 [SemanticMapPrecomputedDataset](../poni/dataset.py)，它只做一件事：

```text
glob(DATASET.root/train/**/*.pbz2)
glob(DATASET.root/val/**/*.pbz2)
```

所以如果 `precomputed_dataset_24.0_123_spath_square/train` 里没有 `.pbz2`，训练集长度就是 0，训练就没法正常进行。

DDP 下 train loader 会用 `DistributedSampler`，所以 2 个 GPU 会分摊同一个离线样本集。

### 7.5 `SemanticMapPrecomputedDataset.__getitem__`

每次 DataLoader 取一条 sample：

```text
__getitem__(idx)
  -> 读 sample_xxxxx.pbz2
  -> fmm_dists int32 / 100.0，还原成米
  -> in_semmap = torch tensor
  -> semmap = torch tensor
  -> object_pfs = compute_object_pfs(fmm_dists)
  -> get_masks_and_labels(in_semmap, semmap, fmm_dists)
  -> object_pfs *= masks
  -> object_pfs *= 1000 转 int，省 dataloader 内存
  -> area_pfs *= 1000 转 int
  -> return {"semmap": in_semmap}, labels
```

这里有两个 mask：

- `loss_masks`：哪里参与 loss。
- `masks`：哪里保留 potential function 的监督值。

`get_masks_and_labels()` 会从输入地图里算：

1. `free_map`：已知 free floor。
2. `exp_map`：已经观测到任意语义的区域。
3. `unk_map`：未知区域。
4. `frontiers = get_frontiers_np(unk_map, free_map)`：free 和 unknown 的边界。
5. `frontiers_mask`：膨胀后的 frontier mask。

默认配置里：

```text
potential_function_frontier_scaling = 1.0
potential_function_non_visible_scaling = 0.0
potential_function_non_frontier_scaling = 0.0
```

因此物体 PF loss 主要落在 frontier 区域。为什么？因为 PONI 在线导航真正要做的不是重建整张地图，而是在“已知/未知边界”上选择值得去探索的方向。

如果 `enable_unexp_area=True`，`get_masks_and_labels()` 还会：

1. 找 frontier contours。
2. 找未知 free space 的 connected components。
3. 计算每个 frontier 后面连着多少未知可通行面积。
4. 得到 `[1,480,480]` 的 `area_pfs`。

### 7.6 一次 batch 更新

[Trainer.train](../train.py) 的训练循环中，每个 batch：

```text
for batch_id, (inputs, labels) in enumerate(train_loader):
  inputs -> cuda
  labels -> cuda
  batch = self.model.undo_memory_opts(batch)
  outputs = self.model.batch_step(batch)
  loss = outputs["loss"]
  self.model.update(loss)
```

`undo_memory_opts()` 做：

```text
inputs["semmap"] = float
labels["object_pfs"] = int / 1000.0
labels["area_pfs"] = int / 1000.0
```

`batch_step()` 做真正前向和 loss：

```text
input_maps = inputs["semmap"]
object_preds, area_preds = self(input_maps)
```

然后 PONI map 输出的 loss：

```text
y_hat = object_preds[:, 2:]
y = labels["object_pfs"][:, 2:]
mask = labels["loss_masks"][:, 2:]
loss = MSE(y_hat, y)
loss = (loss * mask).sum(H,W) / mask.sum(H,W)
loss = loss.mean()
```

为什么 `[:, 2:]`？因为通道 0 是 floor，通道 1 是 wall，它们不是 ObjectNav 目标物体；PONI 只监督 chair/couch/.../bottle 这些目标物体类别。

area loss：

```text
area_pf_loss = MSE(area_preds, labels["area_pfs"]).mean()
loss = object_pf_loss + area_pf_loss
```

最后：

```text
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 7.7 val 和 checkpoint

每 `LOGGING.eval_interval=1000` 次 update：

```text
for val batch:
  no_grad()
  batch_step()
  统计 val loss
```

每 `LOGGING.ckpt_interval=1000` 次 update，rank 0 保存：

```text
checkpoints/last.ckpt
checkpoints/best.ckpt
```

checkpoint 里包括：

```text
state_dict
optimizer_state_dict
lr_scheduler_state_dict
epoch
running_steps
cfg
best_val_loss
num_updates
```

`best.ckpt` 的“best”是离线 val loss 最低，不是在线 ObjectNav SPL 最高。

## 8. 25 train scenes、30 semantic maps、400000 samples 分别是什么意思

### 25 train scenes

[poni/constants.py](../poni/constants.py) 里 Gibson train split 写死 25 个 scene：

```text
Allensville, Beechwood, Marstons, Benevolence, Merom,
Coffeen, Mifflinburg, Newfields, Onaga, Cosmos,
Pinesdale, Pomaria, Forkland, Ranchester, Hanson,
Shelbyville, Hiteman, Stockman, Klickitat, Tolstoy,
Lakeville, Wainscott, Leonardo, Lindenwood, Woodbine
```

训练样本只从这些 scene 的 semantic map 生成。

### 5 val scenes

val split 是：

```text
Darden, Collierville, Markleeville, Wiconisco, Corozal
```

它们只用于离线 val loss，决定 `best.ckpt`，不参与训练梯度。

### 30 semantic maps

“30”指 25 train scene + 5 val scene 的场景数量。每个 scene 可以有多个 floor，所以真实可采样地图单元是 `<scene>_<floor>`，例如 `Collierville_0/1/2`。

### 400000 samples

`create_poni_dataset.py` 对 train split 设置总预算 `400000`。它不是 400000 个 scene，也不是 400000 个 episode，而是随机从 train 楼层图上裁剪/遮挡得到的 400000 条监督样本。

每条样本都像一次“假想局部观察”：

```text
完整已知地图 + 随机短路径可见区域 + 随机中心/朝向
  -> 输入局部已见地图
  -> 标签为该局部图上的目标物体 PF 和 area PF
```

## 9. 和 Gibson ObjectNav train episodes 的关系

PONI 工程里有 Gibson ObjectNav train/val episodes，但它们不是 `train_poni.sh` 的输入。

episodes 的含义是在线导航任务：

```text
scene_id
start_position
start_rotation
目标类别 / 目标实例
episode_id
```

一条 episode 就是“一次导航任务”。它用于：

- 在线评测 PONI checkpoint。
- 评估 success/SPL/DTS。
- 如果训练 SemExp 那种 RL policy，会用于环境交互。

但 PONI 势函数网络本身的训练是离线监督学习，不用 episode。这也是论文里说训练计算量小的原因之一。

所以从工程角度看：

```text
训练 PONI checkpoint:
  需要 semantic_maps + fmm_dists + precomputed_dataset
  不需要 objectnav episodes

评测 PONI checkpoint:
  需要 objectnav episodes + Habitat scene + segmentation model + PONI checkpoint
  不需要 precomputed_dataset
```

## 10. 现在这台机器下一步如果要真的训练

当前本机已有：

```text
/mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/semantic_maps/
```

当前缺：

```text
/mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/fmm_dists_123/*.pbz2
/mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/train/**/*.pbz2
/mnt/jfzn/shensj/PONI/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square/val/**/*.pbz2
```

建议顺序：

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/jfzn/miniconda3/envs/poni
export PONI_ROOT=/mnt/jfzn/shensj/PONI
export PYTHONPATH="$PONI_ROOT:$PYTHONPATH"
cd "$PONI_ROOT"

ACTIVE_DATASET=gibson python scripts/precompute_fmm_dists.py
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split train
ACTIVE_DATASET=gibson python scripts/create_poni_dataset.py --split val

mkdir -p "$PONI_ROOT/experiments/gibson_poni_train_001"
cd "$PONI_ROOT/experiments/gibson_poni_train_001"
bash "$PONI_ROOT/experiment_scripts/gibson/train_poni.sh"
```

如果想先做一个很小的训练冒烟测试，不建议改原脚本，可以在新实验目录直接跑 `train.py` 并把 update 数量调小，同时指向同一个数据根：

```bash
mkdir -p "$PONI_ROOT/experiments/gibson_poni_smoke"
cd "$PONI_ROOT"

python -W ignore -u train.py \
  LOGGING.ckpt_dir "$PONI_ROOT/experiments/gibson_poni_smoke/checkpoints" \
  LOGGING.tb_dir "$PONI_ROOT/experiments/gibson_poni_smoke/tb" \
  LOGGING.log_dir "$PONI_ROOT/experiments/gibson_poni_smoke" \
  DATASET.root "$PONI_ROOT/data/semantic_maps/gibson/precomputed_dataset_24.0_123_spath_square" \
  DATASET.enable_unexp_area True \
  OPTIM.num_total_updates 5 \
  LOGGING.eval_interval 5 \
  LOGGING.ckpt_interval 5
```

这个 smoke test 只验证数据能读、模型能前向/反向、checkpoint 能写；不代表论文指标。

## 11. 训练好后 checkpoint 怎么进入在线导航

训练输出的 checkpoint 后续由 Gibson eval 脚本加载，例如：

```text
experiment_scripts/gibson/eval_poni.sh
  -> semexp/eval_poni.py
  -> semexp/model_pf.py
  -> Potential_Function_Semantic_Policy
  -> torch.load(MODEL_PATH)
```

在线评测时：

1. Habitat episode 给出当前 RGB/depth。
2. Mask R-CNN 预测语义。
3. semantic mapping 累积成 agent-centric/global semantic map。
4. PONI checkpoint 对当前已建地图预测 object PF 和 area PF。
5. 目标类别通道的 PF 与 area PF 结合，选 long-term goal/frontier。
6. FMM local planner 把 long-term goal 转成离散动作。
7. Habitat 执行动作并返回下一帧。

也就是说，PONI checkpoint 是在线模块化导航中的“长程探索目标选择器”，不是低层控制器，也不是语义分割器。

## 12. 可以替换/单独训练的部分

PONI 训练线里可以相对独立替换：

- 语义地图来源：只要能生成同样类别顺序、同样分辨率的 top-down semantic map，就可以重新生成 FMM 和 samples。
- PONI 势函数网络：`train_poni.sh` 训练的是 map output；`train_pred-xy.sh`、`train_pred-theta.sh`、`train_pred-act.sh` 是替代 baseline，不是 PONI 的附加头。
- area head：`DATASET.enable_unexp_area` 控制是否训练；关掉后只预测 object PF。
- 在线语义分割器：可替换，但输出类别顺序必须与 Gibson 17 类一致。
- 在线目标选择权重：评测脚本里的目标 PF / area PF 组合权重可调，这通常不需要重训。

必须联合保持一致：

- `DATASET.output_map_size=24.0`、`map_resolution=5cm` 要和在线 map crop 对齐。
- `object_pf_cutoff_dist=10.0` 要和 checkpoint 里的 cfg 对齐；训练和在线解释 PF 的距离尺度不能乱。
- Gibson 类别顺序必须从离线 semantic map、checkpoint、在线 semantic mapping 到 goal id 全部一致。
- 如果改 masking/frontier label 逻辑，必须重生成 precomputed dataset 并重训。

## 13. 一句话心智模型

Gibson PONI train 可以这样理解：

```text
先用完整 Gibson 语义场景离线制造大量“只看过一部分地图”的训练题；
题目的答案是：从这个局部已知地图出发，各类目标物体最可能/最值得探索的区域在哪里；
train.py 用这些题训练一个 U-Net；
在线 ObjectNav 时，这个 U-Net 作为模块化导航里的长程探索目标选择器。
```

