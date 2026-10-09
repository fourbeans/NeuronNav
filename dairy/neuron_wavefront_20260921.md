# SNN 波前规划独立分支：接口分析、改动与验证记录

## 目标与边界

在 PONI 当前工作树（HEAD `30682c2`）新增带 `_neuron.py` 后缀的模块，比较脉冲波前与 Gibson FMM、MP3D 多目标加权 A* 的规划行为。原有 `.py` 文件不覆盖；现有用户修改不回退；PONI checkpoint、分割、建图、任务成功定义和离线 GT FMM 保留。此次为软件事件仿真，不是已经上类脑芯片，也不包含 E-prop 在线学习。

源代码根目录：`/mnt/jfzn/shensj/PONI`。原始文件 SHA256 快照、运行命令、stdout、逐步 JSONL、地图快照及汇总结果统一放在 `dairy/neuron_wavefront_20260921/`。

这里不是把 A*/FMM 的 ANN 权重转换成 SNN：它们原本就是传统规划算法。本分支把栅格映射成神经元，把移动代价编码为突触/轴突延迟，以首次发放时间求路径；不修改 PONI U-Net、Mask R-CNN 或语义地图融合。此独立最小实现也不声称是先前提到论文的官方代码或完整复现。

## 1. 为什么两条链路必须分别适配

### Gibson

- 入口：`semexp/eval_poni.py:main` → `semexp/envs:make_vec_envs` → `semexp/envs/habitat:construct_envs` / `make_env_fn` → `Sem_Exp_Env_Agent`。
- `_plan` 先更新当前位置、visited 和 collision_map，再调用 `_get_stg`；返回的短期目标通过相对角度选择前进/左右转；只有 `stop && found_goal` 才输出 STOP。
- `_get_stg`（`semexp/envs/habitat/sem_exp.py:509`）先做障碍膨胀、碰撞遮挡、已访问格点放行、起点 3×3 放行、边界填充，再扩张目标掩码。目标可以是一片已见物体或一个探索点。
- 原距离场由 `skfmm.distance` 求得；原 `get_short_term_goal` 在起点附近环带取值，并使用距离下降比率和 25 cm 距离门限。八邻域图距离不严格等于 FMM 的 Eikonal 距离，因此不能承诺直接替换后逐步动作一致。
- 新实现以目标集合为多源发放脉冲，计算到目标的距离，保留原环带解码。保留原目标格点覆盖 masked 障碍的语义；这是复现原行为，不是额外的物理可通行保证。

### MP3D

- `hlab/eval_poni.py` → `TransferEvaluator` → `GlobalAgent` → `PlannerActorSequential` 或 `PlannerActorVector`。
- 当前 `_get_stg` 无条件调用 `pyastar.multi_goal_weighted_astar_planner`；`enable_weighted=False` 不会切换实际分支。
- `dependencies/astar_pycpp/astar.cpp:350` 使用八邻域，直线距离 1、对角距离 sqrt(2)，边代价为“几何长度 × 目的节点净空权重”。默认 `wscale=4, niters=1`，障碍邻近自由格点代价更高；障碍本身不可进入。
- 新实现从起点发放脉冲，任一目标格点首次触发即终止，父节点链生成起点到目标路径。严格保留默认 step_size=5 的路径索引，以及短路径几何长度≤0.25 m的 stop 条件。
- 原 A* 使用目标连通区域质心计算启发式，却可在区域内任一目标格点停止，因此它不是这里的图最优性真值。新增独立 SciPy Dijkstra oracle 检查 SNN 最短路，并另行比较原 A*。
- 邻接规则保留原八邻域的角穿越语义；修改该规则需独立消融。`stg_downsampling != 1` 暂未验证，记录原因并回退，避免复制原代码单轴降采样的可疑行为。
- 保留原控制器的 `is_close_to_goal`、`move_close_limit`、连续 replan 计数与地图重置链。

## 2. 新模块与原因

| 新文件 | 职责与原因 |
|---|---|
| `poni/wavefront_neuron.py` | 单次发放、阈值 1、无泄漏、查询内持续抑制；以延迟编码边代价。每次规划重新初始化状态，避免地图变更后残留脉冲。可用 Numba 加速 CPU 事件仿真。|
| `poni/planner_adapter_neuron.py` | 分别实现 Gibson 图距离+原 FMM 环带解码，以及 MP3D 净空加权图路径+原路径索引。默认浮点延迟；可显式实验延迟量化。|
| `poni/runtime_neuron.py` | 模式选择、保护判定、每进程 JSONL 和有限数量的地图快照。所有候选失败与回退均可追踪。|
| `semexp/envs/habitat/sem_exp_neuron.py` | 继承原 agent，仅替换 `_get_stg` 和增加审计记录；提供可被 forkserver/spawn 导入的顶层环境工厂。|
| `semexp/eval_poni_neuron.py` | 转发原评测参数，只在新入口进程中把环境工厂指向新模块，原文件不动。|
| `hlab/utils/planners_neuron.py` | 继承原 PlannerActor 控制器，新增顺序/多进程封装；子进程明确创建新 planner，避免只在主进程替换成功。|
| `hlab/eval_poni_neuron.py` | 在加载原评测器之前接入新顺序/多进程规划器。|
| `scripts/validate_planner_neuron.py` | 独立图最短路、同步时钟脉冲 oracle、动态地图、无路径、多目标、原控制器状态、IPC、真实地图重放验证。|
| `scripts/smoke_gibson_neuron.py` | 顺序启动少量相同 episode 的 reference/guarded/neuron 评测；每次创建全新日志目录，不覆盖历史实验。|
| `scripts/compare_runs_neuron.py` | 对照场景、起点、目标、动作轨迹、SR/SPL/DTS，汇总候选动作一致率、回退率和软件运行耗时。|

## 3. 模式与“保证效果”的具体含义

- `reference`：执行原规划器，同时记录接口和 episode，用作同设置的对照。
- `shadow`：两者都计算，执行原结果；用于采集候选偏差。
- `guarded`（默认）：两者都计算；仅当短期目标会产生相同转向/前进动作，且 STOP 相同、MP3D replan 相同时采用 SNN；否则使用原结果并记录原因。在相同输入、原控制器与确定性环境下，它保持动作和状态转换一致。实际环境仍须逐条比较。
- `neuron`：有有效 SNN 候选时不调用原规划器；候选异常/不可达等情况仍做安全回退并明记日志。因此这是无动作一致性保护的实验模式，有回退时不能称为纯 SNN 端到端结果。

保护模式仍付出原规划器成本，不能用于证明加速/省电。SNN 核心数学上是正权图的 first-arrival/min-plus 求解；软件事件队列采用与 Dijkstra 相同的时间排序结构，因此不能将 CPU 事件仿真等同于硬件上的并行收益。默认浮点延迟是算法验证配置，硬件有限位宽、最大延迟、路由和传输开销均未验证。

## 4. 验证计划与完成标准

1. 新文件语法检查；核验原 `.py` SHA256 没有改变。
2. 60 个随机加权、多源、正向/反向、量化图与独立 SciPy oracle 一致；一个独立逐时钟 IF 仿真对照；检查动态障碍重置、孤立区域、起点已在目标、预算和非法延迟。
3. 80 个障碍/多目标图对比真实 C++ 加权 A* 的可达性和路径代价；独立循环验证净空权重。
4. 两条真实控制器接口测试，包括 MP3D 连续三步碰撞/靠近目标状态、两个规划子进程及 Gibson 工厂 pickle。
5. Gibson 小样本：同一 seed123 checkpoint、seed100、Box episodes、500 步上限、原 Mask R-CNN 与 PONI、seg_interval=3，分别运行三种模式；只动规划后端。必须比较逐 episode 起点/目标以及原始指标，而不是与历史1000条平均值直接比。
6. 重放 reference 真实在线地图，报告候选的短期目标/动作/STOP 一致情况。
7. MP3D 缺 3D 场景时只报告数值/接口/IPC 验证，明确不声称完成 MP3D ObjectNav。

历史 Gibson Box 1000 条 seed123 基线为 SR=0.731、SPL=0.3968856141、DTS=1.3911093361，仅供背景。小样本结果不支持总体统计不劣结论，更不支持硬件性能结论。

## 5. 执行结果

### 5.1 已通过的 CPU、接口与多进程检查

环境为服务器现有 `poni` conda 环境，Python 3.8，Numba 可用。`micro_v1` 为首次验证；`micro_v2` 增加保护边界测试后复验，结果均为 `passed`。

| 检查 | 实际结果 |
|---|---|
| 独立 SciPy 最短路 oracle | 60/60，通过多源、方向性反向传播和延迟量化检查 |
| 独立逐时钟 IF 仿真 | 通过；验证的是脉冲传播时序，不只比较两个同结构堆队列实现 |
| 动态地图重置/断连/起点在目标/预算/非法权重 | 通过 |
| 原 C++ 加权 A* 对比 | 80/80 可达性一致；72 个可达图代价全部一致，8 个不可达；72 个中46个路径序列相同，其余是等代价路径 |
| Gibson 原 `_plan` 保护测试 | 36/36 最终动作一致 |
| MP3D 原控制器连续状态测试 | 36 组×3步，动作及相关状态一致 |
| MP3D 实际双 worker IPC | 通过 |
| Gibson 顶层工厂 pickle | 通过，随后由真实 Habitat 评测进一步检查 |
| STOP/转向/replan 不同、候选异常 | 均按预期回退；相同动作正常采用候选；空目标拒绝 |

这些测试证明所测图上的正确性与保护逻辑，不等于两个数据集全量导航统计不劣。MP3D 场景目录 `data/scene_datasets/mp3d` 缺失，因此不能在本环境宣称完成 MP3D ObjectNav。

### 5.2 真实 Gibson 小样本协议与资源调整

首次 `gibson_smoke_v1` 用5环境并行，在 reference/FMM 阶段因显存不足失败（`semexp/model.py:333`，额外申请176 MiB失败）。当时服务器8张 A100 均已有较重的其他任务负载；这不是 SNN 失败，也没有计入导航指标。失败命令、stdout 和返回码完整保留，未终止或修改其他用户进程。

改为 `gibson_smoke_v2`：Collierville、Corozal、Darden 三场景，每场景第1条 episode，单环境依次运行 reference、guarded、neuron 共9次。新 `--neuron-scene` 只在新工厂中限制运行时场景列表，不改任何原 YAML/数据集；所有模式使用同一场景/起点/目标协议。新测试脚本默认并发也降为1。模型、地图480×480、5 cm分辨率、500步上限、原成功判定均未因此改变。

按用户最新要求，本轮不等待评测结束、不继续启动 MP3D 数据集验证。Gibson `gibson_smoke_v2` 继续运行；最终文件核验时，三个 reference 场景和 guarded/Collierville 已完成，正在执行 guarded/Corozal，其余由批次顺序执行。远端驱动 PID 为 `3683359`（PID 只描述本次运行，不可用作后续盲目终止目标）。暂不据此分析 SR/SPL 或声称端到端验证完成。

完成标志是 `gibson_smoke_v2/run_status.json`：仅全部9次评测成功退出后写出。任一评测失败会保存该场景 `run_status.json` 和 `eval.log` 并停止批次；不将部分结果伪装成完整对照。比较脚本新增完整性检查，防止批次仍在跑时生成误导性汇总。

MP3D 曾做新入口 `--help` 导入检查，发现只配置项目根 `PYTHONPATH` 时不能找到项目内 `pyastar`。已在新的 `hlab/eval_poni_neuron.py` 显式加入 `dependencies/astar_pycpp` 导入路径；遵照用户要求不再启动 MP3D 评测。本次只做修正后的语法检查，完整评测入口仍待有数据集时验证。此前 CPU 数值、真实 planner 接口及双 worker IPC 已验证的结论不变。

### 5.3 运行完成后再分析的项目

1. 三模式实际 episode 起点、朝向、目标是否逐条相同。
2. guarded 与 reference 的动作键、动作序列和原始 SR/SPL/DTS 是否一致。
3. neuron 的独立指标、相对差值、异常及回退次数；回退非零则不是纯 SNN 结果。
4. 参考轨迹真实地图重放的动作/STOP一致率，结合 guarded 候选采用率解释偏差。
5. 若独立 neuron 导航下降，先定位 FMM 距离/环带解码与碰撞附近的行为；不通过修改成功门限、扩大目标容差或替换感知 checkpoint 美化结果。

## 6. 复现命令与接入方式

以下均在服务器执行。结果目录需要使用新的名称，避免混合或覆盖旧实验。

```bash
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/jfzn/miniconda3/envs/poni/bin/python scripts/validate_planner_neuron.py \
  --output dairy/neuron_wavefront_20260921/micro_new

/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/smoke_gibson_neuron.py \
  --output dairy/neuron_wavefront_20260921/gibson_new --gpu 0 \
  --processes 1 --episodes-per-process 1 --scenes Collierville Corozal Darden
```

已有启动命令只需在独立实验中把 `eval_poni.py` 改成同目录 `eval_poni_neuron.py`，添加 `--neuron-mode guarded --neuron-log-dir <全新目录>`，其他原参数保持。Gibson 从 `semexp` 目录运行，MP3D 从 `hlab` 目录运行；两者都保留项目根在 `PYTHONPATH`。默认 `--neuron-delay-quantum 0`，不要把量化图最短路测试通过解释成量化后导航质量已保证。

回到原启动文件即可完全停用本分支；没有替换原 checkpoint、默认配置或原 `.py`。保护模式首先服务于效果保留，后续只有在更大的独立 `neuron` 消融达到预设非劣标准后，才适合考虑减少原规划器计算。

当前批次进展：

```bash
cd /mnt/jfzn/shensj/PONI
tail -n 20 dairy/neuron_wavefront_20260921/gibson_smoke_v2_driver.log
```

全部完成后生成三模式对照（本次交接尚未执行）：

```bash
cd /mnt/jfzn/shensj/PONI
/mnt/jfzn/miniconda3/envs/poni/bin/python scripts/compare_runs_neuron.py \
  --runs dairy/neuron_wavefront_20260921/gibson_smoke_v2/reference \
         dairy/neuron_wavefront_20260921/gibson_smoke_v2/guarded \
         dairy/neuron_wavefront_20260921/gibson_smoke_v2/neuron \
  --output dairy/neuron_wavefront_20260921/gibson_smoke_v2/comparison.json

PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/jfzn/miniconda3/envs/poni/bin/python scripts/validate_planner_neuron.py \
  --replay-glob 'dairy/neuron_wavefront_20260921/gibson_smoke_v2/reference/*/audit/*.npz' \
  --output dairy/neuron_wavefront_20260921/replay_v1
```

重放脚本不加载 MP3D 场景。日志里的软件毫秒数受 Numba 热身、CPU/GPU 争用影响，不能推导芯片延迟或能耗。

## 7. 文件保全与最终交接核验

- 修改前记录1143个已有 `.py` 文件 SHA256；最终 `sha256sum -c ... --quiet` 返回0，全部原文件内容未变。原工作树已有修改也保留。
- 10个新增 `_neuron.py` 均通过服务器 Python 3.8 的 `py_compile`。仅更新本次新建文件，没有覆盖用户已有 Python 文件。
- `original_py_sha256.txt`：修改前原文件清单；`original_py_verify_final.log`：最后核验输出（成功时为空）；`git_status_before.txt` / `git_status_handoff.txt`：前后工作树状态；`new_files_sha256.txt`：新源码与本文档哈希。
- CPU 最后完整测试记录：`micro_v2/validation.json`、`micro_v2.log`。Gibson 未完成批次：`gibson_smoke_v2_driver.log` 及 `gibson_smoke_v2/`。失败的五环境尝试：`gibson_smoke_v1_driver.log` 及 `gibson_smoke_v1/`，保留用于追溯。
- 尚未执行：完整三模式指标分析、真实地图重放、MP3D 数据集导航、类脑硬件部署。没有把这些项目写成已通过。
