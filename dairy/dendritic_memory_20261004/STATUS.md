# 本次实施状态与实测记录

更新时间：2026-10-04 02:05（服务器 Asia/Shanghai）。

追加核验（02:09）：正式试训已采集 **41,037 步 / 221 episodes**，完成 **422 次优化**；
空间基线 dev 已完成 **65/100**。本轮所有新模块编译通过，88 个原文件哈希再次全部一致，
训练和基线驱动日志未发现 Traceback/OOM，流水线未报错。下文表格保留02:05快照用于追踪。

## 已交付

- 新代码目录：服务器 `/mnt/jfzn/shensj/PONI/dendritic_nav/`，本地同名目录保留镜像。
- 原 Sigma-D、原 PONI、独立 SNN 波前规划入口没有改动；本次仍使用原 FMM。
- 4×4×128 树突状态 + 胞体/读出 256 状态；每个真实步一次更新，零残差头初始化。
- 最多 128 候选、显式坐标访问/失败历史、在线反馈、完整 episode 回放、循环 Double DQN、独立采样/学习进程。
- 没有教师、蒸馏、运行时回退；空间 checkpoint 冻结。
- 配对内部 dev 评估、有限自动流水线、日志/检查点/源码哈希、离线可视化。
- MP3D 新树突接入暂未开始：按计划等待 Gibson 试训门槛，不把 Gibson 固定地图适配器误用于 MP3D 滑窗。

## 验证证据

| 验证 | 实际结果 | 证据相对路径 |
|---|---|---|
| 核心协议/数值/梯度单测 | 17/17 通过，正式存档用时约 1.95 秒 | `inventory/core_tests_v1/test.log`、`result.json` |
| 原文件隔离 | 88 个原源码、配置和空间 checkpoint 的 SHA256 全部未变 | `inventory/original_sha256.json` |
| Gibson 端到端前向及终止处理 | `integration_v01` 完成 2 个自然 episode | `gibson/integration_v01/dev_dend_000/episodes.jsonl`、`completion.json` |
| 实际训练回放 | `integration_v02` 采集 3 episodes / 1302 真实步；其中 1292 步执行树突候选、10 步目标可见接管 | `gibson/integration_v02/actor_000/replay/`、`episodes.jsonl` |
| 极短优化链路 | 从当时已有 802 回放步进行 8 次更新（B=2，窗口 8，warmup=0，仅为集成测试）；优化阶段约 5 秒 | `gibson/integration_v02/learner/metrics.jsonl`、`completion.json` |
| 梯度确实进入树突 | 四条输入投影、gain、mix、候选编码器、残差头参数均变化，固定 mask 与泄漏常数不变 | `gibson/integration_v02/learning_audit.json` |
| 零残差独立配对评估 | 同 4 episodes，Sigma-D 和初始树突版本 Success 均 25%，SPL 均 0.25，差值为 0 | `gibson/integration_v02/development_comparison.json` |
| 图像检查 | 已生成并查看轨迹、16 隔室活动、候选分数图；正式版增加预测地图底图及碰撞/目标切换标记 | 各 actor 下 `visualizations/` |

**4 episodes 的结果只证明初始接线一致，不是训练后提升或全量导航指标。**
其配对 bootstrap 区间为零只是因为每一对差值均为零，不代表有足够样本证明总体等效。

### 极短训练数值审计

- 最后一次 loss：`0.0003673999`，梯度范数 `0.0608711`；8 次更新无 NaN。
- 残差头权重最大绝对变化：`0.0007528020`。
- 四分支投影权重最大绝对变化：约 `0.0002124 / 0.0003479 / 0.0004937 / 0.0004824`。
- 训练后重放 1302 步，平均树突非零事件比例约 `8.57%`；1302 步残差均非零。
- 1292 个可决策回放步中，训练后 greedy 选点相对候选 0 改变了 679 次。
  这提示原空间峰值中有大量近似平分的候选，小残差也可能改变轨迹，所以不能仅凭 loss 变小声称导航更好。
- 上述事件比例来自数值参考，不等价于 GPU 计算量或芯片能耗已下降。

## 正在运行的正式试训

实验：`gibson/pilot_v01`。

| 项目 | 设置或当前状态 |
|---|---|
| 空间权重 | Gibson Sigma-D seed234/from30k 的 `best.ckpt`，冻结 |
| 训练场景 | 23 个官方 train 场景，1 场景/actor |
| 内部 dev | Klickitat、Shelbyville，训练不采这些场景 |
| 训练资源 | 23 个采样进程 + 1 个 learner |
| 正式训练 | 目标 50,000 步；完整 episode 边界停止，因此实际采集略超预算 |
| 优化参数 | warmup 10k、B16、窗口32、Adam1e-4、每64步一次更新、gamma0.995 |
| 02:05 采样进度 | 163 完整 episodes，25,721 步 |
| 02:05 学习进度 | 220 次更新；当次 loss 0.09932、梯度范数 0.13115，无 NaN 报错 |
| 同时进行的基线评估 | 20 路 × 5 episodes，共100；当前完成27 |
| 训练后评估 | 自动使用 `pilot_final.ckpt`，同20路/同随机流/同episode身份，合计100 |
| 完成后 | 生成 `development_comparison.json`；只有开发集 SR/SPL 点估计均不回退才生成 `best_dev.ckpt` |
| 后续 | 停在试训分析闸门，不盲目自动开始1M长训练或MP3D |

不要把 train 集 episode 的即时 success 当作与旧 Sigma-D 官方 val 的可比较指标。

### 实际资源快照

2026-10-04 02:05，`nvidia-smi` 显示 8 张卡都有实际作业：显存约 33.3–52.1GB/卡，利用率瞬时 37–86%。
包括此前已有的作业；未停止、重启或改写它们。
没有启动空占显存的填充进程。评估进程完成后会自然退出，显存占用不是固定承诺。

### 调度优化记录

最初启动 23 路训练 + learner，GPU6 当时已有约76GB占用，因此初始采样放在0/1/2/3/4/5/7。
GPU6 后续释放后，为满足尽量利用剩余资源的要求，把不依赖学习权重的空间基线 dev 提前到训练期间。
dev 从串行每场景50集改为20条固定随机流各5集；两个 dev 场景仍各50集，训练模型将使用完全相同的分流和种子。
这些是同一实验的并行采样随机流，不是20个训练seed实验。

只替换了本次新建的 CPU 流水线管理进程：旧 PID `3630588`，新 PID `3635273`。
停止前核验了 `/proc/PID/cmdline` 的模块名和实验路径；没有重启 learner、训练 actor 或任何原有任务。
详细旧/新命令保存在 `pilot_v01/scheduler_revision_v02.json`。
改动只涉及启动/调度和 dev 场景分流，神经网络、奖励、loss、训练数据输入和空间权重未变。

## 主要文件和检查点

以下路径均相对服务器 `/mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/`。

| 文件/目录 | 用途 |
|---|---|
| `README.md` | 详细接口、公式、结构、数据流、训练方法、原版/新版命令 |
| `inventory/core_tests_v1/` | 17项核心测试的原始日志及JSON |
| `inventory/original_sha256.json` | 原文件不改动的核验基准 |
| `gibson/integration_v02/checkpoints/initial.ckpt` | 初始零残差模型 |
| `gibson/integration_v02/learner/checkpoints/pilot_final.ckpt` | 仅8更新的训练链路测试产物，不是推荐模型 |
| `gibson/integration_v02/learning_audit.json` | 参数变化、事件活动、回放选点变化 |
| `gibson/integration_v02/development_comparison.json` | 4 episode 的零残差配对检查 |
| `gibson/pilot_v01/checkpoints/initial.ckpt` | 正式试训初始权重 |
| `gibson/pilot_v01/checkpoints/published.ckpt` | actor 在episode边界读取的最新发布权重 |
| `gibson/pilot_v01/learner/checkpoints/last.ckpt` | 最近可恢复的训练状态，包括optimizer/target/RNG |
| `gibson/pilot_v01/learner/checkpoints/pilot_final.ckpt` | **训练完成后才生成**；不是预先存在的结果 |
| `gibson/pilot_v01/learner/checkpoints/best_dev.ckpt` | **开发评估合格后才生成**，不能提前使用 |
| `gibson/pilot_v01/learner/metrics.jsonl`、`tensorboard/` | 实际训练曲线 |
| `gibson/pilot_v01/actor_*/replay/` | 完整训练轨迹，不混入dev |
| `gibson/pilot_v01/actor_*/visualizations/` | 训练样例图 |
| `gibson/pilot_v01/dev_baseline_*/` | 100个内部dev的纯空间参考结果 |
| `gibson/pilot_v01/dev_dend_*/` | **训练结束后**的树突dev结果 |
| `gibson/pilot_v01/launch_train.json` | 训练、管理进程PID/GPU/命令 |
| `gibson/pilot_v01/launch_dev_baseline.json` | 20路基线评估PID/GPU/命令 |
| `gibson/pilot_v01/logs_train/`、`logs_dev_baseline/` | 独立驱动日志 |
| `gibson/pilot_v01/logs_train/pipeline_v02.log` | 当前流水线原始日志 |
| `gibson/pilot_v01/pipeline_progress.json` | 所处阶段和等待PID |
| `gibson/pilot_v01/pipeline_complete.json` / `pipeline_failure.json` | 完成/异常时才写入 |
| `gibson/pilot_v01/source_snapshot.zip` | 初次启动时的源码归档 |
| `gibson/pilot_v01/source_snapshot_scheduler_v02.zip` | 调度并行化版本源码归档 |
| 每个run的 `command.sh/config.json/manifest.json/code_sha256.json` | 实际启动参数、环境和代码版本 |

## 首次实现中修正的问题

1. 远端 PyTorch 不支持 `torch.inf` 常量：改为 `float("inf")`，不改变算法。
2. 减法复位若直接使用 STE，会抵消残差状态梯度：使用 detach reset，并增加跨两步梯度测试。
3. wait 步不能推进树突时钟：预计算后若实际未执行动作，撤销树突状态和地图差分缓存。
4. 自动reset不能吞掉terminal：在新环境子类中保存并传回终止转移，与下一episode初始化信息分开。
5. 进程还在加载模型时显存统计偏低：新版调度器先等待所有训练actor完成首个episode，再准入提前评估，避免启动阶段过量分配。

没有通过修改旧 `.py` 来修补上述问题。
