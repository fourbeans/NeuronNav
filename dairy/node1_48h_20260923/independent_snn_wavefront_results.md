# 独立 SNN 波前规划完整实验结果

## 1. 结论先行

本轮独立 SNN 波前实验已经全部跑完，所有正式任务退出码均为 0：

- Gibson：5 个场景 × 200 episodes，共 1,000 episodes。
- MP3D：11 个 val parts，共 2,195 个有效 episodes；原版与 SNN 的 episode ID 完全一致。
- 严格 SNN 审计：Gibson 164,496 次、MP3D 607,718 次规划全部为 `used=neuron`，且 `reference=null`、`fallback=false`、`error=null`。

结论分数据集：

1. **MP3D 已达到“接近或不低于 weighted A*”的目标。** Success 基本持平，30.57% → 30.66%；SPL 从 11.61% 提升到 12.58%。严格 SNN 没有调用 weighted A*。
2. **Gibson 目前只能判定为接近 FMM，但还没有达到“不降效果”的要求。** Success 从 73.10% 降到 70.60%，SPL 从 39.69% 降到 36.62%。严格 SNN 没有调用 FMM，但还不能直接替换正式 FMM 基线。
3. Gibson 的主要失败区间是严格 SNN 遇到地图断连/no-path 的 episodes；当前 8 邻域图距离与连续 FMM Eikonal 距离的差别也造成了路径效率损失。

## 2. 可比性与有效性检查

本次只替换局部规划器，PONI checkpoint、语义建图、目标势函数、数据集 split、评测 seed 和动作控制链保持不变。正式对比均使用 checkpoint 123 和评测 seed 100。

旧 `neuron` 入口不是独立 SNN：

| 数据集 | 旧入口规划次数 | 回退到原规划器 |
|---|---:|---:|
| Gibson | 169,933 | 29,910 次 FMM 回退 |
| MP3D | 607,718 | 12,380 次 weighted A* 回退 |

因此旧入口只作为兼容性诊断，不用于“独立 SNN”结论。新增严格入口的审计结果为：

| 数据集 | 严格规划次数 | 调用 reference | fallback | planner error |
|---|---:|---:|---:|---:|
| Gibson | 164,496 | 0 | 0 | 0 |
| MP3D | 607,718 | 0 | 0 | 0 |

Gibson 原版的 1,000 个 episode 与严格版逐场景结果通过 `goal_distance` 序列精确对齐；MP3D 原版和严格版的 2,195 个 episode ID 完全一致，因此可以进行逐 episode 配对统计。

## 3. Gibson：严格 SNN 对原 FMM

### 3.1 总体指标

| 指标 | 原 FMM | 严格 SNN | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Success | 73.10% | 70.60% | -2.50 pp | -3.42% |
| SPL | 39.69% | 36.62% | -3.07 pp | -7.73% |
| DTS，越低越好 | 1.3911 | 1.4737 | +0.0826 | +5.94% |

逐 episode 配对 bootstrap：

- Success 差值 95% CI：[-5.10 pp, +0.10 pp]；McNemar `p=0.0725`。Success 的下降在 5% 水平没有达到统计显著，但点估计和实际幅度仍不能忽略。
- SPL 差值 95% CI：[-4.72 pp, -1.43 pp]。当前严格 SNN 的路径效率确实低于 FMM。

### 3.2 逐场景结果

| 场景 | FMM Success | SNN Success | ΔSuccess | FMM SPL | SNN SPL | ΔSPL |
|---|---:|---:|---:|---:|---:|---:|
| Collierville | 80.5% | 76.0% | -4.5 pp | 50.79% | 46.06% | -4.73 pp |
| Corozal | 78.0% | 73.0% | -5.0 pp | 44.59% | 40.50% | -4.09 pp |
| Darden | 68.0% | 66.5% | -1.5 pp | 33.28% | 31.57% | -1.72 pp |
| Markleeville | 65.0% | 62.0% | -3.0 pp | 30.61% | 29.53% | -1.08 pp |
| Wiconisco | 74.0% | 75.5% | +1.5 pp | 39.17% | 35.44% | -3.73 pp |

SNN 在 Wiconisco 的 Success 略高，但五个场景的 SPL 都低于 FMM，说明不仅是少数场景失败，当前图距离也存在较稳定的路径效率差距。

### 3.3 no-path 分析

Gibson 使用从目标反向传播的单脉冲波前。当前实现的距离模型是 8 邻域图最短路，而不是 FMM 的连续 Eikonal 距离。严格入口在波前到不了 agent 起点时，不调用 FMM，而是用“不可达常数距离场 + 原环形短期点解码器”继续控制。

| 场景 | 规划次数 | no-path 查询 | 查询占比 | 出现 no-path 的 episodes |
|---|---:|---:|---:|---:|
| Collierville | 31,792 | 15,518 | 48.81% | 42/200 |
| Corozal | 36,532 | 47 | 0.13% | 3/200 |
| Darden | 33,764 | 2,680 | 7.94% | 13/200 |
| Markleeville | 30,528 | 8,970 | 29.38% | 30/200 |
| Wiconisco | 31,880 | 473 | 1.48% | 9/200 |
| **总计** | **164,496** | **27,688** | **16.83%** | **97/1000** |

按“严格 SNN 是否在该 episode 中遇到过 no-path”分组：

| episode 分组 | 数量 | FMM Success | SNN Success | FMM SPL | SNN SPL |
|---|---:|---:|---:|---:|---:|
| 未遇到 no-path | 903 | 77.19% | 77.19% | 42.51% | 40.25% |
| 遇到 no-path | 97 | 35.05% | 9.28% | 13.38% | 2.83% |

这说明：

- 在没有出现断连的 903 个 episodes 上，SNN 与 FMM 的 Success 完全相同，证明正常波前传播已经能维持到达能力。
- 同一组 episodes 上 SNN SPL 仍低 2.26 pp，符合“8 邻域图距离不等于 FMM Eikonal 距离、局部 waypoint 不完全一致”的预期。
- 97 个遇到 no-path 的 episodes 是 Success 损失最集中的区间。这个分组依据 SNN 自身轨迹，属于强相关证据；要严格证明因果，仍需对 no-path 修复做单变量消融。
- Collierville 和 Markleeville 一旦断连，会在后续许多导航步重复 no-path，因此查询占比远高于受影响 episode 占比。

## 4. MP3D：严格 SNN 对 weighted A*

### 4.1 总体指标

| 指标 | weighted A* | 严格 SNN | 绝对变化 | 配对 bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Success | 30.57% | 30.66% | +0.09 pp | [-1.59 pp, +1.78 pp] |
| SPL | 11.61% | 12.58% | +0.98 pp | [+0.21 pp, +1.76 pp] |
| SoftSPL | 17.31% | 18.85% | +1.55 pp | [+0.80 pp, +2.27 pp] |
| Distance to goal，越低越好 | 5.2764 | 5.3309 | +0.0545 | [-0.1449, +0.2539] |

Success 转移为：

- A* 成功、SNN 失败：177 episodes。
- A* 失败、SNN 成功：179 episodes。
- 两者都成功：494 episodes。
- 两者都失败：1,345 episodes。
- McNemar `p=0.9577`，没有证据认为两者 Success 不同。

SPL 与 SoftSPL 的配对 bootstrap 区间均为正。就本次 checkpoint/seed 而言，严格 SNN 不仅保持 Success，而且平均路径效率略优于 weighted A*。

### 4.2 逐分片结果

| Part | Episodes | A* Success | SNN Success | A* SPL | SNN SPL |
|---:|---:|---:|---:|---:|---:|
| 0 | 200 | 42.00% | 38.50% | 18.63% | 17.19% |
| 1 | 201 | 11.94% | 11.44% | 4.82% | 5.23% |
| 2 | 198 | 34.85% | 30.81% | 11.55% | 11.20% |
| 3 | 197 | 21.32% | 22.84% | 9.39% | 10.24% |
| 4 | 200 | 29.00% | 30.50% | 8.27% | 11.29% |
| 5 | 200 | 12.00% | 12.00% | 4.85% | 5.18% |
| 6 | 200 | 42.50% | 43.00% | 19.14% | 20.62% |
| 7 | 201 | 31.34% | 33.83% | 11.94% | 14.28% |
| 8 | 200 | 29.50% | 33.50% | 7.17% | 7.96% |
| 9 | 200 | 33.50% | 32.00% | 14.64% | 14.90% |
| 10 | 198 | 48.48% | 48.99% | 17.30% | 20.40% |

MP3D 的 SNN 波前使用与原 `astar.cpp` 相同的障碍邻近权重构造，并通过脉冲首达时间求正权 8 邻域图最短路。它在数学上等价于 delay-coded Dijkstra/min-plus 传播；与 A* 的主要差别是搜索顺序、路径 tie-breaking 和局部动作解码，而不是学习得到的近似网络。

严格 MP3D 中有 12,380/607,718=2.04% 的查询发生 no-path，其中：

- `disconnected_goal`：12,004 次。
- `empty_or_blocked_goal_mask`：376 次。

这些查询使用上游随机短期点并设置 `replan=True`，没有调用 A*。严格版 11 个分片的最终 `stats.json` 与旧可回退版逐字节相同，说明本轮中用独立 no-path 分支移除 A* 回退后，没有改变 episode 级结果；旧版仍因实际调用过 A* 而不能单独作为独立 SNN 证据。

## 5. 计算量与部署含义

以下延迟是当前服务器上的软件事件模拟时间，不代表类脑芯片延迟或能耗：

| 数据集 | 平均规划延迟 | P95 | 平均 spikes/query | 平均发射事件/query |
|---|---:|---:|---:|---:|
| Gibson | 71.11 ms | 96.78 ms | 219,697 | 1,746,139 |
| MP3D | 22.96 ms | 51.67 ms | 52,740 | 417,014 |

Gibson 事件量明显更高，是因为当前实现从所有目标反向传播并基本填满整张可行地图，之后再用 FMM 的环形短期点解码规则取 waypoint。MP3D 从起点传播，到首次到达目标即停止，因此能够早停。

这意味着当前结果证明了**算法效果可行性**，但还没有证明芯片效率优势。尤其 Gibson 的每次查询约 22 万 spikes、175 万突触事件，仍需做事件裁剪和早停设计。

## 6. 下一步修改优先级

### P0：先修 Gibson no-path，保持严格 SNN

不引入 FMM 安全回退。建议增加可独立开关的 SNN-only 消融：

1. 从 agent 起点求可达连通域。
2. 目标落在不可达分量时，将目标代理点投影到起点可达域中、距离原目标最近且满足安全距离的边界/前沿位置。
3. 仍由 SNN 波前传播和脉冲首达时间选路。
4. 记录 `goal_projection_distance`、修复前后连通性和是否重复 no-path。

目标是先把 Gibson no-path 查询率从 16.83% 压到 1%–2% 以下，并恢复 97 个困难 episodes 的 Success。

### P1：减小 Gibson 与 FMM 的距离度量差异

当前仅有横/竖 `1`、对角 `sqrt(2)` 的 8 邻域 delay，不能精确近似连续 Eikonal 解。可以逐项验证：

- 更高阶 chamfer stencil，例如增加 `(1,2)`、`(2,1)` 方向及对应 delay。
- 局部方向相关 delay 或多分支树突整合，近似 FMM 的各向同性传播。
- 保持单脉冲首达和无矩阵状态，不引入 Softmax/QK/RMSNorm。

验收重点是“无 no-path 的 903 episodes”中 SPL 从 40.25% 回到 FMM 的 42.51%附近。

### P2：降低 Gibson 全图传播事件量

建议开发独立版本比较：

- 从起点正向传播，到目标首达后沿 predecessor path 直接取前 5 个栅格的 waypoint；或
- 反向传播只持续到起点及局部解码环所需的时间窗，而不是耗尽全图事件队列。

验收指标除 SR/SPL 外，增加 spikes/query、events/query、P50/P95 planner latency 和地图尺寸归一化事件密度。

### 建议实验顺序

1. 当前严格版，作为冻结基线。
2. 仅加入 no-path 可达目标投影。
3. no-path 修复 + 高阶空间 stencil。
4. 在效果恢复后再加入早停/事件裁剪。

每一步先跑 5 场景 × 20 episodes 小验证；只有 Success、SPL 和 no-path 指标同时改善，才跑完整 1,000 episodes。MP3D 当前效果已经通过，不建议在同一轮同时改动其代价模型。

## 7. 复现与产物

重新生成机器可读汇总：

```bash
cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
python scripts/summarize_wavefront_results_neuron.py \
  --root dairy/node1_48h_20260923 \
  --output dairy/node1_48h_20260923/independent_snn_wavefront_summary.json
```

结果文件：

```text
dairy/node1_48h_20260923/independent_snn_wavefront_results.md
dairy/node1_48h_20260923/independent_snn_wavefront_summary.json
scripts/summarize_wavefront_results_neuron.py
```

原始实验矩阵、启动命令、代码哈希和输出目录继续保存在同一目录的 `README.md`。

## 8. 当前判断

- **MP3D：通过本阶段验证。** 可以把严格 SNN 波前作为 weighted A* 的候选替换分支，下一步重点转向多 seed 复验和芯片事件映射。
- **Gibson：部分通过。** 正常连通 episodes 的 Success 已与 FMM 持平，说明核心脉冲波前可用；但 no-path 处理和距离场近似导致总体 SPL 明显下降，修复前不能宣称不弱于 FMM，也不应替换正式基线。

