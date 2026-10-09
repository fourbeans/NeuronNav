# Sigma-D最终结果与待补实验（2026-09-29）

## 1. 当前状态

- Gibson与MP3D的Sigma-D完整训练均已结束。
- Gibson原版/Sigma-D完整导航各为1,000 episodes；MP3D原版/Sigma-D各为2,195 episodes。
- MP3D原版11个part于2026-09-28 21:43 CST全部完成，episode键与Sigma-D完全一致。
- 没有发现Traceback、CUDA OOM、RuntimeError、FileNotFound、Killed或分片缺失。
- 2026-09-29 14:04 CST没有科学训练或评估在运行；8张GPU仅运行显存占位器，约42.98 GB/卡，利用率均为0%。

## 2. MP3D严格同协议对比

共同条件：PONI训练seed123、导航seed100、2,195 episodes、原weighted A*、相同语义建图和评估配置。

| 指标 | 原PONI | Sigma-D best@2k | Sigma-D减原版 |
|---|---:|---:|---:|
| Success | 30.5695% | 30.5239% | -0.0456 pp，即少1个成功episode |
| SPL | 11.6061% | 11.8189% | +0.2128 pp |
| Distance-to-goal，越低越好 | 5.2764 | 5.2369 | -0.0395 |

逐episode配对统计：

- Success：Sigma-D独有成功205个，原版独有成功206个，McNemar精确检验`p=1.0`；差值近似95%区间为[-1.856, +1.765] pp。
- SPL：平均差值+0.2128 pp，近似95%区间为[-0.568, +0.993] pp。
- Distance-to-goal：平均差值-0.0395，近似95%区间为[-0.2460, +0.1671]。

因此本轮MP3D应表述为：Sigma-D在单个训练seed/导航seed上达到与原PONI近似等价的导航效果，未观察到实际退化；没有统计证据支持“显著更好”。如果要正式使用“非劣”措辞，需要在新增seed实验前预先约定非劣界值。

结果文件：

```text
dairy/sigma_d_unet_20260924/navigation/mp3d_original_seed123_8gpu/summary_2195.json
dairy/sigma_d_unet_20260924/navigation/mp3d_seed123_best2k_4gpu/summary_2195.json
```

## 3. Gibson当前结论及公平性问题

| 模型 | Success | SPL |
|---|---:|---:|
| 原PONI，训练seed123 | 71.70% | 37.56% |
| Sigma-D，训练seed123 best@1k | 69.20% | 36.64% |
| Sigma-D，训练seed234 best@32k | 72.90% | 42.17% |

Sigma-D seed234的点估计优于原版seed123，但这是不同训练seed，不能单独证明结构优于原版。服务器已有`pretrained_models/gibson_models/poni_seed_234.ckpt`，所以最高优先级缺口是用相同导航协议评估原PONI seed234，再与Sigma-D seed234配对比较。

## 4. 待补实验优先级

### P0：当前结论必须补齐

1. **Gibson同训练seed公平基线**：原PONI seed234与Sigma-D seed234 best@32k，导航seed100、1,000 episodes、原FMM。
2. **导航seed复验**：至少再取2个预先固定的导航seed，原版与Sigma-D成对运行；Gibson和MP3D都需要。报告跨seed均值、标准差和逐episode配对区间。
3. **预设非劣界值**：建议在复验前写死Success与SPL可接受下降界值，避免看完结果后再选择界值。

### P1：证明“增量脉冲”设计确实有用

1. **状态消融**：同一权重比较跨导航步保留Sigma-Delta状态与每步清空状态。
2. **增量编码消融**：比较增量事件输入与等价的逐帧稠密输入；不把同一地图人为重复T次。
3. **部署统计**：逐层记录事件率、spikes/synaptic-events、峰值状态内存、GPU软件延迟，并给出映射到类脑芯片时的AC/MAC与能耗代理。当前效果实验本身不能证明芯片节能。

### P2：两条SNN线路的联合系统

1. **MP3D联合评估**：Sigma-D PONI加严格SNN weighted-wavefront。两项独立实验都已通过，下一步需要验证组合后是否仍保持2,195-episode效果。
2. **Gibson暂不直接联合**：严格SNN波前当前相对FMM为Success -2.50 pp、SPL -3.07 pp。先完成no-path可达目标投影、高阶stencil和事件早停，再与Sigma-D组合。

### P3：论文级稳健性，而非当前原型的立即阻塞项

1. MP3D新增Sigma-D训练seed234/345；Gibson补Sigma-D seed345，和已有原PONI同seed checkpoint做训练seed级统计。
2. 按场景、目标类别、起始距离、地图变化量分桶，检查平均值是否掩盖困难子集退化。
3. 在上述结果稳定后再决定是否增加简化树突记忆；它不是当前Sigma-D基线闭环的必需项。

## 5. 当前阶段判断

- **MP3D Sigma-D U-Net：本阶段基本通过，但仍是单训练seed/单导航seed证据。**
- **Gibson Sigma-D U-Net：找到了效果较好的checkpoint，但必须补原PONI seed234公平基线。**
- **完整神经形态导航系统：尚未完成。**MP3D缺联合实验；Gibson还需先修SNN波前。
