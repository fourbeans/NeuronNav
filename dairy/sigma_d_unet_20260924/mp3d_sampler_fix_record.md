# MP3D K=4 路径采样修复记录

## 现象与根因

首次两条 MP3D 2k gate 均在 `B6ByNegPMKs_0`、epoch 0、index 536 中止，
traceback 为 `shortest path length 2 < 4`。模型 forward/backward 没有报错。根因是原
PONI 路径采样器可以随机抽到极近终点，而时序数据必须有至少 K 个严格递增
前缀。

## 修改

只修改新文件 `poni/dataset_sigma_d.py`，原 `poni/dataset.py` 不变：

1. 从随机起点计算一次 FMM 可达距离；
2. 从较远的 50% 可达网格中随机抽终点；
3. 仍调用原 `plan_path` 产生真实 shortest path；
4. 只接受长度至少 K 且裁剪后逐帧严格增长的序列；
5. 如 scene-floor 不可用，按 epoch 的确定性排列换下一楼层，不补帧、不重复静态地图；
6. metadata 记录原请求 index、实际 source index 与 fallback hops。

## 回归验证

对原失败 index 536 直接取样：

```text
frames = (4,23,480,480)
scene_floor = jh4fc5c5qoQ_0
scene_fallback_hops = 1
path_length = 37
changed_elements = [2868,4070,3758,2674]
```

每个时间步都有新的可见语义元素，满足 K=4 严格增长契约。首次失败的
run 保留 `driver.log` 用于审计，无 checkpoint；修复后以 `_rerun1` 新目录重启，
没有覆盖失败记录。

`_rerun1` 越过原 index 后，还暴露出原 `plan_path()` 偶尔到达 10,000 步
保护上限的病态轨迹。严格增长检查最终会拒绝它，但会浪费计算。
`_rerun2` 增加了返回后长度拒绝，但这仍无法阻止原 helper 先完成循环和打印。
最终版使用 Sigma-D 数据层内部的无重复节点、有界 FMM 回溯：一旦 STG
重访网格就立即拒绝并换终点。在两次运行都未到首个 checkpoint 时停止，
以不覆盖的 `_rerun3` 作为正式 MP3D 2k gate。
