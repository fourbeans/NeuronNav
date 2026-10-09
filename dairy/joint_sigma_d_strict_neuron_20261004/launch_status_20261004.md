# 联合评测启动记录（2026-10-04 07:18 CST）

## 实施范围

新增两个入口：

- `semexp/eval_poni_sigma_d_strict_neuron.py`：Gibson 原语义分割与在线建图 → Sigma-D U-Net 势函数 → 修复 no-path 的严格 SNN 波前 → 原控制器。
- `hlab/eval_poni_sigma_d_strict_neuron.py`：MP3D 原 RedNet 与在线建图 → `PFExpSigmaD` → 严格加权 SNN 波前 → 原控制器。

两者仅在各自评测进程内组合已有模块。启动之前检查了目标路径不存在；没有修改现有原版 PONI、独立 Sigma-D、独立严格 SNN 入口或检查点。源文件与检查点哈希在 `artifact_sha256.txt`。

## 目的与固定协议

MP3D 首先测试单独验证过的 Sigma-D U-Net 与严格加权波前能否在同一 2195-episode 评测中维持导航效果。使用先前独立 Sigma-D MP3D 最佳检查点 `mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`，导航 seed 100、val_part_0 至 val_part_10、`PFExpSigmaD`、`smart_local_boundaries=True`、`num_local_steps=1`、`pf_masking_opt=unexplored`。

Gibson 测试已完成 no-path 投影修复的严格波前与 Sigma-D U-Net 的联合表现。使用先前独立 Sigma-D Gibson seed234 best@32k 检查点 `gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt`；Gibson val v1.1、导航 seed 100、五场景共 1000 episodes；地图、语义分割和潜势掩码参数与既有 Sigma-D Gibson 评测相同。no-path 处理仍采用起点可达域脉冲波、目标几何投影与安全代理点的修复版。

## 实际启动与产物

在 `/mnt/jfzn/shensj/PONI` 执行了 `README.md` 中的两个 `nohup bash` 命令。启动器 PID：Gibson `3991621`、MP3D `3991622`。两个 run 目录在启动时新建且写入保护为“存在即拒绝覆盖”。

```text
dairy/joint_sigma_d_strict_neuron_20261004/gibson_seed234_best32k/
dairy/joint_sigma_d_strict_neuron_20261004/mp3d_seed123_best2k/
```

Gibson 完成后读 `gibson_objectnav/dump/seed_100/final_eval_stats.json`；MP3D 各 part 完成后读 `tb_seed_100_val_part_N/stats.json`，启动器会在 11 个 part 均成功时自动调用既有汇总脚本，生成 `summary_2195.json`。逐规划记录在各自 `audit/*.jsonl` 与 `audit_val_part_N/*.jsonl`；进程退出码在 MP3D `exit_codes.tsv`。

Gibson 在 GPU0；MP3D 的 11 个 part 分配为 `0→1, 1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→1, 8→2, 9→3, 10→7`。准确 PID 及分配见 `mp3d_seed123_best2k/processes.tsv`。

## 启动验证

- 新脚本通过 Bash 语法解析，两个 Python 入口通过 AST 语法解析。
- 两个入口的 `--help` 冒烟测试通过；MP3D 入口额外检查策略注册和 `GlobalAgent` 已绑定严格规划器。
- 11 个 MP3D part 与 Gibson 入口日志均打印相应联合模式标记。
- 07:18 前，Gibson 五份审计累计 1480 次 `plan`，其中 46 次触发 no-path 投影修复；MP3D 11 份审计累计 9360 次 `plan`。全部 `used=neuron`、`fallback=false`、`reference=null`、无错误字段。
- 各进程已进入实际导航；日志没有 Traceback、OOM 或数据缺失。此时的中途 Success/SPL 不作为结果。
- GPU6 与同机其他作业共用，启动后曾短时接近 79 GB，随后回落并稳定在约 42–54 GB；已确认我们的 part5 持续写审计，未进行重启。后续若出现 OOM，应只处理本次 MP3D run，并保留现有目录为 aborted，不能混合不同配置的结果。

## 完成后必须核对

1. Gibson `final_eval_stats.json` 含 1000 个 episode，五场景 cohort 与先前 Sigma-D Gibson seed234 基线一致。
2. MP3D `summary_2195.json` 含 11 个 part、2195 个互不重复的 episode ID，且与独立 Sigma-D 和严格 SNN 的 episode ID 集合相等。
3. 全量审计要求 `used_reference=0`、`fallback=0`、`error=0`；Gibson 另报告 no-path 修复次数与涉及 episode 数。
4. 报告联合与两个单模块基线的 Success、SPL、DTS / distance-to-goal，并明确两种改动的交互效果。Gibson no-path 单独修复在 1000 episodes 上只带来 Success +0.9 pp、SPL 近乎不变，因此不能从修复本身推断联合效果会优于原 FMM。
