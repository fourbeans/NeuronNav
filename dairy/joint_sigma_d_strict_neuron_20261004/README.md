# Sigma-D + strict SNN wavefront joint evaluations (2026-10-04)

Two independent ObjectNav evaluations use previously selected Sigma-D checkpoints
and the strict SNN wavefront planners. Gibson uses the reachable-component
goal-projection no-path repair in `poni/planner_adapter_strict_neuron.py`.
MP3D retains the existing weighted SNN wavefront path.

## Interface and data flow

- Gibson: RGB-D → original semantic segmentation and online map → Sigma-D
  U-Net potential function → semantic goal map → `SemExpEnvAgentStrictNeuron`
  → `gibson_plan_strict` → action. One Sigma-D policy instance serves five
  vector environments; state reset behavior is unchanged from the independent
  Sigma-D evaluation.
- MP3D: RGB-D → original RedNet semantic segmentation and online map →
  registered `PFExpSigmaD` policy → semantic goal map →
  `PlannerActorStrictNeuron` → `mp3d_plan_strict` → action. The strict runtime
  writes `reference=null` and `fallback=false` for every plan.

The two new entry points compose existing runtime modules inside each process.
They do not change original PONI, Sigma-D, or strict wavefront entry points.

## Fixed checkpoints and protocol

- Gibson: `dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt`, previously evaluated as `gibson_seed234_best32k`; Gibson val v1.1, navigation seed 100, five scenes × 200 episodes, `num_local_steps=1`, `global_downscaling=1`, nearest-location masking and `unexplored` potential masking.
- MP3D: `dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt`, previously evaluated as `mp3d_seed123_best2k_4gpu`; MP3D val parts 0–10, navigation seed 100, expected 2195 episodes, smart local boundaries and one local step.

## Launch and outputs

From `/mnt/jfzn/shensj/PONI`:

```bash
nohup bash experiment_scripts/joint_sigma_d_strict_neuron_20261004/run_gibson_joint.sh \
  > dairy/joint_sigma_d_strict_neuron_20261004/gibson_launcher.log 2>&1 &
nohup bash experiment_scripts/joint_sigma_d_strict_neuron_20261004/launch_mp3d_joint.sh \
  > dairy/joint_sigma_d_strict_neuron_20261004/mp3d_launcher.log 2>&1 &
```

Gibson output: `gibson_seed234_best32k/driver.log`, `audit/*.jsonl`, and
`gibson_objectnav/dump/seed_100/final_eval_stats.json`.

MP3D output: `mp3d_seed123_best2k/driver_val_part_N.log`,
`logs_seed_100_val_part_N.txt`, `tb_seed_100_val_part_N/stats.json`,
`audit_val_part_N/*.jsonl`, `exit_codes.tsv`, and `summary_2195.json`.

Each launcher refuses to overwrite its run directory. GPU 0 hosts Gibson;
MP3D parts are distributed over GPUs 1–7. Results must be compared with the
same-checkpoint single-component evaluations only after episode IDs and counts
are checked.
