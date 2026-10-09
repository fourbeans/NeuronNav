# Gibson strict-SNN no-path repair and seed234 baseline (2026-10-03)

## Purpose

1. Evaluate the original Gibson PONI `poni_seed_234.ckpt` with exactly the
   current Sigma-D navigation protocol (`seed=100`, Gibson val v1.1, five
   workers, 200 episodes per worker, 1000 total, identical masking flags).
2. Replace the Gibson strict-SNN planner's constant-distance no-path behavior
   with an event-wave reachable-goal projection. MP3D code and behavior are
   explicitly outside this change.

## Algorithm

For ordinary queries, the existing reverse first-spike wave and FMM ring
decoder are unchanged. If the requested goal is disconnected from the start:

1. A start-seeded wave produces the reachable component
   `R = {x | t_start(x) < infinity}`.
2. A goal-seeded wave on an obstacle-free lattice propagates until the first
   eligible reachable cell spikes; it does not fill the remaining map.
3. The proxy is `g' = argmin_{x in R, t_start(x)>=5} d_g(x)`. Positive delays
   and deterministic heap ordering make early termination identical to the
   full-field argmin, including ties. The five-cell constraint is relaxed only
   for tiny components.
4. A reverse wave inside the real free map plans from `g'` to the start. The
   fifth predecessor-chain cell is the short-term goal. `stop` is forced false
   because `g'` is a temporary waypoint, not semantic task completion.

The normal path pays no additional propagation. Repaired calls record
`no_path_detected=true`, `no_path_repaired=true`, `projection_used=true`, and
separate component/projection/repair spike counts. `no_path=false` means the
planner returned a valid actionable path rather than an uninformative field.

## Commands

Original PONI seed234, exact current protocol:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash \
  /mnt/jfzn/shensj/PONI/experiment_scripts/gibson/eval_poni_original_seed234_same_protocol.sh \
  > /mnt/jfzn/shensj/PONI/dairy/sigma_d_unet_20260924/navigation/gibson_original_seed234_same_protocol_launch.log 2>&1 &
```

Focused unit/regression validation:

```bash
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
cd /mnt/jfzn/shensj/PONI
PYTHONPATH=. python scripts/validate_gibson_nopath_neuron.py \
  --output dairy/gibson_wavefront_nopath_fix_20261003/unit_validation.json
```

Twenty-episode real-environment smoke test:

```bash
CUDA_VISIBLE_DEVICES=2 nohup bash \
  /mnt/jfzn/shensj/PONI/experiment_scripts/gibson/run_gibson_strict_projection_smoke_neuron.sh \
  > /mnt/jfzn/shensj/PONI/dairy/gibson_wavefront_nopath_fix_20261003/smoke_launch.log 2>&1 &
```

Full repaired strict-SNN evaluation (five scenes in parallel on GPUs 1--5):

```bash
nohup bash \
  /mnt/jfzn/shensj/PONI/experiment_scripts/gibson/launch_gibson_strict_projection_full_neuron.sh \
  > /mnt/jfzn/shensj/PONI/dairy/gibson_wavefront_nopath_fix_20261003/full_1000_launch.log 2>&1 &
```

## Outputs

- Baseline metrics: `dairy/sigma_d_unet_20260924/navigation/gibson_original_seed234_same_protocol/gibson_objectnav/dump/seed_100/final_eval_stats.json`
- Baseline log: `dairy/sigma_d_unet_20260924/navigation/gibson_original_seed234_same_protocol/driver.log`
- Unit report: `dairy/gibson_wavefront_nopath_fix_20261003/unit_validation.json`
- Smoke metrics: `dairy/gibson_wavefront_nopath_fix_20261003/smoke_20eps/dump/seed_100/final_eval_stats.json`
- Smoke audit: `dairy/gibson_wavefront_nopath_fix_20261003/smoke_20eps/audit/*.jsonl`
- Full repaired run: `dairy/gibson_wavefront_nopath_fix_20261003/full_1000/<scene>/`

Checksums and the exact pre-change source are retained in this directory so
that MP3D non-regression and provenance can be audited.

## Executed validation and current status

Executed on 2026-10-03 (Asia/Shanghai):

- Focused synthetic test passed. A reachable query stayed on the original
  branch; an enclosed target produced a valid 19-cell path to a safe proxy,
  a nontrivial short-term goal, and `stop=false`.
- Reachable-path regression passed on 64/64 randomized maps. STG, stop,
  replan, complete predecessor path, spike count, event count, and arrival
  cost exactly matched the pre-repair implementation.
- The `mp3d_plan_strict` function SHA-256 is unchanged:
  `27374bbe653ae40144b4f885da073c6006a3eed14e97479cd5b4497695938652`.
- The 20-episode Collierville smoke and old strict run have an exactly equal
  `goal_distance` sequence. Old constant-fill strict SNN obtained
  Success/SPL/DTS `0.75/0.518308/0.724421`; repaired projection obtained
  `0.85/0.534329/0.227386`. Three episodes contained disconnections. The old
  run recorded 1147 no-path plans; the repaired run made 413 successful
  projections and zero repaired calls reported STOP. This is a smoke result,
  not the final 1000-episode conclusion.
- A second 20-episode smoke compared full-field projection with early-stop
  projection. All 1758 planning actions and complete navigation metric arrays
  match exactly. Across 413 repairs, mean projection spikes fell from 232324
  to 2666.23 (-98.85%), and mean repair-branch software time fell from
  126.09 ms to 69.34 ms (-45.0%). See
  `projection_optimization_comparison.json`.
- The seed234 original-PONI baseline was machine-compared with the completed
  seed123 same-protocol run. All 85 non-path Namespace fields match. Only the
  intentional checkpoint and output-directory fields differ. See
  `original_seed234_protocol_comparison.json`.
- The seed234 baseline is running at
  `dairy/sigma_d_unet_20260924/navigation/gibson_original_seed234_same_protocol/`.
- The optimized repaired full 1000-episode run is running five scenes
  concurrently at
  `dairy/gibson_wavefront_nopath_fix_20261003/full_1000/`.

Two early baseline launches were stopped during initialization because their
GPU visibility did not reproduce the old run's automatic scheduling fields.
They are retained under `aborted_original_seed234_*` for provenance and must
not be treated as results. The active run has the exact old values:
`num_processes=5`, `num_processes_on_first_gpu=5`, and
`num_processes_per_gpu=0`.

An initial five-scene full run used the already-correct but inefficient
full-field projection. It was stopped after the early-stop equivalence test and
is retained as `aborted_full_1000_full_field_projection/`. The active
`full_1000/` run restarted every scene from episode zero with the optimized
source, so no mixed implementation appears in the formal result.
