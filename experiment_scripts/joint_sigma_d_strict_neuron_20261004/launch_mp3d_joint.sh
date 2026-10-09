#!/usr/bin/env bash
set -euo pipefail

poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/joint_sigma_d_strict_neuron_20261004/mp3d_seed123_best2k"
checkpoint="$poni_root/dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt"
part_runner="$poni_root/experiment_scripts/joint_sigma_d_strict_neuron_20261004/run_mp3d_part_joint.sh"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
test -f "$checkpoint"
mkdir -p "$run_root"

# Gibson uses GPU 0.  Spread eleven MP3D val parts over the other seven GPUs.
gpus=(1 2 3 4 5 6 7 1 2 3 7)
pids=()
for part_id in {0..10}; do
  gpu=${gpus[$part_id]}
  bash "$part_runner" "$part_id" "$gpu" "$run_root" "$checkpoint" &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$part_id" "$gpu" "$!" >> "$run_root/processes.tsv"
done

status=0
for part_id in {0..10}; do
  if wait "${pids[$part_id]}"; then
    code=0
  else
    code=$?
    status=1
  fi
  printf '%s\t%s\n' "$part_id" "$code" >> "$run_root/exit_codes.tsv"
done

if [[ $status -eq 0 ]]; then
  source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
  conda activate poni
  python "$poni_root/scripts/summarize_mp3d_navigation_sigma_d.py" \
    --run-root "$run_root" \
    --output "$run_root/summary_2195.json" \
    --expected-parts 11 --expected-episodes 2195 \
    --poll-seconds 1 --timeout-hours 0.01 \
    --model-family sigma-d-strict-snn \
    > "$run_root/summary.log" 2>&1
fi
exit "$status"
