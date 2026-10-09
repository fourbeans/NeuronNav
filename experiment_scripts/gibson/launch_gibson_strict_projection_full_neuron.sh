#!/usr/bin/env bash
set -euo pipefail

poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/gibson_wavefront_nopath_fix_20261003/full_1000"
scene_runner="$poni_root/experiment_scripts/gibson/eval_poni_strict_projection_scene_neuron.sh"
checkpoint="$poni_root/pretrained_models/gibson_models/poni_seed_123.ckpt"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

scenes=(Collierville Corozal Darden Markleeville Wiconisco)
gpus=(1 2 3 4 5)
pids=()
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  gpu=${gpus[$index]}
  bash "$scene_runner" "$gpu" "$scene" 200 "$run_root/$scene" "$checkpoint" \
    > "$run_root/${scene}_launcher.log" 2>&1 &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$scene" "$gpu" "$!" >> "$run_root/processes.tsv"
done

status=0
for index in "${!pids[@]}"; do
  scene=${scenes[$index]}
  if wait "${pids[$index]}"; then
    code=0
  else
    code=$?
    status=1
  fi
  printf '%s\t%s\n' "$scene" "$code" >> "$run_root/exit_codes.tsv"
done
exit "$status"
