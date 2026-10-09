#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo 'usage: launch_mp3d_trace_neuron.sh EPISODES_PER_PART RUN_LABEL' >&2
  exit 2
fi
episodes=$1
label=$2
poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/snn_wavefront_opt_20261004/$label"
runner="$poni_root/experiment_scripts/snn_wavefront_opt_20261004/run_mp3d_part_trace_neuron.sh"
[[ "$episodes" =~ ^[0-9]+$ ]]
[[ "$label" =~ ^[a-z0-9_-]+$ ]]
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite: $run_root" >&2
  exit 3
fi
gpus=(1 2 3 4 5 6 7 1 2 3 4)
mkdir -p "$run_root"
pids=()
for part_id in {0..10}; do
  bash "$runner" "$part_id" "${gpus[$part_id]}" "$episodes" "$run_root" &
  pids+=("$!")
  printf '%s\t%s\t%s\n' "$part_id" "${gpus[$part_id]}" "$!" >> "$run_root/processes.tsv"
done
status=0
for part_id in {0..10}; do
  if wait "${pids[$part_id]}"; then code=0; else code=$?; status=1; fi
  printf '%s\t%s\n' "$part_id" "$code" >> "$run_root/exit_codes.tsv"
done
exit "$status"
