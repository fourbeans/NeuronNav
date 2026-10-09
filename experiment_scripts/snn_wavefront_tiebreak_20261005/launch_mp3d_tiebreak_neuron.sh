#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo 'usage: launch_mp3d_tiebreak_neuron.sh EPISODES_PER_PART GPU_PROFILE RUN_LABEL' >&2
  exit 2
fi
episodes=$1
gpu_profile=$2
label=$3
poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/snn_wavefront_tiebreak_20261005/$label"
runner="$poni_root/experiment_scripts/snn_wavefront_tiebreak_20261005/run_mp3d_tiebreak_part_neuron.sh"
[[ "$episodes" =~ ^[0-9]+$ ]]
[[ "$label" =~ ^[a-z0-9_-]+$ ]]
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite: $run_root" >&2
  exit 3
fi
if [[ "$gpu_profile" == pilot || "$gpu_profile" == full ]]; then
  gpus=(0 1 2 3 4 5 6 7 0 1 2)
else
  echo "invalid GPU profile: $gpu_profile" >&2
  exit 2
fi
mkdir -p "$run_root"
printf 'episodes_per_part=%s\ngpu_profile=%s\n' "$episodes" "$gpu_profile" > "$run_root/config.txt"
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
if [[ $status -eq 0 ]]; then
  source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
  conda activate poni
  python "$poni_root/scripts/summarize_waveopt_neuron.py" \
    --dataset mp3d --run-root "$run_root" > "$run_root/summary.log" 2>&1
fi
exit "$status"
