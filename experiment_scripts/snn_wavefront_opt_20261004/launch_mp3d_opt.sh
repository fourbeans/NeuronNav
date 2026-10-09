#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo 'usage: launch_mp3d_opt.sh MODE LOOKAHEAD_CM EPISODES_PER_PART GPU_PROFILE RUN_LABEL' >&2
  exit 2
fi
mode=$1
lookahead_cm=$2
episodes=$3
gpu_profile=$4
label=$5
poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/snn_wavefront_opt_20261004/$label"
runner="$poni_root/experiment_scripts/snn_wavefront_opt_20261004/run_mp3d_part_opt.sh"
[[ "$mode" == length || "$mode" == visible ]]
[[ "$episodes" =~ ^[0-9]+$ ]]
[[ "$label" =~ ^[a-z0-9_-]+$ ]]
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite: $run_root" >&2
  exit 3
fi
if [[ "$gpu_profile" == pilot ]]; then
  gpus=(2 3 4 5 6 7 2 3 4 5 6)
elif [[ "$gpu_profile" == full ]]; then
  gpus=(1 2 3 4 5 6 7 1 2 3 7)
else
  echo "invalid GPU profile: $gpu_profile" >&2
  exit 2
fi
mkdir -p "$run_root"
pids=()
for part_id in {0..10}; do
  bash "$runner" "$mode" "$lookahead_cm" "$part_id" "${gpus[$part_id]}" "$episodes" "$run_root" &
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
    --dataset mp3d --run-root "$run_root" \
    > "$run_root/summary.log" 2>&1
fi
exit "$status"
