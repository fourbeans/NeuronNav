#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo 'usage: launch_mp3d_refine_neuron.sh CAP_CM COST_GUARD EPISODES_PER_PART GPU_PROFILE RUN_LABEL [ANGLE_DEG]' >&2
  exit 2
fi
cap_cm=$1
cost_guard=$2
episodes=$3
gpu_profile=$4
label=$5
angle_deg=${6:-none}
poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/snn_wavefront_refine_20261004/$label"
runner="$poni_root/experiment_scripts/snn_wavefront_refine_20261004/run_mp3d_refine_part_neuron.sh"
[[ "$cost_guard" == 0 || "$cost_guard" == 1 ]]
[[ "$episodes" =~ ^[0-9]+$ ]]
[[ "$label" =~ ^[a-z0-9_-]+$ ]]
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite: $run_root" >&2
  exit 3
fi
if [[ "$gpu_profile" == pilot_a ]]; then
  gpus=(0 1 2 3 0 1 2 3 0 1 2)
elif [[ "$gpu_profile" == pilot_b ]]; then
  gpus=(4 5 6 7 4 5 6 7 4 5 6)
elif [[ "$gpu_profile" == full ]]; then
  gpus=(0 1 2 3 4 5 6 7 0 1 2)
else
  echo "invalid GPU profile: $gpu_profile" >&2
  exit 2
fi
mkdir -p "$run_root"
printf 'cap_cm=%s\ncost_guard=%s\nepisodes_per_part=%s\ngpu_profile=%s\nangle_deg=%s\n' \
  "$cap_cm" "$cost_guard" "$episodes" "$gpu_profile" "$angle_deg" > "$run_root/config.txt"
pids=()
for part_id in {0..10}; do
  bash "$runner" "$cap_cm" "$cost_guard" "$part_id" "${gpus[$part_id]}" "$episodes" "$run_root" "$angle_deg" &
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
