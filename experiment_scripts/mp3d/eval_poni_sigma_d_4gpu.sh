#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 SIGMA_D_CHECKPOINT GPU_0 GPU_1 GPU_2 GPU_3 [RUN_NAME]" >&2
  exit 2
fi

poni_root=/mnt/jfzn/shensj/PONI
checkpoint=$(realpath "$1")
devices=("$2" "$3" "$4" "$5")
run_name=${6:-mp3d_sigma_d_eval_4gpu}
run_root="$poni_root/dairy/sigma_d_unet_20260924/navigation/$run_name"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="${PYTHONPATH:-}:$poni_root:$poni_root/hlab:$poni_root/dependencies/astar_pycpp"
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet

{
  echo "checkpoint=$checkpoint"
  echo "devices=${devices[*]}"
  echo "seed=100"
  echo "parts=val_part_0..val_part_10"
  echo "planner=original_weighted_astar"
  echo "policy=PFExpSigmaD"
} > "$run_root/run_manifest.txt"

cd "$poni_root/hlab"
pids=()
for part_id in $(seq 0 10); do
  val_part="val_part_${part_id}"
  slot=$((part_id % 4))
  device=${devices[$slot]}
  CUDA_VISIBLE_DEVICES="$device" python -u eval_poni_sigma_d.py \
    --exp-config transfer_configs/transfer_objectnav_mp3d.yaml \
    TASK_CONFIG.DATASET.DATA_PATH "../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz" \
    EVAL.SPLIT "$val_part" \
    TASK_CONFIG.SEED 100 \
    TENSORBOARD_DIR "$run_root/tb_seed_100_${val_part}" \
    LOG_FILE "$run_root/logs_seed_100_${val_part}.txt" \
    GLOBAL_AGENT.name PFExpSigmaD \
    GLOBAL_AGENT.smart_local_boundaries True \
    GLOBAL_AGENT.num_local_steps 1 \
    PF_EXP_POLICY.pf_model_path "$checkpoint" \
    PF_EXP_POLICY.pf_masking_opt unexplored \
    > "$run_root/driver_${val_part}.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "$val_part gpu=$device pid=$pid" >> "$run_root/process_manifest.txt"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "one or more MP3D validation parts failed" >&2
  exit 1
fi
