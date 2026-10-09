#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PART_ID GPU RUN_ROOT SIGMA_D_CHECKPOINT" >&2
  exit 2
fi
part_id=$1
gpu=$2
run_root=$(realpath -m "$3")
checkpoint=$(realpath "$4")
poni_root=/mnt/jfzn/shensj/PONI
val_part="val_part_${part_id}"
mkdir -p "$run_root/audit_${val_part}"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root:$poni_root/hlab:$poni_root/dependencies/astar_pycpp:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$poni_root/hlab"
CUDA_VISIBLE_DEVICES="$gpu" python -u eval_poni_sigma_d_strict_neuron.py \
  --neuron-mode neuron \
  --neuron-log-dir "$run_root/audit_${val_part}" \
  --neuron-capture-limit 0 \
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
  > "$run_root/driver_${val_part}.log" 2>&1
