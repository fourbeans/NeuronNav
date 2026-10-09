#!/usr/bin/env bash
set -u

if [ "$#" -lt 4 ]; then
  echo "usage: $0 GPU CHECKPOINT OUTPUT_ROOT PART_ID..." >&2
  exit 2
fi
gpu="$1"
checkpoint="$2"
output_root="$3"
shift 3
parts=("$@")
poni_root=/mnt/jfzn/shensj/PONI

if [ -e "$output_root" ]; then
  echo "refusing existing output: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root:$poni_root/dependencies/astar_pycpp"
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$poni_root/hlab"

status=0
for part_id in "${parts[@]}"; do
  val_part="val_part_${part_id}"
  dest="$output_root/$val_part"
  mkdir -p "$dest/audit"
  CUDA_VISIBLE_DEVICES="$gpu" python eval_poni_strict_neuron.py \
    --neuron-mode neuron \
    --neuron-log-dir "$dest/audit" \
    --neuron-capture-limit 0 \
    --exp-config transfer_configs/transfer_objectnav_mp3d.yaml \
    TASK_CONFIG.DATASET.DATA_PATH "../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz" \
    EVAL.SPLIT "$val_part" \
    TASK_CONFIG.SEED 100 \
    TENSORBOARD_DIR "$dest/tb" \
    LOG_FILE "$dest/eval.log" \
    GLOBAL_AGENT.name PFExp \
    GLOBAL_AGENT.smart_local_boundaries True \
    GLOBAL_AGENT.num_local_steps 1 \
    PF_EXP_POLICY.pf_model_path "$checkpoint" \
    PF_EXP_POLICY.pf_masking_opt unexplored \
    > "$dest/stdout.log" 2>&1
  code="$?"
  printf '%s\t%s\n' "$val_part" "$code" >> "$output_root/exit_codes.tsv"
  if [ "$code" -ne 0 ]; then
    status=1
    break
  fi
done
exit "$status"
