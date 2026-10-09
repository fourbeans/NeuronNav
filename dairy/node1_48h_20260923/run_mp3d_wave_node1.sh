#!/usr/bin/env bash
set -u

if [ "$#" -ne 4 ]; then
  echo "usage: $0 MODE GPU CHECKPOINT OUTPUT_ROOT" >&2
  exit 2
fi

mode="$1"
gpu="$2"
checkpoint="$3"
output_root="$4"
poni_root=/mnt/jfzn/shensj/PONI

if [ "$mode" != reference ] && [ "$mode" != neuron ]; then
  echo "MODE must be reference or neuron" >&2
  exit 2
fi
if [ -e "$output_root" ]; then
  echo "refusing existing output: $output_root" >&2
  exit 2
fi

mkdir -p "$output_root"
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root:$poni_root/dependencies/astar_pycpp"
export MAGNUM_LOG=quiet
export GLOG_minloglevel=2
export HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$poni_root/hlab"
pids=()
parts=()
for part_id in 0 1 2 3 4 5 6 7 8 9 10; do
  val_part="val_part_${part_id}"
  part_root="$output_root/$val_part"
  mkdir -p "$part_root/audit"
  CUDA_VISIBLE_DEVICES="$gpu" python eval_poni_neuron.py \
    --neuron-mode "$mode" \
    --neuron-log-dir "$part_root/audit" \
    --neuron-capture-limit 0 \
    --exp-config transfer_configs/transfer_objectnav_mp3d.yaml \
    TASK_CONFIG.DATASET.DATA_PATH "../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz" \
    EVAL.SPLIT "$val_part" \
    TASK_CONFIG.SEED 100 \
    TENSORBOARD_DIR "$part_root/tb" \
    LOG_FILE "$part_root/eval.log" \
    GLOBAL_AGENT.name PFExp \
    GLOBAL_AGENT.smart_local_boundaries True \
    GLOBAL_AGENT.num_local_steps 1 \
    PF_EXP_POLICY.pf_model_path "$checkpoint" \
    PF_EXP_POLICY.pf_masking_opt unexplored \
    > "$part_root/stdout.log" 2>&1 &
  pids+=("$!")
  parts+=("$val_part")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    printf '%s\t0\n' "${parts[$i]}" >> "$output_root/exit_codes.tsv"
  else
    code="$?"
    printf '%s\t%s\n' "${parts[$i]}" "$code" >> "$output_root/exit_codes.tsv"
    status=1
  fi
done
exit "$status"
