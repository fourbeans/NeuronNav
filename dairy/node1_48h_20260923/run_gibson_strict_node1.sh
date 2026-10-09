#!/usr/bin/env bash
set -u

if [ "$#" -lt 4 ]; then
  echo "usage: $0 GPU CHECKPOINT OUTPUT_ROOT SCENE..." >&2
  exit 2
fi
gpu="$1"
checkpoint="$2"
output_root="$3"
shift 3
scenes=("$@")
poni_root=/mnt/jfzn/shensj/PONI

if [ -e "$output_root" ]; then
  echo "refusing existing output: $output_root" >&2
  exit 2
fi
mkdir -p "$output_root"
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root"
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$poni_root/semexp"

status=0
for scene in "${scenes[@]}"; do
  dest="$output_root/$scene"
  mkdir -p "$dest/audit"
  CUDA_VISIBLE_DEVICES="$gpu" python eval_poni_strict_neuron.py \
    --neuron-mode neuron \
    --neuron-log-dir "$dest/audit" \
    --neuron-capture-limit 0 \
    --neuron-scene "$scene" \
    --split val --seed 100 --eval 1 \
    --task_config tasks/objectnav_gibson_box.yaml --version v1.1 \
    --auto_gpu_config 0 --num_processes 1 \
    --num_processes_on_first_gpu 1 --num_processes_per_gpu 1 --sim_gpu_id 0 \
    --pf_model_path "$checkpoint" \
    -d "$dest" --num_eval_episodes 200 \
    --num_local_steps 1 --exp_name seed_100 --global_downscaling 1 \
    --mask_nearest_locations --pf_masking_opt unexplored --seg_interval 3 \
    > "$dest/stdout.log" 2>&1
  code="$?"
  printf '%s\t%s\n' "$scene" "$code" >> "$output_root/exit_codes.tsv"
  if [ "$code" -ne 0 ]; then
    status=1
    break
  fi
done
exit "$status"
