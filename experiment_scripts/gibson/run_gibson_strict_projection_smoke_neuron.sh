#!/usr/bin/env bash
set -euo pipefail

poni_root=/mnt/jfzn/shensj/PONI
checkpoint="$poni_root/pretrained_models/gibson_models/poni_seed_123.ckpt"
run_root="$poni_root/dairy/gibson_wavefront_nopath_fix_20261003/smoke_20eps"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
mkdir -p "$run_root/audit"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root"
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$poni_root/semexp"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" python -u eval_poni_strict_neuron.py \
  --neuron-mode neuron \
  --neuron-log-dir "$run_root/audit" \
  --neuron-capture-limit 0 \
  --neuron-scene Collierville \
  --split val --seed 100 --eval 1 \
  --task_config tasks/objectnav_gibson_box.yaml --version v1.1 \
  --auto_gpu_config 0 --num_processes 1 \
  --num_processes_on_first_gpu 1 --num_processes_per_gpu 1 --sim_gpu_id 0 \
  --pf_model_path "$checkpoint" \
  -d "$run_root" --num_eval_episodes 20 \
  --num_local_steps 1 --exp_name seed_100 --global_downscaling 1 \
  --mask_nearest_locations --pf_masking_opt unexplored --seg_interval 3 \
  > "$run_root/stdout.log" 2>&1
