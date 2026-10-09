#!/usr/bin/env bash
set -euo pipefail

poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/joint_sigma_d_strict_neuron_20261004/gibson_seed234_best32k"
checkpoint="$poni_root/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
test -f "$checkpoint"
mkdir -p "$run_root/audit"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PYTHONPATH="$poni_root:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$poni_root/semexp"
CUDA_VISIBLE_DEVICES=0 python -u eval_poni_sigma_d_strict_neuron.py \
  --neuron-mode neuron \
  --neuron-log-dir "$run_root/audit" \
  --neuron-capture-limit 0 \
  --split val --seed 100 --eval 1 \
  --pf_model_path "$checkpoint" \
  -d "$run_root/gibson_objectnav" \
  --num_local_steps 1 \
  --exp_name seed_100 \
  --global_downscaling 1 \
  --mask_nearest_locations \
  --pf_masking_opt unexplored \
  > "$run_root/driver.log" 2>&1
