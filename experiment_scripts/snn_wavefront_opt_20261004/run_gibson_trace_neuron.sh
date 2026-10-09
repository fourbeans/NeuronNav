#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo 'usage: run_gibson_trace_neuron.sh GPU EPISODES RUN_LABEL' >&2
  exit 2
fi
gpu=$1
episodes=$2
label=$3
poni_root=/mnt/jfzn/shensj/PONI
base="$poni_root/dairy/snn_wavefront_opt_20261004"
run_root="$base/$label"
checkpoint="$poni_root/dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt"
[[ "$episodes" =~ ^[0-9]+$ ]]
[[ "$label" =~ ^[a-z0-9_-]+$ ]]
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite: $run_root" >&2
  exit 3
fi
test -f "$checkpoint"
mkdir -p "$run_root/audit"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root:${PYTHONPATH:-}"
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$poni_root/semexp"
CUDA_VISIBLE_DEVICES="$gpu" python -u eval_poni_sigma_d_trace_neuron.py \
  --neuron-mode neuron \
  --neuron-log-dir "$run_root/audit" \
  --neuron-capture-limit 30 \
  --neuron-capture-every 20 \
  --split val --seed 100 --eval 1 \
  --pf_model_path "$checkpoint" \
  -d "$run_root/gibson_objectnav" \
  --num_eval_episodes "$episodes" \
  --num_local_steps 1 \
  --exp_name seed_100 \
  --global_downscaling 1 \
  --mask_nearest_locations \
  --pf_masking_opt unexplored \
  > "$run_root/driver.log" 2>&1
