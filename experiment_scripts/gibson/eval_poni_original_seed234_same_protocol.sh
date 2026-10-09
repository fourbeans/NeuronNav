#!/usr/bin/env bash
set -euo pipefail

poni_root=/mnt/jfzn/shensj/PONI
checkpoint="$poni_root/pretrained_models/gibson_models/poni_seed_234.ckpt"
run_name=gibson_original_seed234_same_protocol
run_root="$poni_root/dairy/sigma_d_unet_20260924/navigation/$run_name"
if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run: $run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export GLOG_minloglevel=2 MAGNUM_LOG=quiet
export PYTHONPATH="${PYTHONPATH:-}:$poni_root"

cd "$poni_root/semexp"
python -u eval_poni.py \
  --split val \
  --seed 100 \
  --eval 1 \
  --pf_model_path "$checkpoint" \
  -d "$run_root/gibson_objectnav" \
  --num_local_steps 1 \
  --exp_name seed_100 \
  --global_downscaling 1 \
  --mask_nearest_locations \
  --pf_masking_opt unexplored \
  2>&1 | tee "$run_root/driver.log"
