#!/usr/bin/env bash
set -euo pipefail

cd /mnt/jfzn/shensj/PONI
nav_root=dairy/sigma_d_unet_20260924/navigation
run_name=gibson_seed123_full_best
checkpoint=dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed123/checkpoints/best.ckpt

mkdir -p "$nav_root"
if [[ -e "$nav_root/$run_name" ]]; then
  echo "refusing to overwrite $nav_root/$run_name" >&2
  exit 3
fi

nohup env CUDA_VISIBLE_DEVICES=0 \
  bash experiment_scripts/gibson/eval_poni_sigma_d.sh \
  "$checkpoint" "$run_name" \
  > "$nav_root/${run_name}.launch.log" 2>&1 &

pid=$!
echo "$pid" > "$nav_root/${run_name}.pid"
echo "gibson_navigation_pid=$pid"
