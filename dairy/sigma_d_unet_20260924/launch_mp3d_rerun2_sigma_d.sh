#!/usr/bin/env bash
set -euo pipefail

cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
root_dir=dairy/sigma_d_unet_20260924/experiments

for lr_tag in 1e5 1e4; do
  if [[ "$lr_tag" == "1e5" ]]; then
    gpu=2
    lr=0.00001
  else
    gpu=3
    lr=0.0001
  fi
  run_dir="$root_dir/mp3d_2k_q4095_lr${lr_tag}_rerun2"
  if [[ -e "$run_dir" ]]; then
    echo "refusing to overwrite $run_dir" >&2
    exit 3
  fi
  mkdir -p "$run_dir"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" python -u train_sigma_d.py \
    --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
    LOGGING.output_dir "$run_dir" \
    LOGGING.auto_resume True \
    MODEL.sigma_d.qmax 4095 \
    OPTIM.lr "$lr" \
    OPTIM.num_total_updates 2000 \
    DATASET.train_samples_per_epoch 2000 \
    DATASET.val_samples_per_epoch 128 \
    OPTIM.batch_size 1 \
    OPTIM.grad_accum_steps 1 \
    LOGGING.log_interval 20 \
    LOGGING.eval_interval 500 \
    LOGGING.ckpt_interval 500 \
    > "$run_dir/driver.log" 2>&1 &
  echo "mp3d_lr${lr_tag}_pid=$!"
done
