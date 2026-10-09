#!/usr/bin/env bash
set -euo pipefail

cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni

run_dir=dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite $run_dir" >&2
  exit 3
fi
mkdir -p "$run_dir"

nohup env CUDA_VISIBLE_DEVICES=2,3 python -u -m torch.distributed.launch \
  --use_env --nproc_per_node=2 --master_port=29502 \
  train_sigma_d.py \
  --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
  LOGGING.output_dir "$run_dir" \
  LOGGING.auto_resume True \
  SEED 123 \
  DATASET.seed 123 \
  MODEL.pretrained_path pretrained_models/mp3d_models/poni_seed_123.ckpt \
  MODEL.sigma_d.qmax 4095 \
  OPTIM.lr 0.00001 \
  OPTIM.num_total_updates 40000 \
  DATASET.train_samples_per_epoch 20000 \
  DATASET.val_samples_per_epoch 1000 \
  OPTIM.batch_size 1 \
  OPTIM.grad_accum_steps 4 \
  LOGGING.log_interval 20 \
  LOGGING.eval_interval 1000 \
  LOGGING.ckpt_interval 1000 \
  > "$run_dir/driver.log" 2>&1 &

echo "mp3d_full_seed123_pid=$!"
