#!/usr/bin/env bash
set -euo pipefail

cd /mnt/jfzn/shensj/PONI
source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni

root_dir=dairy/sigma_d_unet_20260924/experiments

mp3d_lr1e5_dir="$root_dir/mp3d_2k_q4095_lr1e5_rerun1"
mkdir -p "$mp3d_lr1e5_dir"
nohup env CUDA_VISIBLE_DEVICES=2 python -u train_sigma_d.py \
  --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
  LOGGING.output_dir "$mp3d_lr1e5_dir" \
  LOGGING.auto_resume True \
  MODEL.sigma_d.qmax 4095 \
  OPTIM.lr 0.00001 \
  OPTIM.num_total_updates 2000 \
  DATASET.train_samples_per_epoch 2000 \
  DATASET.val_samples_per_epoch 128 \
  OPTIM.batch_size 1 \
  OPTIM.grad_accum_steps 1 \
  LOGGING.log_interval 20 \
  LOGGING.eval_interval 500 \
  LOGGING.ckpt_interval 500 \
  > "$mp3d_lr1e5_dir/driver.log" 2>&1 &
echo "mp3d_lr1e5_pid=$!"

mp3d_lr1e4_dir="$root_dir/mp3d_2k_q4095_lr1e4_rerun1"
mkdir -p "$mp3d_lr1e4_dir"
nohup env CUDA_VISIBLE_DEVICES=3 python -u train_sigma_d.py \
  --exp-config configs/sigma_d/mp3d_sigma_d.yaml \
  LOGGING.output_dir "$mp3d_lr1e4_dir" \
  LOGGING.auto_resume True \
  MODEL.sigma_d.qmax 4095 \
  OPTIM.lr 0.0001 \
  OPTIM.num_total_updates 2000 \
  DATASET.train_samples_per_epoch 2000 \
  DATASET.val_samples_per_epoch 128 \
  OPTIM.batch_size 1 \
  OPTIM.grad_accum_steps 1 \
  LOGGING.log_interval 20 \
  LOGGING.eval_interval 500 \
  LOGGING.ckpt_interval 500 \
  > "$mp3d_lr1e4_dir/driver.log" 2>&1 &
echo "mp3d_lr1e4_pid=$!"

gibson_seed234_dir="$root_dir/gibson_full_q4095_lr1e5_seed234"
mkdir -p "$gibson_seed234_dir"
nohup env CUDA_VISIBLE_DEVICES=6,7 python -u -m torch.distributed.launch \
  --use_env --nproc_per_node=2 --master_port=29501 \
  train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d.yaml \
  LOGGING.output_dir "$gibson_seed234_dir" \
  LOGGING.auto_resume True \
  SEED 234 \
  DATASET.seed 234 \
  MODEL.pretrained_path pretrained_models/gibson_models/poni_seed_234.ckpt \
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
  > "$gibson_seed234_dir/driver.log" 2>&1 &
echo "gibson_seed234_pid=$!"
