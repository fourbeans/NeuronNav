#!/usr/bin/env bash
# Exact historical commands for sigma-d runs started on 2026-09-24.
# This file is documentation; execute one section at a time, never all at once.

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
cd /mnt/jfzn/shensj/PONI

# S0 contract, CPU and full-FP32 GPU.
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/validate_sigma_d.py \
  --checkpoint pretrained_models/gibson_models/poni_seed_123.ckpt \
  --device cpu \
  --output dairy/sigma_d_unet_20260924/tests/model_contract_gibson_seed123_cpu.json
CUDA_VISIBLE_DEVICES=0 python scripts/validate_sigma_d.py \
  --checkpoint pretrained_models/gibson_models/poni_seed_123.ckpt \
  --device cuda:0 \
  --output dairy/sigma_d_unet_20260924/tests/model_contract_gibson_seed123_gpu_fp32.json

# Single-GPU two-update 480x480 smoke.
CUDA_VISIBLE_DEVICES=0 python train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d_smoke.yaml

# Historical qmax=15 short-fit. This is not the final qmax setting.
CUDA_VISIBLE_DEVICES=0 python train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d.yaml \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/gibson_sigma_d_overfit128 \
  LOGGING.auto_resume False MODEL.sigma_d.qmax 15 \
  OPTIM.num_total_updates 200 DATASET.train_samples_per_epoch 128 \
  DATASET.val_samples_per_epoch 32 OPTIM.grad_accum_steps 1 \
  LOGGING.log_interval 5 LOGGING.eval_interval 50 LOGGING.ckpt_interval 50

# Two-GPU DDP smoke.
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --use_env \
  --nproc_per_node=2 train_sigma_d.py \
  --exp-config configs/sigma_d/gibson_sigma_d_smoke.yaml \
  LOGGING.output_dir dairy/sigma_d_unet_20260924/experiments/gibson_sigma_d_ddp_smoke \
  MODEL.sigma_d.qmax 4095 DATASET.train_samples_per_epoch 4 \
  DATASET.val_samples_per_epoch 4 OPTIM.num_total_updates 2 \
  LOGGING.eval_interval 1 LOGGING.ckpt_interval 1

# Offline comparison template. The recorded sweep substituted theta and qmax.
CUDA_VISIBLE_DEVICES=1 python scripts/compare_poni_sigma_d.py \
  --checkpoint pretrained_models/gibson_models/poni_seed_123.ckpt \
  --semantic-root data/semantic_maps/gibson/semantic_maps \
  --fmm-root data/semantic_maps/gibson/fmm_dists_123 \
  --dataset gibson --samples 32 --sequence-length 4 --workers 4 \
  --device cuda:0 --threshold 0.0078125 --output-threshold 0.00390625 \
  --qmax 4095 \
  --output dairy/sigma_d_unet_20260924/experiments/gibson_qmax_sweep/q4095.json

