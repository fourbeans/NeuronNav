#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo 'usage: run_mp3d_part_opt.sh MODE LOOKAHEAD_CM PART_ID GPU EPISODES RUN_ROOT' >&2
  exit 2
fi
mode=$1
lookahead_cm=$2
part_id=$3
gpu=$4
episodes=$5
run_root=$6
poni_root=/mnt/jfzn/shensj/PONI
checkpoint="$poni_root/dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt"
val_part="val_part_${part_id}"
test -f "$checkpoint"
mkdir -p "$run_root/audit_${val_part}"

source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh
conda activate poni
export PYTHONPATH="$poni_root:$poni_root/hlab:$poni_root/dependencies/astar_pycpp:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet GLOG_minloglevel=2 HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

capture_limit=0
if [[ $episodes -gt 0 ]]; then capture_limit=30; export PONI_WAVEOPT_TRACE=1; fi
cd "$poni_root/hlab"
command=(python -u eval_poni_sigma_d_opt_neuron.py
  --neuron-mode neuron
  --neuron-log-dir "$run_root/audit_${val_part}"
  --neuron-capture-limit "$capture_limit"
  --neuron-capture-every 20
  --waveopt-readout "$mode"
  --waveopt-lookahead-cm "$lookahead_cm"
  --exp-config transfer_configs/transfer_objectnav_mp3d.yaml
  TASK_CONFIG.DATASET.DATA_PATH '../data/datasets/objectnav/mp3d/v1/val_parts/{split}/{split}.json.gz'
  EVAL.SPLIT "$val_part"
  TASK_CONFIG.SEED 100
  TENSORBOARD_DIR "$run_root/tb_seed_100_${val_part}"
  LOG_FILE "$run_root/logs_seed_100_${val_part}.txt"
  GLOBAL_AGENT.name PFExpSigmaD
  GLOBAL_AGENT.smart_local_boundaries True
  GLOBAL_AGENT.num_local_steps 1
  PF_EXP_POLICY.pf_model_path "$checkpoint"
  PF_EXP_POLICY.pf_masking_opt unexplored)
if [[ $episodes -gt 0 ]]; then
  command+=(TEST_EPISODE_COUNT "$episodes")
fi
CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$run_root/driver_${val_part}.log" 2>&1
