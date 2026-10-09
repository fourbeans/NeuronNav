#!/bin/sh
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.collect_dend --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04 --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/official_val_v04/val_baseline_002 --actor-id 2 --actors 20 --steps 13000 --role val --seed 125 --episodes 25 --baseline
