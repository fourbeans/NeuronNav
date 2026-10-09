#!/bin/sh
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.collect_dend --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v01 --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v01/dev_baseline_005 --actor-id 5 --actors 20 --steps 2550 --role dev --seed 128 --episodes 5 --baseline
