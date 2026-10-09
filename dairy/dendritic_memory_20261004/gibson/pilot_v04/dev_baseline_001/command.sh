#!/bin/sh
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.collect_dend --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04 --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04/dev_baseline_001 --actor-id 1 --actors 10 --steps 5100 --role dev --seed 124 --episodes 10 --baseline
