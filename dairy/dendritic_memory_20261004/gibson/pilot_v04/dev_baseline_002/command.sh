#!/bin/sh
/mnt/jfzn/miniconda3/envs/poni/bin/python -m dendritic_nav.runners.collect_dend --experiment /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04 --output /mnt/jfzn/shensj/PONI/dairy/dendritic_memory_20261004/gibson/pilot_v04/dev_baseline_002 --actor-id 2 --actors 10 --steps 5100 --role dev --seed 125 --episodes 10 --baseline
