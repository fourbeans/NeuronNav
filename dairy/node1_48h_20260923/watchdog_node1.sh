#!/usr/bin/env bash
set -u

poni_root=/mnt/jfzn/shensj/PONI
run_root="$poni_root/dairy/node1_48h_20260923"
python_bin=/mnt/jfzn/miniconda3/envs/poni/bin/python
deadline=$(( $(date +%s) + 49 * 3600 ))
watch_log="$run_root/watchdog.log"
gpu_log="$run_root/gpu_watch.csv"
low_counts=(0 0 0 0 0 0 0 0)
checkpoints=(
  pretrained_models/gibson_models/poni_seed_123.ckpt
  pretrained_models/gibson_models/poni_seed_123.ckpt
  pretrained_models/gibson_models/poni_seed_234.ckpt
  pretrained_models/gibson_models/poni_seed_234.ckpt
  pretrained_models/gibson_models/poni_seed_345.ckpt
  pretrained_models/gibson_models/poni_seed_345.ckpt
  pretrained_models/mp3d_models/poni_seed_123.ckpt
  pretrained_models/mp3d_models/poni_seed_123.ckpt
)

printf 'timestamp,index,memory_used_mib,memory_total_mib,utilization_gpu_percent,temperature_c\n' > "$gpu_log"

launch_sustain() {
  local gpu="$1"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local output="$run_root/sustain_gpu${gpu}_restart_${stamp}"
  local driver="$run_root/sustain_gpu${gpu}_restart_${stamp}_driver.log"
  cd "$poni_root" || return 1
  nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$poni_root" \
    "$python_bin" scripts/poni_gpu_sustain_neuron.py \
    --checkpoint "${checkpoints[$gpu]}" \
    --output "$output" \
    --duration-hours 49 \
    --target-total-memory-mib 50000 \
    --safety-free-mib 16000 \
    --micro-batch 10 \
    --max-resident-batches 160 \
    --physical-gpu-index "$gpu" \
    --seed "$((2000 + gpu))" \
    > "$driver" 2>&1 < /dev/null &
  printf '%s restarted gpu=%s pid=%s output=%s\n' "$(date --iso-8601=seconds)" "$gpu" "$!" "$output" >> "$watch_log"
}

while [ "$(date +%s)" -lt "$deadline" ]; do
  timestamp=$(date --iso-8601=seconds)
  mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
  for row in "${rows[@]}"; do
    printf '%s,%s\n' "$timestamp" "$row" >> "$gpu_log"
    gpu=$(printf '%s' "$row" | cut -d, -f1 | tr -d ' ')
    used=$(printf '%s' "$row" | cut -d, -f2 | tr -d ' ')
    pattern="poni_gpu_sustain_neuron.py.*--physical-gpu-index $gpu"
    if ! pgrep -u "$(id -u)" -f "$pattern" > /dev/null; then
      low_counts[$gpu]=0
      launch_sustain "$gpu"
      continue
    fi
    if [ "$used" -lt 40960 ]; then
      low_counts[$gpu]=$(( low_counts[$gpu] + 1 ))
    else
      low_counts[$gpu]=0
    fi
    if [ "${low_counts[$gpu]}" -ge 3 ]; then
      printf '%s low-memory restart gpu=%s used_mib=%s\n' "$timestamp" "$gpu" "$used" >> "$watch_log"
      mapfile -t pids < <(pgrep -u "$(id -u)" -f "$pattern")
      if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}"
      fi
      sleep 5
      launch_sustain "$gpu"
      low_counts[$gpu]=0
    fi
  done
  sleep 60
done

printf '%s watchdog complete\n' "$(date --iso-8601=seconds)" >> "$watch_log"
