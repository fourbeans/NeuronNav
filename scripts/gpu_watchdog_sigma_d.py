#!/usr/bin/env python3
"""Keep eight GPUs above the agreed memory floor without touching science jobs.

The watchdog never kills or mutates a scientific process.  If total memory on
one physical GPU stays below the floor and no occupancy companion for that GPU
is alive, it starts the existing real-PONI sustain workload.  Occupancy output
is sent to /tmp and is not part of scientific experiment results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import time
from pathlib import Path


def gpu_rows():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, used, total, util = [int(part.strip()) for part in line.split(",")]
        rows.append((index, used, total, util))
    return rows


def occupancy_alive(gpu: int) -> bool:
    result = subprocess.run(
        [
            "pgrep",
            "-u",
            str(os.getuid()),
            "-f",
            "poni_gpu_sustain_neuron.py.*--physical-gpu-index " + str(gpu),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def launch(root: Path, gpu: int, remaining_hours: float) -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path("/tmp") / f"sigma_d_watchdog_gpu{gpu}_{stamp}"
    command = [
        "/mnt/jfzn/miniconda3/envs/poni/bin/python",
        "scripts/poni_gpu_sustain_neuron.py",
        "--checkpoint",
        "pretrained_models/gibson_models/poni_seed_123.ckpt",
        "--output",
        str(output),
        "--duration-hours",
        str(max(remaining_hours + 2.0, 3.0)),
        "--target-total-memory-mib",
        "52000",
        "--safety-free-mib",
        "22000",
        "--micro-batch",
        "10",
        "--max-resident-batches",
        "160",
        "--physical-gpu-index",
        str(gpu),
        "--seed",
        str(4000 + gpu),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONPATH"] = str(root)
    process = subprocess.Popen(
        command,
        cwd=str(root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/jfzn/shensj/PONI")
    parser.add_argument("--duration-hours", type=float, default=49.0)
    parser.add_argument("--floor-mib", type=int, default=40960)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--low-polls", type=int, default=2)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log_path = Path(args.log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.duration_hours * 3600.0
    low_counts = {gpu: 0 for gpu in range(8)}
    with log_path.open("a", buffering=1) as log:
        log.write(
            f"{dt.datetime.now().isoformat()} start duration_hours={args.duration_hours} "
            f"floor_mib={args.floor_mib}\n"
        )
        while time.time() < deadline:
            try:
                rows = gpu_rows()
                now = dt.datetime.now().isoformat()
                for gpu, used, total, util in rows:
                    if used < args.floor_mib:
                        low_counts[gpu] += 1
                    else:
                        low_counts[gpu] = 0
                    log.write(
                        f"{now} gpu={gpu} used={used} total={total} util={util} "
                        f"low_polls={low_counts[gpu]}\n"
                    )
                    if low_counts[gpu] < args.low_polls or occupancy_alive(gpu):
                        continue
                    remaining = max((deadline - time.time()) / 3600.0, 0.0)
                    pid = launch(root, gpu, remaining)
                    log.write(
                        f"{now} launched gpu={gpu} pid={pid} "
                        f"remaining_hours={remaining:.2f}\n"
                    )
                    low_counts[gpu] = 0
            except Exception as exc:
                log.write(f"{dt.datetime.now().isoformat()} error={exc!r}\n")
            time.sleep(args.poll_seconds)
        log.write(f"{dt.datetime.now().isoformat()} complete\n")


if __name__ == "__main__":
    main()
