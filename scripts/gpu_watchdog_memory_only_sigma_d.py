#!/usr/bin/env python3
"""Maintain a GPU memory floor using non-compute Sigma-D reserve processes."""

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
    return [tuple(int(part.strip()) for part in line.split(",")) for line in output.splitlines()]


def holder_alive(gpu: int) -> bool:
    result = subprocess.run(
        [
            "pgrep",
            "-u",
            str(os.getuid()),
            "-f",
            f"gpu_memory_reserve_sigma_d.py.*--physical-gpu-index {gpu}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def launch(root: Path, log_dir: Path, gpu: int, remaining_hours: float) -> int:
    command = [
        "/mnt/jfzn/miniconda3/envs/poni/bin/python",
        "scripts/gpu_memory_reserve_sigma_d.py",
        "--physical-gpu-index",
        str(gpu),
        "--target-total-memory-mib",
        "43000",
        "--safety-free-mib",
        "30000",
        "--duration-hours",
        str(max(remaining_hours + 2.0, 3.0)),
        "--log",
        str(log_dir / f"gpu{gpu}.jsonl"),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/jfzn/shensj/PONI")
    parser.add_argument("--duration-hours", type=float, default=96.0)
    parser.add_argument("--floor-mib", type=int, default=40960)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--low-polls", type=int, default=2)
    parser.add_argument("--log", required=True)
    parser.add_argument("--holder-log-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    log_path = Path(args.log).resolve()
    holder_log_dir = Path(args.holder_log_dir).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    holder_log_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.duration_hours * 3600.0
    low_counts = {gpu: 0 for gpu in range(8)}

    with log_path.open("a", buffering=1) as log:
        log.write(
            f"{dt.datetime.now().isoformat()} start duration_hours={args.duration_hours} "
            f"floor_mib={args.floor_mib} compute_workload=False\n"
        )
        while time.time() < deadline:
            try:
                now = dt.datetime.now().isoformat()
                for gpu, used, total, util in gpu_rows():
                    low_counts[gpu] = low_counts[gpu] + 1 if used < args.floor_mib else 0
                    log.write(
                        f"{now} gpu={gpu} used={used} total={total} util={util} "
                        f"low_polls={low_counts[gpu]}\n"
                    )
                    if low_counts[gpu] < args.low_polls or holder_alive(gpu):
                        continue
                    remaining = max((deadline - time.time()) / 3600.0, 0.0)
                    pid = launch(root, holder_log_dir, gpu, remaining)
                    log.write(f"{now} launched gpu={gpu} pid={pid} remaining_hours={remaining:.2f}\n")
                    low_counts[gpu] = 0
            except Exception as exc:
                log.write(f"{dt.datetime.now().isoformat()} error={exc!r}\n")
            time.sleep(args.poll_seconds)
        log.write(f"{dt.datetime.now().isoformat()} complete\n")


if __name__ == "__main__":
    main()
