#!/usr/bin/env python3
"""Reserve otherwise idle GPU memory without consuming accelerator compute.

This operational helper is deliberately not a training workload.  It keeps a
list of byte tensors alive so that an agreed memory floor remains occupied,
but it performs no forward/backward passes.  The allocation is elastic: chunks
are released whenever a co-resident scientific job needs the configured safety
margin.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch


def device_memory_mib(physical_gpu_index: int) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total = (int(value.strip()) for value in output.split(","))
    return used, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--target-total-memory-mib", type=int, default=43000)
    parser.add_argument("--safety-free-mib", type=int, default=30000)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--duration-hours", type=float, default=96.0)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if min(
        args.target_total_memory_mib,
        args.safety_free_mib,
        args.chunk_mib,
        args.poll_seconds,
        args.duration_hours,
    ) <= 0:
        parser.error("memory, polling, and duration arguments must be positive")

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    chunk_bytes = args.chunk_mib * 1024 * 1024
    deadline = time.time() + args.duration_hours * 3600.0

    with log_path.open("a", buffering=1) as log:
        while time.time() < deadline:
            used, total = device_memory_mib(args.physical_gpu_index)

            # A scientific process may grow after this holder starts.  Release
            # our newest chunks first and return them to CUDA immediately.
            while chunks and total - used < args.safety_free_mib:
                chunks.pop()
                torch.cuda.empty_cache()
                used, total = device_memory_mib(args.physical_gpu_index)

            # Reserve only enough otherwise-idle memory to reach the target.
            while (
                used + args.chunk_mib <= args.target_total_memory_mib
                and total - used - args.chunk_mib >= args.safety_free_mib
            ):
                try:
                    chunks.append(
                        torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0")
                    )
                    used, total = device_memory_mib(args.physical_gpu_index)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    break

            row = {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "gpu": args.physical_gpu_index,
                "device_used_mib": used,
                "device_total_mib": total,
                "holder_chunks": len(chunks),
                "holder_requested_mib": len(chunks) * args.chunk_mib,
                "compute_workload": False,
            }
            log.write(json.dumps(row) + "\n")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
