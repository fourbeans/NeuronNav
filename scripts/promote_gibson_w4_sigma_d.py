#!/usr/bin/env python3
"""Promote the live Gibson Sigma-D run from two to four GPUs safely.

The handoff is intentionally gated by two independently verifiable events:

1. the original two-rank run saves a requested exact epoch boundary;
2. the GPU4 Gibson navigation evaluation exits and releases that device.

The script first creates a new normalized checkpoint without modifying the
source run.  It then terminates only the explicitly supplied old launcher and
starts a four-rank continuation in a separate output directory, preserving an
effective batch size of eight.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


def process_alive(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    try:
        state = stat_path.read_text().split()[2]
    except (OSError, IndexError):
        return False
    return state != "Z"


def write_log(handle, message: str) -> None:
    handle.write(f"{dt.datetime.now().isoformat()} {message}\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--destination-run", required=True)
    parser.add_argument("--derived-checkpoint", required=True)
    parser.add_argument("--navigation-pid", type=int, required=True)
    parser.add_argument("--old-launcher-pid", type=int, required=True)
    parser.add_argument("--target-update", type=int, default=30000)
    parser.add_argument("--expected-next-batch", type=int, default=10000)
    parser.add_argument("--master-port", type=int, default=29504)
    parser.add_argument("--log", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    source_run = (root / args.source_run).resolve()
    destination_run = (root / args.destination_run).resolve()
    derived = (root / args.derived_checkpoint).resolve()
    log_path = (root / args.log).resolve()
    last_checkpoint = source_run / "checkpoints" / "last.ckpt"
    best_checkpoint = source_run / "checkpoints" / "best.ckpt"
    helper = root / "scripts" / "prepare_epoch_boundary_resume_sigma_d.py"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", buffering=1) as log:
        write_log(
            log,
            f"waiting for Gibson update={args.target_update} boundary "
            "and navigation completion",
        )
        last_mtime = None
        while not derived.exists():
            try:
                mtime = last_checkpoint.stat().st_mtime_ns
                if mtime != last_mtime:
                    last_mtime = mtime
                    checkpoint = torch.load(str(last_checkpoint), map_location="cpu")
                    update = int(checkpoint["global_update"])
                    next_batch = int(checkpoint.get("next_batch_in_epoch", -1))
                    write_log(log, f"observed update={update} next_batch={next_batch}")
                    del checkpoint
                    if (
                        update == args.target_update
                        and next_batch == args.expected_next_batch
                    ):
                        subprocess.run(
                            [
                                sys.executable,
                                str(helper),
                                "--input",
                                str(last_checkpoint),
                                "--output",
                                str(derived),
                                "--expected-update",
                                str(args.target_update),
                                "--expected-next-batch",
                                str(args.expected_next_batch),
                            ],
                            cwd=str(root),
                            check=True,
                        )
                        write_log(log, f"created derived checkpoint {derived}")
                        break
                    if update > args.target_update:
                        raise RuntimeError(
                            f"missed required update={args.target_update} boundary"
                        )
            except FileNotFoundError:
                pass
            time.sleep(args.poll_seconds)

        while process_alive(args.navigation_pid):
            time.sleep(args.poll_seconds)
        write_log(log, f"navigation pid={args.navigation_pid} completed")

        if not process_alive(args.old_launcher_pid):
            raise RuntimeError(
                f"old Gibson launcher {args.old_launcher_pid} exited before handoff"
            )
        os.kill(args.old_launcher_pid, signal.SIGTERM)
        write_log(log, f"sent SIGTERM to old launcher pid={args.old_launcher_pid}")
        for _ in range(60):
            if not process_alive(args.old_launcher_pid):
                break
            time.sleep(1)
        if process_alive(args.old_launcher_pid):
            raise RuntimeError("old launcher did not exit after 60 seconds")

        destination_run.mkdir(parents=True, exist_ok=False)
        (destination_run / "checkpoints").mkdir()
        shutil.copy2(best_checkpoint, destination_run / "checkpoints" / "best.ckpt")
        driver = (destination_run / "driver.log").open("w")
        command = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.launch",
            "--use_env",
            "--nproc_per_node=4",
            f"--master_port={args.master_port}",
            "train_sigma_d.py",
            "--exp-config",
            "configs/sigma_d/gibson_sigma_d.yaml",
            "--resume",
            str(derived),
            "LOGGING.output_dir",
            str(destination_run),
            "LOGGING.auto_resume",
            "False",
            "SEED",
            "234",
            "DATASET.seed",
            "234",
            "MODEL.pretrained_path",
            "pretrained_models/gibson_models/poni_seed_234.ckpt",
            "MODEL.sigma_d.qmax",
            "4095",
            "OPTIM.lr",
            "0.00001",
            "OPTIM.num_total_updates",
            "40000",
            "DATASET.train_samples_per_epoch",
            "20000",
            "DATASET.val_samples_per_epoch",
            "1000",
            "OPTIM.batch_size",
            "1",
            "OPTIM.grad_accum_steps",
            "2",
            "LOGGING.log_interval",
            "20",
            "LOGGING.eval_interval",
            "1000",
            "LOGGING.ckpt_interval",
            "1000",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        environment["PYTHONPATH"] = str(root)
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=driver,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (destination_run / "launcher.pid").write_text(f"{process.pid}\n")
        write_log(
            log,
            f"launched four-GPU Gibson pid={process.pid} GPUs=4,5,6,7 "
            "effective_batch=8",
        )


if __name__ == "__main__":
    main()
