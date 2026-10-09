"""Sustain a real PONI U-Net training workload for GPU stability testing.

This is an operational benchmark, not a scientific training run.  It loads a
real PONI checkpoint, runs the original encoder and both decoder heads, and
keeps synthetic semantic-map/PF batches resident on the selected GPU.  The
pretrained checkpoint is never modified.
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train import SemanticMapperModule


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["cfg"]
    model = SemanticMapperModule(cfg)
    state = {}
    for key, value in checkpoint["state_dict"].items():
        state[key.replace("encoder.module.", "encoder.")
                 .replace("object_decoder.module.", "object_decoder.")
                 .replace("area_decoder.module.", "area_decoder.")] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    if model.area_decoder is None:
        raise RuntimeError("The sustain benchmark requires the original two-head PONI checkpoint")
    return model.to(device), cfg


def make_batch(batch_size, channels, map_size, device):
    # PONI inputs are cumulative free/wall/semantic maps in [0, 1].  Sparse
    # Bernoulli tensors exercise the same network without claiming real labels.
    semmap = (torch.rand(batch_size, channels, map_size, map_size, device=device) < 0.08).float()
    object_target = torch.rand(batch_size, channels, map_size, map_size, device=device)
    area_target = torch.rand(batch_size, 1, map_size, map_size, device=device)
    return semmap, object_target, area_target


def train_step(model, optimizer, batch):
    semmap, object_target, area_target = batch
    optimizer.zero_grad(set_to_none=True)
    object_pred, area_pred = model(semmap)
    loss = F.mse_loss(object_pred, object_target) + F.mse_loss(area_pred, area_target)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite PONI sustain loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def mib(value):
    return value / (1024.0 * 1024.0)


def device_memory_mib(physical_gpu_index):
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + str(physical_gpu_index),
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total = [float(value.strip()) for value in output.split(",")]
    return used, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-hours", type=float, default=49.0)
    parser.add_argument("--target-total-memory-mib", type=int, default=50000)
    parser.add_argument("--safety-free-mib", type=int, default=16000)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--max-resident-batches", type=int, default=64)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES")
    if args.micro_batch < 1 or args.duration_hours <= 0:
        parser.error("micro-batch and duration-hours must be positive")
    if args.target_total_memory_mib <= 0 or args.safety_free_mib <= 0:
        parser.error("memory targets must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "metrics.jsonl"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    model, cfg = load_model(args.checkpoint, device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    channels = int(cfg.MODEL.num_categories)
    map_size = int(cfg.MODEL.map_size)

    resident = []
    start = time.time()
    deadline = start + args.duration_hours * 3600.0
    step = 0
    last_report = start

    with log_path.open("x") as log:
        init = {
            "type": "init",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "duration_hours": args.duration_hours,
            "target_total_memory_mib": args.target_total_memory_mib,
            "safety_free_mib": args.safety_free_mib,
            "micro_batch": args.micro_batch,
            "channels": channels,
            "map_size": map_size,
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "scientific_training": False,
        }
        log.write(json.dumps(init) + "\n")
        log.flush()

        while time.time() < deadline:
            # Track total device use, including the navigation experiment.  Add
            # PONI batches when another job releases memory; release them if a
            # co-resident job grows and the safety margin becomes too small.
            device_used_mib, device_total_mib = device_memory_mib(args.physical_gpu_index)
            device_free_mib = device_total_mib - device_used_mib
            while (device_used_mib < args.target_total_memory_mib
                   and len(resident) < args.max_resident_batches):
                try:
                    resident.append(make_batch(args.micro_batch, channels, map_size, device))
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    torch.cuda.empty_cache()
                    break
                device_used_mib, device_total_mib = device_memory_mib(args.physical_gpu_index)
                device_free_mib = device_total_mib - device_used_mib
            while device_free_mib < args.safety_free_mib and len(resident) > 1:
                resident.pop()
                torch.cuda.empty_cache()
                device_used_mib, device_total_mib = device_memory_mib(args.physical_gpu_index)
                device_free_mib = device_total_mib - device_used_mib
            if not resident:
                resident.append(make_batch(args.micro_batch, channels, map_size, device))

            tick = time.time()
            batch = resident[step % len(resident)]
            loss = train_step(model, optimizer, batch)
            torch.cuda.synchronize()
            step += 1
            now = time.time()
            if step == 1 or now - last_report >= 30.0:
                row = {
                    "type": "progress",
                    "step": step,
                    "elapsed_hours": (now - start) / 3600.0,
                    "step_seconds": now - tick,
                    "loss": loss,
                    "resident_batches": len(resident),
                    "resident_samples": len(resident) * args.micro_batch,
                    "memory_allocated_mib": mib(torch.cuda.memory_allocated()),
                    "memory_reserved_mib": mib(torch.cuda.memory_reserved()),
                    "max_memory_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                    "max_memory_reserved_mib": mib(torch.cuda.max_memory_reserved()),
                    "device_used_mib": device_used_mib,
                    "device_free_mib": device_free_mib,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                last_report = now

        done = {
            "type": "complete",
            "steps": step,
            "elapsed_hours": (time.time() - start) / 3600.0,
            "memory_reserved_mib": mib(torch.cuda.memory_reserved()),
        }
        log.write(json.dumps(done) + "\n")
        log.flush()
        print(json.dumps(done), flush=True)


if __name__ == "__main__":
    main()
