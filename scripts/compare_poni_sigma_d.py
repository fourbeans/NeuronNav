#!/usr/bin/env python3
"""Offline non-inferiority comparison on identical progressive map sequences."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from poni.dataset_sigma_d import SigmaDTemporalSemanticMapDataset
from poni.model_sigma_d import PONISigmaD
from train import SemanticMapperModule


def _load_original(checkpoint, device):
    model = SemanticMapperModule(checkpoint["cfg"])
    state = {}
    for key, value in checkpoint["state_dict"].items():
        key = key.replace(".module.", ".")
        if key.startswith("module."):
            key = key[len("module.") :]
        state[key] = value
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _losses(obj, area, batch, eps=1e-16):
    target = batch["object_pfs"].to(obj.device, non_blocking=True).float()
    mask = batch["loss_masks"].to(obj.device, non_blocking=True).float()
    pred = obj[:, :, 2:]
    target = target[:, :, 2:]
    mask = mask[:, :, 2:]
    per_category = ((pred - target).square() * mask).sum((-1, -2))
    per_category = per_category / (mask.sum((-1, -2)) + eps)
    object_loss = per_category.mean()
    if area is None:
        area_loss = object_loss.new_zeros(())
    else:
        area_target = batch["area_pfs"].to(area.device, non_blocking=True).float()
        area_loss = (area - area_target).square().mean()
    return object_loss, area_loss, object_loss + area_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sigma-checkpoint", default="",
        help="Optional trained sigma-d checkpoint; original checkpoint remains the baseline.",
    )
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--fmm-root", required=True)
    parser.add_argument("--dataset", choices=("gibson", "mp3d"), required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=1.0 / 128.0)
    parser.add_argument("--output-threshold", type=float, default=1.0 / 256.0)
    parser.add_argument("--qmax", type=int, default=15)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["cfg"]
    cfg.defrost()
    cfg.DATASET.dset_name = args.dataset
    cfg.DATASET.root = args.semantic_root
    cfg.DATASET.fmm_dists_saved_root = args.fmm_root
    cfg.DATASET.enable_unexp_area = bool(checkpoint["cfg"].DATASET.enable_unexp_area)
    cfg.freeze()
    dataset = SigmaDTemporalSemanticMapDataset(
        cfg.DATASET,
        split="val",
        sequence_length=args.sequence_length,
        samples_per_epoch=args.samples,
        seed=int(cfg.DATASET.seed),
        max_sampling_attempts=32,
        require_strict_growth=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
    )
    original = _load_original(checkpoint, device)
    sigma = PONISigmaD(
        num_categories=int(cfg.MODEL.num_categories),
        nsf=int(cfg.MODEL.nsf),
        embedding_size=int(cfg.MODEL.embedding_size),
        enable_area_head=bool(cfg.DATASET.enable_unexp_area),
        threshold=args.threshold,
        output_threshold=args.output_threshold,
        qmax=args.qmax,
        learnable_threshold=False,
        power_of_two_threshold=True,
    ).to(device).eval()
    load_report = sigma.load_original_poni_checkpoint(checkpoint, strict=True)
    sigma_source = "original_poni_warm_start"
    if args.sigma_checkpoint:
        trained = torch.load(args.sigma_checkpoint, map_location="cpu")
        if trained.get("model_family") != "sigma_d":
            raise RuntimeError("--sigma-checkpoint is not a sigma_d checkpoint")
        sigma.load_state_dict(trained["state_dict"], strict=True)
        sigma_source = str(Path(args.sigma_checkpoint).resolve())

    sums = defaultdict(float)
    count = 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True).float()
            batch_size, steps, channels, height, width = frames.shape
            flat = frames.reshape(batch_size * steps, channels, height, width)
            original_obj, original_area = original(flat)
            original_obj = original_obj.reshape(batch_size, steps, channels, height, width)
            if original_area is not None:
                original_area = original_area.reshape(batch_size, steps, 1, height, width)

            state = None
            sigma_obj_steps = []
            sigma_area_steps = []
            event_magnitude = 0
            active = 0
            saturated = 0
            values = 0
            for step in range(steps):
                result = sigma.forward_step(
                    frames[:, step], state,
                    reset_mask=torch.full(
                        (batch_size,), step == 0, dtype=torch.bool, device=device
                    ),
                )
                state = result.state
                sigma_obj_steps.append(result.object_pf)
                if result.area_pf is not None:
                    sigma_area_steps.append(result.area_pf)
                for layer in result.event_stats.get("layers", {}).values():
                    event_magnitude += int(layer.events.item())
                    active += int(layer.active.item())
                    saturated += int(layer.saturated.item())
                    values += int(layer.values)
            sigma_obj = torch.stack(sigma_obj_steps, dim=1)
            sigma_area = torch.stack(sigma_area_steps, dim=1) if sigma_area_steps else None

            original_losses = _losses(original_obj, original_area, batch)
            sigma_losses = _losses(sigma_obj, sigma_area, batch)
            weight = batch_size
            for prefix, values_tuple in (
                ("original", original_losses), ("sigma_d", sigma_losses)
            ):
                for name, value in zip(("object_loss", "area_loss", "task_loss"), values_tuple):
                    sums[f"{prefix}/{name}"] += float(value.item()) * weight
            sums["difference/object_mae"] += float(
                (sigma_obj - original_obj).abs().mean().item()
            ) * weight
            sums["difference/object_max_abs"] = max(
                sums["difference/object_max_abs"],
                float((sigma_obj - original_obj).abs().max().item()),
            )
            if sigma_area is not None:
                sums["difference/area_mae"] += float(
                    (sigma_area - original_area).abs().mean().item()
                ) * weight
            sums["events/magnitude"] += event_magnitude
            sums["events/active"] += active
            sums["events/saturated"] += saturated
            sums["events/values"] += values
            count += weight

    metrics = {}
    max_keys = {"difference/object_max_abs"}
    counter_keys = {
        "events/magnitude", "events/active", "events/saturated", "events/values"
    }
    for key, value in sums.items():
        metrics[key] = value if key in max_keys or key in counter_keys else value / count
    metrics["delta/task_loss"] = metrics["sigma_d/task_loss"] - metrics["original/task_loss"]
    metrics["ratio/task_loss"] = metrics["sigma_d/task_loss"] / max(
        metrics["original/task_loss"], 1e-16
    )
    metrics["events/active_rate"] = sums["events/active"] / max(sums["events/values"], 1)
    metrics["events/saturation_rate"] = sums["events/saturated"] / max(
        sums["events/values"], 1
    )
    report = {
        "schema": "poni-sigma-d-offline-comparison-v1",
        "dataset": args.dataset,
        "samples": count,
        "sequence_length": args.sequence_length,
        "threshold": args.threshold,
        "output_threshold": args.output_threshold,
        "qmax": args.qmax,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sigma_source": sigma_source,
        "loaded_layers": len(load_report.loaded_layers),
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
