#!/usr/bin/env python3
"""Numerical contract tests for the incremental Sigma-Delta PONI U-Net.

This is deliberately an independent validation entry point.  It does not
train, mutate checkpoints, or silently fall back to original PONI.  The only
use of original PONI is as a read-only first-frame numerical reference.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch

from poni.model_sigma_d import PONISigmaD
from train import SemanticMapperModule


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _sum_stats(result, field: str) -> int:
    return sum(
        int(getattr(layer_stats, field).detach().cpu().item())
        for layer_stats in result.event_stats.get("layers", {}).values()
    )


def _build_original(checkpoint: Dict[str, Any], device: torch.device):
    cfg = checkpoint["cfg"]
    model = SemanticMapperModule(cfg)
    normalized = {}
    for key, value in checkpoint["state_dict"].items():
        key = key.replace(".module.", ".")
        if key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)
    return model.eval().to(device)


def _build_sigma(checkpoint: Dict[str, Any], device: torch.device, args):
    cfg = checkpoint["cfg"]
    model = PONISigmaD(
        num_categories=cfg.MODEL.num_categories,
        nsf=cfg.MODEL.nsf,
        embedding_size=cfg.MODEL.embedding_size,
        enable_area_head=cfg.DATASET.enable_unexp_area,
        threshold=args.threshold,
        output_threshold=args.output_threshold,
        qmax=args.qmax,
        learnable_threshold=False,
        power_of_two_threshold=True,
    )
    report = model.load_original_poni_checkpoint(checkpoint, strict=True)
    return model.eval().to(device), report


def validate(args) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # TF32 changes the accumulation order enough that a folded Conv+BN can
    # cross a Sigma-Delta rounding boundary.  Contract validation uses full
    # FP32; performance experiments may enable TF32 explicitly and record it.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = checkpoint["cfg"]
    channels = int(cfg.MODEL.num_categories)

    original = _build_original(checkpoint, device)
    sigma, load_report = _build_sigma(checkpoint, device, args)
    x0 = torch.rand(args.batch_size, channels, args.map_size, args.map_size, device=device)

    start = time.time()
    with torch.no_grad():
        ref_obj, ref_area = original(x0)
        dense = sigma.forward_step(x0)
    dense_obj_max = _as_float((ref_obj - dense.object_pf).abs().max())
    dense_obj_mean = _as_float((ref_obj - dense.object_pf).abs().mean())
    if ref_area is None:
        dense_area_max = None
    else:
        dense_area_max = _as_float((ref_area - dense.area_pf).abs().max())

    # A repeated map must emit no events and preserve the prediction exactly.
    with torch.no_grad():
        repeated = sigma.forward_step(x0.clone(), dense.state)
    repeated_error = _as_float((dense.object_pf - repeated.object_pf).abs().max())
    repeated_active = _sum_stats(repeated, "active")
    repeated_events = _sum_stats(repeated, "events")

    # A signed add/remove perturbation must exercise both incremental paths.
    x1 = x0.clone()
    x1[:, :, : args.map_size // 3, : args.map_size // 3] = 1.0 - x1[
        :, :, : args.map_size // 3, : args.map_size // 3
    ]
    with torch.no_grad():
        changed = sigma.forward_step(x1, repeated.state)
    changed_active = _sum_stats(changed, "active")
    changed_events = _sum_stats(changed, "events")
    changed_saturated = _sum_stats(changed, "saturated")

    # Per-row reset must be identical to an independent dense first frame for
    # the reset row and to an independent incremental stream for the kept row.
    if args.batch_size < 2:
        xm0 = torch.rand(2, channels, args.map_size, args.map_size, device=device)
    else:
        xm0 = x0[:2]
    xm1 = xm0.clone()
    xm1[0, :, :5, :7] = 1.0 - xm1[0, :, :5, :7]
    xm1[1, :, -5:, -7:] = 1.0 - xm1[1, :, -5:, -7:]
    with torch.no_grad():
        batch_init = sigma.forward_step(xm0)
        mixed = sigma.forward_step(
            xm1, batch_init.state, reset_mask=torch.tensor([False, True], device=device)
        )
        row0_init = sigma.forward_step(xm0[0:1])
        row0_update = sigma.forward_step(xm1[0:1], row0_init.state)
        row1_dense = sigma.forward_step(xm1[1:2])
    mixed_keep_error = _as_float(
        (mixed.object_pf[0:1] - row0_update.object_pf).abs().max()
    )
    mixed_reset_error = _as_float(
        (mixed.object_pf[1:2] - row1_dense.object_pf).abs().max()
    )

    # A short real recurrent graph must carry finite, non-zero gradients.
    sigma.train()
    sigma.zero_grad(set_to_none=True)
    gx0 = torch.rand(1, channels, args.grad_map_size, args.grad_map_size, device=device)
    gx1 = gx0.clone()
    gx1[:, :, :5, :5] = 1.0 - gx1[:, :, :5, :5]
    g0 = sigma.forward_step(gx0)
    g1 = sigma.forward_step(gx1, g0.state)
    grad_loss = g0.object_pf.square().mean() + g1.object_pf.square().mean()
    if g0.area_pf is not None:
        grad_loss = grad_loss + g0.area_pf.square().mean() + g1.area_pf.square().mean()
    grad_loss.backward()
    gradients = [p.grad for p in sigma.parameters() if p.requires_grad and p.grad is not None]
    grad_abs_sum = sum(_as_float(g.abs().sum()) for g in gradients)
    grad_all_finite = all(bool(torch.isfinite(g).all().item()) for g in gradients)

    state_keys = list(sigma.state_dict().keys())
    forbidden_state_keys = [
        key for key in state_keys if any(token in key for token in ("previous_input", "reconstruction", "preactivation"))
    ]

    tolerances = {
        "dense_obj_max": 1e-4,
        "dense_area_max": 1e-4,
        "repeated_error": 0.0,
        # Separate batch kernels can differ by a few FP32 ulps although the
        # explicit-state merge is mathematically identical.
        "mixed_error": 1e-5,
    }
    checks = {
        "checkpoint_mapping_complete": not load_report.missing_layers,
        "dense_object_matches_original": dense_obj_max <= tolerances["dense_obj_max"],
        "dense_area_matches_original": dense_area_max is None
        or dense_area_max <= tolerances["dense_area_max"],
        "zero_delta_exact": repeated_error == tolerances["repeated_error"],
        "zero_delta_no_events": repeated_active == 0 and repeated_events == 0,
        "changed_map_emits_events": changed_active > 0 and changed_events > 0,
        "mixed_keep_matches": mixed_keep_error <= tolerances["mixed_error"],
        "mixed_reset_matches": mixed_reset_error <= tolerances["mixed_error"],
        "gradient_nonzero": grad_abs_sum > 0,
        "gradient_finite": grad_all_finite,
        "runtime_state_excluded": not forbidden_state_keys,
    }
    report = {
        "schema": "sigma-d-validation-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "map_size": args.map_size,
        "grad_map_size": args.grad_map_size,
        "threshold": args.threshold,
        "output_threshold": args.output_threshold,
        "qmax": args.qmax,
        "loaded_layers": len(load_report.loaded_layers),
        "unused_source_keys": len(load_report.unused_source_keys),
        "metrics": {
            "dense_object_max_abs": dense_obj_max,
            "dense_object_mean_abs": dense_obj_mean,
            "dense_area_max_abs": dense_area_max,
            "repeated_prediction_max_abs": repeated_error,
            "repeated_active": repeated_active,
            "repeated_events": repeated_events,
            "changed_active": changed_active,
            "changed_events": changed_events,
            "changed_saturated": changed_saturated,
            "mixed_keep_max_abs": mixed_keep_error,
            "mixed_reset_max_abs": mixed_reset_error,
            "gradient_abs_sum": grad_abs_sum,
            "gradient_tensor_count": len(gradients),
            "forbidden_runtime_state_keys": forbidden_state_keys,
            "elapsed_seconds": time.time() - start,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    if not all(math.isfinite(v) for v in (dense_obj_max, dense_obj_mean, grad_abs_sum)):
        report["passed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--grad-map-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=1.0 / 128.0)
    parser.add_argument("--output-threshold", type=float, default=1.0 / 255.0)
    parser.add_argument("--qmax", type=int, default=15)
    args = parser.parse_args()
    report = validate(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
