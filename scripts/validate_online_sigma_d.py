"""Reproducible contract test for the online Sigma-D policy adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semexp.model_pf_sigma_d import Potential_Function_Semantic_PolicySigmaD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Potential_Function_Semantic_PolicySigmaD(args.checkpoint).to(device).eval()
    channels = int(model.cfg.MODEL.num_categories)
    value = torch.zeros(2, channels, args.size, args.size, device=device)
    masks = torch.ones(2, 1, device=device)

    with torch.no_grad():
        object0, area0 = model(value, None, masks, None)
        object1, area1 = model(value, None, masks, None)
        changed = value.clone()
        changed[0, min(2, channels - 1), args.size // 2, args.size // 2] = 1.0
        object2, area2 = model(changed, None, masks, None)
        model.reset_runtime_state(torch.tensor([True, False]))
        object3, area3 = model(changed, None, masks, None)

    repeat_exact = torch.equal(object0, object1) and (
        area0 is None or torch.equal(area0, area1)
    )
    keep_exact = torch.equal(object2[1], object3[1]) and (
        area2 is None or torch.equal(area2[1], area3[1])
    )
    state_keys = [
        key for key in model.state_dict() if "runtime" in key or "previous_input" in key
    ]
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "input_shape": list(value.shape),
        "object_shape": list(object3.shape),
        "area_shape": None if area3 is None else list(area3.shape),
        "repeat_exact": bool(repeat_exact),
        "partial_reset_keep_exact": bool(keep_exact),
        "runtime_state_dict_keys": state_keys,
        "reset_statistics": dict(model.reset_statistics),
        "passed": bool(repeat_exact and keep_exact and not state_keys),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
