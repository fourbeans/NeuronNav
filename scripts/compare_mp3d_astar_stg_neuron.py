"""Offline STG/action agreement between strict SNN and original weighted A*."""

import argparse
import json
from pathlib import Path

import numpy as np

from plot_waveopt_neuron import _args_from_snapshot, _load_audit, _reference_stg
from poni.planner_adapter_neuron import turn_action
from poni.planner_adapter_strict_neuron import mp3d_plan_strict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compared = equal_stg = equal_action = skipped = 0
    examples = []
    for snapshot, event, _ in _load_audit(args.snapshot_root, "mp3d"):
        if not snapshot.exists():
            continue
        with np.load(snapshot) as data:
            try:
                planner_args = _args_from_snapshot(data)
                old = mp3d_plan_strict(
                    *planner_args, map_resolution=int(data["map_resolution"]),
                    wscale=float(data["wscale"]), niters=int(data["niters"]),
                )
                reference = _reference_stg(data, "mp3d")
                if reference is None:
                    skipped += 1
                    continue
                start = planner_args[1]
                heading, turn_angle = float(data["heading"]), float(data["turn_angle"])
                action_old = turn_action(old.stg, start, heading, turn_angle)
                action_ref = turn_action(reference, start, heading, turn_angle)
            except Exception as exc:
                skipped += 1
                if len(examples) < 10:
                    examples.append({"snapshot": str(snapshot), "error": repr(exc)})
                continue
            compared += 1
            same_stg = np.allclose(old.stg, reference, atol=1e-8, rtol=0)
            equal_stg += same_stg
            equal_action += action_old == action_ref
            if not same_stg and len(examples) < 10:
                examples.append({
                    "snapshot": str(snapshot), "seq": event["seq"],
                    "snn_stg": list(old.stg), "astar_stg": list(reference),
                    "snn_action": action_old, "astar_action": action_ref,
                    "snn_path_len": len(old.path),
                })
    result = {
        "compared_snapshots": compared, "skipped": skipped,
        "equal_stg": equal_stg, "equal_stg_fraction": equal_stg / compared if compared else None,
        "equal_action": equal_action,
        "equal_action_fraction": equal_action / compared if compared else None,
        "examples": examples,
        "caveat": "Frozen sampled maps, not whole-episode performance; weighted A* is offline only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
