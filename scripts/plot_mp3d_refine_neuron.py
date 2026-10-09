"""MP3D bounded-visible map, trajectory, and paired-outcome diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from plot_waveopt_neuron import _args_from_snapshot, _draw_map, _load_audit, outcome_chart, trajectory
from poni.planner_adapter_mp3d_refine_neuron import mp3d_plan_refine
from poni.planner_adapter_neuron import turn_action
from poni.planner_adapter_strict_neuron import mp3d_plan_strict


def plans(data, cap_cm, cost_guard, angle_deg):
    args = _args_from_snapshot(data)
    kwargs = dict(map_resolution=int(data["map_resolution"]),
                  wscale=float(data["wscale"]), niters=int(data["niters"]))
    return (mp3d_plan_strict(*args, **kwargs),
            mp3d_plan_refine(*args, **kwargs, cap_cm=cap_cm, cost_guard=cost_guard,
                             max_direction_delta_deg=angle_deg))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cap-cm", type=float, required=True)
    parser.add_argument("--cost-guard", type=int, choices=(0, 1), required=True)
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--max-maps", type=int, default=4)
    options = parser.parse_args()
    output = options.run_root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for snapshot, event, _ in _load_audit(options.run_root, "mp3d"):
        if not snapshot.exists():
            continue
        with np.load(snapshot) as data:
            old, new = plans(data, options.cap_cm, bool(options.cost_guard), options.angle_deg)
            if len(old.path) <= 1:
                continue
            args = _args_from_snapshot(data)
            heading = float(data["heading"])
            turn_angle = float(data["turn_angle"])
            old_action = turn_action(old.stg, args[1], heading, turn_angle)
            new_action = turn_action(new.stg, args[1], heading, turn_angle)
            shift = np.linalg.norm(np.subtract(new.stg, old.stg))
            score = (100 if old_action != new_action else 0) + min(shift, 50)
            candidates.append((score, snapshot, event, old_action != new_action))
    selected = sorted(candidates, key=lambda row: -row[0])[:options.max_maps]
    for index, (_, snapshot, event, _) in enumerate(selected, 1):
        with np.load(snapshot) as data:
            old, new = plans(data, options.cap_cm, bool(options.cost_guard), options.angle_deg)
            _draw_map(data, "mp3d", old, new, event, output / f"map_{index:02d}.png")
    result = {
        "map_examples": len(selected),
        "snapshot_action_divergences": sum(item[3] for item in candidates),
        "snapshots_considered": len(candidates),
        "trajectory": trajectory(options.run_root, "mp3d", output),
        "success_flips": outcome_chart(options.run_root, output),
        "output": str(output),
    }
    (options.run_root / "figures_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
