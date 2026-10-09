"""Classify offline strict-SNN versus weighted-A* path differences on frozen maps."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyastar

from plot_waveopt_neuron import _args_from_snapshot, _load_audit
from poni.planner_adapter_neuron import path_cost, proximity_weights
from poni.planner_adapter_opt_neuron import _free_for_ray
from poni.planner_adapter_strict_neuron import mp3d_plan_strict


def compare_snapshot(data):
    args = _args_from_snapshot(data)
    map_resolution = int(data["map_resolution"])
    wscale, niters = float(data["wscale"]), int(data["niters"])
    snn = mp3d_plan_strict(*args, map_resolution=map_resolution,
                           wscale=wscale, niters=niters)
    free = _free_for_ray(*args, gibson=False)
    padded_free = np.pad(free, 1, constant_values=True)
    goals = cv2.dilate(np.pad(np.asarray(args[2], np.float32), 1), args[6])
    obstacles = (1 - padded_free.astype(np.float32))
    cols, rows = pyastar.multi_goal_weighted_astar_planner(
        obstacles, [int(args[1][1]) + 1, int(args[1][0]) + 1],
        goals.astype(np.float32), True, wscale=wscale, niters=niters,
    )
    if cols is None or not snn.path:
        return None
    astar_path = [(int(r), int(c)) for r, c in zip(rows[::-1], cols[::-1])]
    snn_path = [(int(r) + 1, int(c) + 1) for r, c in snn.path]
    weights = proximity_weights(~padded_free, wscale=wscale, niters=niters)
    astar_cost = path_cost(astar_path, weights)
    snn_cost = path_cost(snn_path, weights)
    return {
        "same_goal_cell": astar_path[-1] == snn_path[-1],
        "same_stg": astar_path[min(4, len(astar_path) - 1)]
                    == snn_path[min(4, len(snn_path) - 1)],
        "snn_cost": snn_cost,
        "astar_cost": astar_cost,
        "cost_delta_astar_minus_snn": astar_cost - snn_cost,
        "same_cost_1e-4": abs(astar_cost - snn_cost) <= 1e-4,
        "snn_path_len": len(snn_path), "astar_path_len": len(astar_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    skipped = 0
    for snapshot, _, _ in _load_audit(args.snapshot_root, "mp3d"):
        if not snapshot.exists():
            continue
        try:
            with np.load(snapshot) as data:
                row = compare_snapshot(data)
        except Exception:
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        row["snapshot"] = str(snapshot)
        rows.append(row)
    mismatches = [row for row in rows if not row["same_stg"]]
    result = {
        "compared": len(rows), "skipped": skipped,
        "same_stg": sum(row["same_stg"] for row in rows),
        "same_goal_cell": sum(row["same_goal_cell"] for row in rows),
        "same_cost_1e-4": sum(row["same_cost_1e-4"] for row in rows),
        "different_stg_same_cost": sum(row["same_cost_1e-4"] for row in mismatches),
        "different_stg_different_goal": sum(not row["same_goal_cell"] for row in mismatches),
        "astar_cost_lower_than_snn": sum(row["cost_delta_astar_minus_snn"] < -1e-4 for row in rows),
        "astar_cost_higher_than_snn": sum(row["cost_delta_astar_minus_snn"] > 1e-4 for row in rows),
        "examples": sorted(mismatches, key=lambda row: -abs(row["cost_delta_astar_minus_snn"]))[:10],
        "caveat": "Offline sampled maps; cost equality can hide different paths and does not predict episode outcomes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
