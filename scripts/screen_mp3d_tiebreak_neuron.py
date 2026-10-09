"""Frozen-map screen of a local equal-cost SNN path tie-breaker."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from plot_waveopt_neuron import _args_from_snapshot, _load_audit
from poni.path_tiebreak_neuron import count_bends, decode_straight_ties
from poni.planner_adapter_neuron import path_cost, proximity_weights, turn_action
from poni.wavefront_neuron import propagate


def screen(data):
    grid, start, goal, collision, visited, obstacle_kernel, goal_kernel = _args_from_snapshot(data)
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    free = np.pad(traversible != 0, 1, constant_values=True)
    goals = cv2.dilate(np.pad(np.asarray(goal, dtype=np.float32), 1), goal_kernel) == 1
    padded_start = (r + 1, c + 1)
    weights = proximity_weights(~free, wscale=float(data["wscale"]), niters=int(data["niters"]))
    wave = propagate(free, [padded_start], weights=weights, targets=goals)
    if wave.reached < 0:
        return None
    target = divmod(wave.reached, free.shape[1])
    old = wave.path_to(target)
    new, changed = decode_straight_ties(wave, weights, padded_start, target)
    old_cost, new_cost = path_cost(old, weights), path_cost(new, weights)
    if abs(old_cost - new_cost) > 1e-5:
        raise AssertionError(f"path cost changed: {old_cost} -> {new_cost}")
    old_stg = tuple(map(int, np.subtract(old[min(4, len(old) - 1)], 1)))
    new_stg = tuple(map(int, np.subtract(new[min(4, len(new) - 1)], 1)))
    heading, turn_angle = float(data["heading"]), float(data["turn_angle"])
    old_action = turn_action(old_stg, start, heading, turn_angle)
    new_action = turn_action(new_stg, start, heading, turn_angle)
    return {
        "same_stg": old_stg == new_stg,
        "same_action": old_action == new_action,
        "old_action": old_action, "new_action": new_action,
        "old_bends": count_bends(old), "new_bends": count_bends(new),
        "old_length": len(old), "new_length": len(new),
        "changed_parents": changed,
        "old_stg": old_stg, "new_stg": new_stg,
        "cost_difference": new_cost - old_cost,
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
                result = screen(data)
        except Exception as exc:
            skipped += 1
            if skipped <= 3:
                print(f"skipped {snapshot}: {exc}")
            continue
        if result is None:
            skipped += 1
            continue
        result["snapshot"] = str(snapshot)
        rows.append(result)
    changed = [row for row in rows if not row["same_stg"]]
    result = {
        "snapshots": len(rows), "skipped": skipped,
        "changed_stg": len(changed),
        "changed_action": sum(not row["same_action"] for row in rows),
        "old_forward_to_new_turn": sum(
            row["old_action"] == 1 and row["new_action"] in (2, 3) for row in rows),
        "old_turn_to_new_forward": sum(
            row["old_action"] in (2, 3) and row["new_action"] == 1 for row in rows),
        "opposite_turn": sum(
            row["old_action"] in (2, 3) and row["new_action"] in (2, 3)
            and row["old_action"] != row["new_action"] for row in rows),
        "bends_reduced": sum(row["new_bends"] < row["old_bends"] for row in rows),
        "bends_increased": sum(row["new_bends"] > row["old_bends"] for row in rows),
        "mean_bend_delta": float(np.mean([row["new_bends"] - row["old_bends"] for row in rows])) if rows else None,
        "max_abs_cost_difference": max((abs(row["cost_difference"]) for row in rows), default=None),
        "examples": changed[:10],
        "caveat": "Offline snapshots only; a smoother equal-cost path need not improve navigation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
