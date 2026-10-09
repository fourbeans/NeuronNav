"""Summarize strict-MP3D readout usage from per-worker JSONL audits."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    files = glob.glob(str(args.run_root / "audit_val_part_*/*.jsonl"))
    if not files:
        parser.error("no audit files found")
    total = changed = no_path = action_changed = fallback = 0
    invalid_strict_calls = plan_failures = 0
    distances = []
    ratios = []
    index_counts = {}
    for filename in files:
        for line in open(filename):
            event = json.loads(line)
            if event.get("type") == "plan_failure":
                plan_failures += 1
                continue
            if event.get("type") != "plan":
                continue
            total += 1
            fallback += bool(event.get("fallback"))
            invalid_strict_calls += bool(
                event.get("used") != "neuron" or event.get("reference") is not None
                or event.get("error") is not None
            )
            candidate = event["candidate"]
            info = candidate["info"]
            no_path += bool(info.get("no_path"))
            if not info.get("readout_applied"):
                continue
            changed += 1
            distances.append(float(info["readout_path_distance_cm"]))
            if info["readout_ray_path_cost_ratio"] is not None:
                ratios.append(float(info["readout_ray_path_cost_ratio"]))
            index = int(info["readout_path_index"])
            index_counts[str(index)] = index_counts.get(str(index), 0) + 1
            # This is a geometric STG difference, not necessarily an action change.
            action_changed += tuple(candidate["stg"]) != tuple(info["readout_old_stg"])
    result = {
        "plan_calls": total,
        "readout_applied_calls": changed,
        "readout_applied_fraction": changed / total if total else None,
        "geometric_stg_changed_calls": action_changed,
        "no_path_calls": no_path,
        "fallback_calls": fallback,
        "invalid_strict_calls": invalid_strict_calls,
        "plan_failures": plan_failures,
        "selected_path_index_histogram": index_counts,
        "selected_arc_cm_mean": float(np.mean(distances)) if distances else None,
        "selected_arc_cm_max": float(np.max(distances)) if distances else None,
        "ray_path_cost_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "ray_path_cost_ratio_max": float(np.max(ratios)) if ratios else None,
    }
    destination = args.run_root / "readout_usage.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
