"""Strict-SNN compliance and tie-break usage summary from MP3D audits."""

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
    if len(files) != 11:
        parser.error(f"expected 11 audit files, got {len(files)}")
    plans = failures = invalid = no_path = changed_stg = 0
    bend_deltas = []
    cost_deltas = []
    changed_parents = []
    for filename in files:
        for line in open(filename):
            event = json.loads(line)
            if event.get("type") == "plan_failure":
                failures += 1
                continue
            if event.get("type") != "plan":
                continue
            plans += 1
            info = event["selected"]["info"]
            invalid += bool(event.get("used") != "neuron" or event.get("fallback")
                            or event.get("reference") is not None or event.get("error") is not None)
            no_path += bool(info.get("no_path"))
            if info.get("readout_mode") != "equal_arrival_straight_tie":
                continue
            changed_stg += tuple(event["selected"]["stg"]) != tuple(info["tie_old_stg"])
            bend_deltas.append(info["tie_new_bends"] - info["tie_old_bends"])
            cost_deltas.append(info["tie_cost_delta"])
            changed_parents.append(info["tie_changed_parents"])
    result = {
        "plan_calls": plans, "plan_failures": failures, "invalid_strict_calls": invalid,
        "no_path_calls": no_path,
        "tie_path_calls": len(bend_deltas),
        "changed_stg_calls": changed_stg,
        "mean_bend_delta": float(np.mean(bend_deltas)) if bend_deltas else None,
        "bends_reduced_calls": sum(x < 0 for x in bend_deltas),
        "bends_increased_calls": sum(x > 0 for x in bend_deltas),
        "max_abs_cost_delta": max((abs(x) for x in cost_deltas), default=None),
        "mean_changed_parents": float(np.mean(changed_parents)) if changed_parents else None,
    }
    (args.run_root / "tiebreak_usage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
