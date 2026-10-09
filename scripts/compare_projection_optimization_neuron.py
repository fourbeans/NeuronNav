#!/usr/bin/env python3
"""Verify full-field and early-stop goal projection choose identical actions."""

import argparse
import glob
import json

import numpy as np


def plans(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "plan":
                    rows.append(record.get("selected") or record["candidate"])
    return rows


def metrics(path):
    with open(path) as handle:
        return json.load(handle)["total_raw_metrics"]


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.5)),
        "p95": float(np.quantile(values, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-field-audit", required=True)
    parser.add_argument("--early-stop-audit", required=True)
    parser.add_argument("--full-field-stats", required=True)
    parser.add_argument("--early-stop-stats", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old = plans(args.full_field_audit)
    new = plans(args.early_stop_audit)
    if len(old) != len(new):
        raise RuntimeError("Plan counts differ: {} != {}".format(len(old), len(new)))
    repaired = []
    for index, (left, right) in enumerate(zip(old, new), 1):
        for key in ("stg", "stop", "replan", "turn_action"):
            if left[key] != right[key]:
                raise RuntimeError("Plan {} differs at {}".format(index, key))
        left_info, right_info = left["info"], right["info"]
        for key in (
            "arrival_cost", "no_path", "no_path_detected", "no_path_repaired",
            "projection_used", "projection_distance_cells", "proxy_goal",
            "component_spikes", "repair_spikes",
        ):
            if left_info.get(key) != right_info.get(key):
                raise RuntimeError("Plan {} info differs at {}".format(index, key))
        if right_info.get("projection_used"):
            repaired.append((left_info, right_info))

    old_metrics = metrics(args.full_field_stats)
    new_metrics = metrics(args.early_stop_stats)
    for key in old_metrics:
        if old_metrics[key] != new_metrics[key]:
            raise RuntimeError("Navigation metric array differs at {}".format(key))
    report = {
        "status": "pass",
        "plans_exact_action_match": len(old),
        "repaired_plans": len(repaired),
        "navigation_metrics_exact_match": True,
        "full_field_projection_spikes": distribution(
            [left["projection_spikes"] for left, _ in repaired]
        ),
        "early_stop_projection_spikes": distribution(
            [right["projection_spikes"] for _, right in repaired]
        ),
        "full_field_elapsed_ms": distribution(
            [left["elapsed_ms"] for left, _ in repaired]
        ),
        "early_stop_elapsed_ms": distribution(
            [right["elapsed_ms"] for _, right in repaired]
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    with open(args.output, "w") as handle:
        handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
