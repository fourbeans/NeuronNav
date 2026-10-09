#!/usr/bin/env python3
"""Summarize the repaired smoke against the identical old strict episodes."""

import argparse
import glob
import json
from collections import Counter

import numpy as np


def load_metrics(path):
    with open(path) as handle:
        return json.load(handle)["total_raw_metrics"]


def means(metrics, count=None):
    count = len(metrics["success"]) if count is None else count
    return {
        key: float(np.mean(metrics[key][:count]))
        for key in ("success", "spl", "dts")
    }


def audit(files, episode_limit):
    counts = Counter()
    episode = None
    episodes_with_repairs = set()
    episodes_with_old_no_path = set()
    for path in files:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                kind = record.get("type", "unknown")
                if kind == "episode_start":
                    episode = int(record["episode_no"])
                if episode is None or episode > episode_limit or kind != "plan":
                    continue
                counts["plans"] += 1
                selected = record.get("selected") or record.get("candidate") or {}
                info = selected.get("info") or {}
                if info.get("projection_used"):
                    counts["projection_used"] += 1
                if info.get("no_path_repaired"):
                    counts["repaired"] += 1
                    episodes_with_repairs.add(episode)
                    counts["repair_stop_true"] += int(bool(selected.get("stop")))
                if info.get("no_path"):
                    counts["old_no_path"] += 1
                    episodes_with_old_no_path.add(episode)
    counts["episodes_with_repairs"] = len(episodes_with_repairs)
    counts["episodes_with_old_no_path"] = len(episodes_with_old_no_path)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-stats", required=True)
    parser.add_argument("--new-audit", required=True)
    parser.add_argument("--old-stats", required=True)
    parser.add_argument("--old-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    new = load_metrics(args.new_stats)
    old = load_metrics(args.old_stats)
    count = len(new["success"])
    if not np.array_equal(
        np.asarray(new["goal_distance"]), np.asarray(old["goal_distance"][:count])
    ):
        raise RuntimeError("Smoke and old strict runs are not the same episodes")
    report = {
        "episodes": count,
        "episode_identity": "exact_goal_distance_sequence_match",
        "old_constant_fill_strict_snn": means(old, count),
        "repaired_projection_strict_snn": means(new),
        "delta_repaired_minus_old": {
            key: means(new)[key] - means(old, count)[key]
            for key in ("success", "spl", "dts")
        },
        "old_audit": audit(glob.glob(args.old_audit), count),
        "new_audit": audit(glob.glob(args.new_audit), count),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    with open(args.output, "w") as handle:
        handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
