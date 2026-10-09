"""Matched frozen-map CPU latency check for old and tie-break SNN planners."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from plot_waveopt_neuron import _args_from_snapshot, _load_audit
from poni.planner_adapter_mp3d_tiebreak_neuron import mp3d_plan_tiebreak
from poni.planner_adapter_strict_neuron import mp3d_plan_strict
from poni.wavefront_neuron import warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-snapshots", type=int, default=50)
    args = parser.parse_args()
    warmup()
    records = []
    for snapshot, _, _ in _load_audit(args.snapshot_root, "mp3d"):
        if len(records) >= args.max_snapshots:
            break
        if not snapshot.exists():
            continue
        with np.load(snapshot) as data:
            planner_args = _args_from_snapshot(data)
            kwargs = dict(map_resolution=int(data["map_resolution"]),
                          wscale=float(data["wscale"]), niters=int(data["niters"]))
            try:
                planners = (("old", mp3d_plan_strict), ("new", mp3d_plan_tiebreak))
                if len(records) % 2:
                    planners = planners[::-1]
                results = {}
                elapsed = {}
                for label, planner in planners:
                    start = time.perf_counter()
                    results[label] = planner(*planner_args, **kwargs)
                    elapsed[label] = (time.perf_counter() - start) * 1000
                old, new = results["old"], results["new"]
            except Exception:
                continue
            if not old.path or not new.path:
                continue
            if abs(old.info["path_cost"] - new.info["path_cost"]) > 1e-5:
                raise AssertionError("weighted path cost changed")
            records.append((elapsed["old"], elapsed["new"], len(old.path)))
    rows = np.asarray(records, dtype=float)
    result = {
        "matched_snapshots": len(records),
        "old_median_ms": float(np.median(rows[:, 0])) if len(records) else None,
        "new_median_ms": float(np.median(rows[:, 1])) if len(records) else None,
        "old_p95_ms": float(np.percentile(rows[:, 0], 95)) if len(records) else None,
        "new_p95_ms": float(np.percentile(rows[:, 1], 95)) if len(records) else None,
        "median_extra_ms": float(np.median(rows[:, 1] - rows[:, 0])) if len(records) else None,
        "median_old_path_cells": float(np.median(rows[:, 2])) if len(records) else None,
        "caveat": "Sequential CPU benchmark on a shared server; not chip latency or whole-agent latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
