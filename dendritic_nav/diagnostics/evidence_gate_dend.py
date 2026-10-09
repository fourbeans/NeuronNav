"""Estimate online coordinate-entry evidence gates from saved real trajectories."""
import argparse
from collections import defaultdict
from pathlib import Path
import json

import numpy as np


def counts(data):
    entries = defaultdict(int)
    previous = None
    result = np.zeros(len(data["agent_xy"]), dtype=np.int16)
    for i, xy in enumerate(data["agent_xy"]):
        key = tuple(np.floor(xy / 0.25).astype(int))
        result[i] = entries[key]
        if key != previous:
            entries[key] += 1
            previous = key
    return result


def summarize(paths):
    out = {str(n): dict(steps=0, non_anchor=0, active=0, selected=0)
           for n in (2, 3, 4)}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            entry = counts(data)
            f = data["features"]
            selected = data["action"] > 0
            # Previous local stall/collision are observable, not ground truth.
            stalled = f[:, 2, 12] >= 0.1
            collided = f[:, 1, 5] >= np.log1p(2) / 5 - 1e-6
            for n, row in out.items():
                active = (entry >= int(n)) | stalled | collided
                row["steps"] += len(entry)
                row["non_anchor"] += int(selected.sum())
                row["active"] += int(active.sum())
                row["selected"] += int((active & selected).sum())
    for row in out.values():
        row["active_rate"] = row["active"] / max(row["steps"], 1)
        row["fraction_non_anchor_allowed"] = row["selected"] / max(row["non_anchor"], 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trace", action="append", default=[])
    args = parser.parse_args()
    root = Path(args.experiment)
    groups = {"train": sorted(root.glob("actor_*/replay/episode_*.npz"))}
    groups.update({path: [Path(path)] for path in args.trace})
    print(json.dumps({name: summarize(paths) for name, paths in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
