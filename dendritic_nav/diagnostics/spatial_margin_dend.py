"""Read-only audit of score margins and executed goals in actor replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    totals = dict(steps=0, decisions=0, non_anchor=0, goal_switches=0,
                  close_001=0, close_006=0, close_02=0, close_06=0,
                  distant_2m=0, spatial_sacrifice=0.0)
    files = sorted(Path(root).glob("actor_*/replay/episode_*.npz"))
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            totals["steps"] += len(data["action"])
            totals["goal_switches"] += int(data["goal_switch"].sum())
            active = data["action"] >= 0
            spatial = data["spatial"][active]
            valid = data["valid"][active]
            action = data["action"][active].astype(int)
            if not len(action):
                continue
            second = np.max(np.where(valid[:, 1:], spatial[:, 1:], -np.inf), axis=1)
            gap = spatial[:, 0] - second
            rc = data["rc"][active]
            selected = rc[np.arange(len(action)), action]
            distance = np.linalg.norm((selected - rc[:, 0]) * 0.05, axis=1)
            totals["decisions"] += len(action)
            totals["non_anchor"] += int((action != 0).sum())
            for threshold, key in ((0.001, "close_001"), (0.006, "close_006"),
                                   (0.02, "close_02"), (0.06, "close_06")):
                totals[key] += int((gap <= threshold).sum())
            totals["distant_2m"] += int((distance > 2).sum())
            totals["spatial_sacrifice"] += float(
                (spatial[:, 0] - spatial[np.arange(len(action)), action]).sum()
            )
    count = max(totals["decisions"], 1)
    totals["files"] = len(files)
    totals["non_anchor_rate"] = totals["non_anchor"] / count
    totals["goal_switch_rate_per_step"] = totals["goal_switches"] / max(totals["steps"], 1)
    totals["distant_2m_rate"] = totals["distant_2m"] / count
    totals["mean_spatial_sacrifice"] = totals.pop("spatial_sacrifice") / count
    for key in ("close_001", "close_006", "close_02", "close_06"):
        totals[key + "_rate"] = totals[key] / count
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    args = parser.parse_args()
    print(json.dumps({path: summarize(path) for path in args.experiments}, indent=2))


if __name__ == "__main__":
    main()
