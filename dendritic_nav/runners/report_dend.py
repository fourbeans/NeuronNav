"""Summarize paired development episodes, with clear point/uncertainty gates."""
import argparse
import json
from pathlib import Path
import numpy as np


def load_episodes(paths):
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    return rows


def compare(baseline, dendritic):
    base = {e["episode_id"]: e for e in baseline}
    dend = {e["episode_id"]: e for e in dendritic}
    common = sorted(set(base) & set(dend))
    if len(common) != len(baseline) or len(common) != len(dendritic):
        raise ValueError(
            "Evaluation episodes are not exactly paired (or contain duplicate IDs)"
        )
    if not common:
        raise ValueError("No evaluation episodes")
    rng = np.random.RandomState(123)
    result = dict(episodes=len(common))
    for metric in ("success", "spl"):
        a = np.asarray([base[k][metric] for k in common])
        b = np.asarray([dend[k][metric] for k in common])
        delta = b - a
        bootstrap = np.mean(
            delta[rng.randint(len(delta), size=(5000, len(delta)))], axis=1
        )
        result[metric] = dict(
            baseline=float(a.mean()),
            dendritic=float(b.mean()),
            difference=float(delta.mean()),
            paired_ci95=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        )
    result["point_estimate_non_regression"] = all(
        result[k]["difference"] >= -1e-9 for k in ("success", "spl")
    )
    result["pilot_gross_regression"] = (
        result["success"]["difference"] < -0.05 or result["spl"]["difference"] < -0.03
    )
    result[
        "interpretation"
    ] = "Development diagnostic only; not a claim of final-val improvement."
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    args = p.parse_args()
    root = Path(args.experiment)
    result = compare(
        load_episodes(sorted(root.glob("dev_baseline_*/episodes.jsonl"))),
        load_episodes(sorted(root.glob("dev_dend_*/episodes.jsonl"))),
    )
    with open(root / "development_comparison.json", "x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
