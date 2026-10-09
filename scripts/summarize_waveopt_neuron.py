"""Summarize a complete or pilot SNN-wave readout evaluation without mixing cohorts."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


PONI_ROOT = Path(__file__).resolve().parents[1]
SIGMA_NAV = PONI_ROOT / "dairy/sigma_d_unet_20260924/navigation"
JOINT = PONI_ROOT / "dairy/joint_sigma_d_strict_neuron_20261004"


def _paired(new, old):
    delta = np.asarray(new, dtype=float) - np.asarray(old, dtype=float)
    n = len(delta)
    return {
        "mean": float(delta.mean()),
        "paired_se": float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "improved": int((delta > 1e-9).sum()),
        "worsened": int((delta < -1e-9).sum()),
        "equal": int((abs(delta) <= 1e-9).sum()),
    }


def _gibson(path, full_baseline):
    p = path / "gibson_objectnav/dump/seed_100/final_eval_stats.json"
    raw = json.loads(p.read_text())["total_raw_metrics"]
    n = len(raw["success"])
    if n % 5:
        raise ValueError(f"Gibson count {n} is not five balanced scenes")
    per_scene = n // 5
    if full_baseline:
        return raw
    return raw, per_scene


def _gibson_reference(path, per_scene):
    raw = _gibson(path, True)
    full_count = len(raw["success"]) // 5
    return {
        name: [raw[name][scene * full_count + ep]
               for scene in range(5) for ep in range(per_scene)]
        for name in raw
    }


def summarize_gibson(run_root):
    current, per_scene = _gibson(run_root, False)
    baselines = {
        "sigma_d_fmm": SIGMA_NAV / "gibson_seed234_best32k",
        "sigma_d_snn_old": JOINT / "gibson_seed234_best32k",
    }
    metrics = ("success", "spl", "dts")
    result = {
        "dataset": "gibson", "episodes": len(current["success"]),
        "episodes_per_scene": per_scene,
        "metrics": {k: float(np.mean(current[k])) for k in metrics},
        "baselines": {},
    }
    for name, path in baselines.items():
        prior = _gibson_reference(path, per_scene)
        cohort = bool(np.allclose(current["goal_distance"], prior["goal_distance"],
                                  rtol=0, atol=1e-8))
        result["baselines"][name] = {
            "cohort_match": cohort,
            "metrics": {k: float(np.mean(prior[k])) for k in metrics},
            "difference": {k: _paired(current[k], prior[k]) for k in metrics} if cohort else None,
        }
    return result


def _mp3d(run_root):
    stats = {}
    counts = {}
    for p in sorted(glob.glob(str(run_root / "tb_seed_100_val_part_*/stats.json"))):
        part = p.split("val_part_")[-1].split("/")[0]
        data = json.loads(Path(p).read_text())
        overlap = stats.keys() & data.keys()
        if overlap:
            raise ValueError(f"repeated episode IDs: {len(overlap)}")
        stats.update(data)
        counts[part] = len(data)
    if len(counts) != 11:
        raise ValueError(f"expected 11 MP3D parts, found {len(counts)}")
    return stats, counts


def summarize_mp3d(run_root):
    current, counts = _mp3d(run_root)
    baselines = {
        "sigma_d_astar": SIGMA_NAV / "mp3d_seed123_best2k_4gpu",
        "sigma_d_snn_old": JOINT / "mp3d_seed123_best2k",
    }
    keys = sorted(current)
    metrics = ("success", "spl", "distance_to_goal")
    result = {
        "dataset": "mp3d", "episodes": len(keys), "part_episode_counts": counts,
        "metrics": {k: float(np.mean([current[e][k] for e in keys])) for k in metrics},
        "baselines": {},
    }
    for name, path in baselines.items():
        prior, _ = _mp3d(path)
        cohort = set(current).issubset(prior)
        result["baselines"][name] = {
            "cohort_match": cohort,
            "metrics": {k: float(np.mean([prior[e][k] for e in keys])) for k in metrics} if cohort else None,
            "difference": {
                k: _paired([current[e][k] for e in keys], [prior[e][k] for e in keys])
                for k in metrics
            } if cohort else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gibson", "mp3d"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_gibson(args.run_root) if args.dataset == "gibson" else summarize_mp3d(args.run_root)
    summary["run_root"] = str(args.run_root)
    destination = args.run_root / "summary_waveopt.json"
    destination.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
