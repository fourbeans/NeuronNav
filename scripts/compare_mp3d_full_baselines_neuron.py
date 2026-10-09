"""Paired MP3D full-run comparison of old strict SNN and weighted A*."""

import argparse
import json
from pathlib import Path

import numpy as np

from summarize_waveopt_neuron import JOINT, SIGMA_NAV, _mp3d, _paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snn, snn_counts = _mp3d(JOINT / "mp3d_seed123_best2k")
    astar, astar_counts = _mp3d(SIGMA_NAV / "mp3d_seed123_best2k_4gpu")
    if set(snn) != set(astar):
        raise ValueError("full MP3D episode ID sets differ")
    keys = sorted(snn)
    if any(abs(snn[k]["goal_distance"] - astar[k]["goal_distance"]) > 1e-6
           for k in keys):
        raise ValueError("full MP3D goal distances differ")
    result = {
        "episodes": len(keys), "snn_part_counts": snn_counts,
        "astar_part_counts": astar_counts,
        "metrics": {},
        "success_snn_only": sum(snn[k]["success"] > astar[k]["success"] for k in keys),
        "success_astar_only": sum(snn[k]["success"] < astar[k]["success"] for k in keys),
    }
    snn_only = [k for k in keys if snn[k]["success"] > astar[k]["success"]]
    astar_only = [k for k in keys if snn[k]["success"] < astar[k]["success"]]
    result["success_flip_endpoints"] = {
        "snn_only_astar_median_remaining_m": float(np.median(
            [astar[k]["distance_to_goal"] for k in snn_only])),
        "astar_only_snn_median_remaining_m": float(np.median(
            [snn[k]["distance_to_goal"] for k in astar_only])),
        "astar_only_snn_within_0_1m": sum(
            snn[k]["distance_to_goal"] < 0.1 for k in astar_only),
        "astar_only_snn_within_0_25m": sum(
            snn[k]["distance_to_goal"] < 0.25 for k in astar_only),
        "astar_only_snn_within_0_5m": sum(
            snn[k]["distance_to_goal"] < 0.5 for k in astar_only),
        "astar_only_snn_within_1m": sum(
            snn[k]["distance_to_goal"] < 1.0 for k in astar_only),
        "astar_only_snn_within_2m": sum(
            snn[k]["distance_to_goal"] < 2.0 for k in astar_only),
        "astar_only_snn_over_5m": sum(
            snn[k]["distance_to_goal"] > 5.0 for k in astar_only),
    }
    for metric in ("success", "spl", "distance_to_goal"):
        new = [snn[k][metric] for k in keys]
        old = [astar[k][metric] for k in keys]
        result["metrics"][metric] = {
            "snn": float(np.mean(new)), "astar": float(np.mean(old)),
            "snn_minus_astar": _paired(new, old),
        }
    distances = np.array([snn[k]["goal_distance"] for k in keys])
    edges = np.quantile(distances, [0, 0.25, 0.5, 0.75, 1])
    result["goal_distance_quartiles"] = []
    for i in range(4):
        selected = [k for k, d in zip(keys, distances)
                    if (edges[i] <= d < edges[i + 1] if i < 3
                        else edges[i] <= d <= edges[i + 1])]
        result["goal_distance_quartiles"].append({
            "min_m": float(edges[i]), "max_m": float(edges[i + 1]),
            "episodes": len(selected),
            "snn_success": float(np.mean([snn[k]["success"] for k in selected])),
            "astar_success": float(np.mean([astar[k]["success"] for k in selected])),
            "snn_spl": float(np.mean([snn[k]["spl"] for k in selected])),
            "astar_spl": float(np.mean([astar[k]["spl"] for k in selected])),
            "snn_only_success": sum(snn[k]["success"] > astar[k]["success"]
                                    for k in selected),
            "astar_only_success": sum(snn[k]["success"] < astar[k]["success"]
                                      for k in selected),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
