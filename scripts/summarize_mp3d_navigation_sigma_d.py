#!/usr/bin/env python3
"""Wait for and aggregate all MP3D ObjectNav validation-part statistics."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path


PRIMARY_METRICS = ("success", "spl", "distance_to_goal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-parts", type=int, default=11)
    parser.add_argument("--expected-episodes", type=int, default=2195)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument(
        "--model-family",
        default="sigma-d",
        help="label stored in the report schema, e.g. sigma-d or original-poni",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    deadline = time.time() + args.timeout_hours * 3600.0
    pattern = "tb_seed_100_val_part_*/stats.json"

    while True:
        paths = sorted(run_root.glob(pattern))
        if len(paths) == args.expected_parts:
            break
        if time.time() >= deadline:
            raise TimeoutError(
                f"found {len(paths)}/{args.expected_parts} stats files under {run_root}"
            )
        time.sleep(args.poll_seconds)

    all_episodes = {}
    part_counts = {}
    for path in paths:
        with path.open() as handle:
            part = json.load(handle)
        part_name = path.parent.name.replace("tb_seed_100_", "")
        part_counts[part_name] = len(part)
        overlap = set(all_episodes).intersection(part)
        if overlap:
            raise RuntimeError(f"duplicate episode keys in {path}: {sorted(overlap)[:3]}")
        all_episodes.update(part)

    if len(all_episodes) != args.expected_episodes:
        raise RuntimeError(
            f"episode mismatch: {len(all_episodes)} != {args.expected_episodes}"
        )

    metrics = {}
    for metric in PRIMARY_METRICS:
        values = [float(row[metric]) for row in all_episodes.values()]
        mean = statistics.fmean(values)
        std = statistics.pstdev(values)
        metrics[metric] = {
            "mean": mean,
            "std": std,
            "standard_error": std / math.sqrt(len(values)),
        }

    report = {
        "schema": f"mp3d-{args.model_family}-navigation-summary-v1",
        "run_root": str(run_root),
        "episodes": len(all_episodes),
        "parts": len(paths),
        "part_episode_counts": part_counts,
        "metrics": metrics,
        "source_stats": [str(path) for path in paths],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
