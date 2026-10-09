#!/usr/bin/env python3
"""Summarize the node1 independent SNN-wave experiments.

This reads result/audit files only.  It does not modify checkpoints or runs.
"""

import argparse
import glob
import json
import math
import os
from collections import Counter

import numpy as np


METRICS = ("success", "spl", "softspl", "dts", "distance_to_goal", "reward")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def avg(values):
    return float(np.mean(values)) if values else None


def means(rows, metrics):
    return {key: avg([row[key] for row in rows]) for key in metrics}


def bootstrap_ci(differences, seed=20260923, samples=10000):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        rng.choice(values, len(values), replace=True).mean() for _ in range(samples)
    ])
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def exact_mcnemar_p(ref_success, snn_success):
    ref = np.asarray(ref_success)
    snn = np.asarray(snn_success)
    ref_only = int(((ref == 1) & (snn == 0)).sum())
    snn_only = int(((ref == 0) & (snn == 1)).sum())
    n = ref_only + snn_only
    if n == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(ref_only, snn_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2 ** n))
    return {"reference_only": ref_only, "snn_only": snn_only, "pvalue": pvalue}


def audit_summary(files):
    counts = Counter()
    models = Counter()
    no_path_reasons = Counter()
    elapsed = []
    spikes = []
    events = []
    for path in files:
        with open(path) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] += 1
                    continue
                record_type = record.get("type", "unknown")
                counts[record_type] += 1
                if record_type != "plan":
                    continue
                counts["used_neuron"] += int(record.get("used") == "neuron")
                counts["used_reference"] += int(record.get("used") == "reference")
                counts["reference_nonnull"] += int(record.get("reference") is not None)
                counts["fallback_true"] += int(bool(record.get("fallback")))
                counts["error_nonnull"] += int(record.get("error") is not None)
                selected = record.get("selected") or record.get("candidate") or {}
                info = selected.get("info") or {}
                if info.get("no_path"):
                    counts["no_path"] += 1
                    reason = info.get("no_path_reason") or info.get("no_path_policy")
                    no_path_reasons[str(reason)] += 1
                if "distance_model" in info:
                    models[str(info["distance_model"])] += 1
                if "elapsed_ms" in record:
                    elapsed.append(float(record["elapsed_ms"]))
                if "spikes" in info:
                    spikes.append(float(info["spikes"]))
                if "events_emitted" in info:
                    events.append(float(info["events_emitted"]))

    def distribution(values):
        if not values:
            return None
        values = np.asarray(values)
        return {
            "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(values.max()),
        }

    plans = counts["plan"]
    return {
        "files": len(files),
        "counts": dict(counts),
        "distance_models": dict(models),
        "no_path_reasons": dict(no_path_reasons),
        "no_path_rate": counts["no_path"] / plans if plans else None,
        "elapsed_ms": distribution(elapsed),
        "spikes": distribution(spikes),
        "events_emitted": distribution(events),
    }


def gibson_summary(root):
    reference_path = os.path.join(
        root, "gibson_ref123_gpu1/dump/seed_100/final_eval_stats.json"
    )
    reference = load_json(reference_path)["total_raw_metrics"]
    strict_files = sorted(glob.glob(os.path.join(
        root, "gibson_strict123_gpu*/*/dump/seed_100/final_eval_stats.json"
    )))
    strict_all = {key: [] for key in reference}
    paired = {key: [] for key in ("success", "spl", "dts")}
    scenes = {}
    episode_groups = {"without_no_path": [], "with_no_path": []}

    for stats_path in strict_files:
        scene = stats_path.split(os.sep)[-4]
        strict = load_json(stats_path)["total_raw_metrics"]
        chunk = next(
            idx for idx in range(5)
            if np.array_equal(
                np.asarray(reference["goal_distance"][idx * 200:(idx + 1) * 200]),
                np.asarray(strict["goal_distance"]),
            )
        )
        ref_scene = {
            key: reference[key][chunk * 200:(chunk + 1) * 200] for key in reference
        }
        for key, values in strict.items():
            strict_all[key].extend(values)
        for key in paired:
            paired[key].extend(np.asarray(strict[key]) - np.asarray(ref_scene[key]))

        audit_path = glob.glob(stats_path.split("/dump/")[0] + "/audit/*.jsonl")[0]
        current_episode = None
        plans = [0] * 200
        no_path = [0] * 200
        with open(audit_path) as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("type") == "episode_start":
                    current_episode = record["episode_no"] - 1
                elif (record.get("type") == "plan" and current_episode is not None
                      and 0 <= current_episode < 200):
                    plans[current_episode] += 1
                    info = (record.get("selected") or {}).get("info") or {}
                    no_path[current_episode] += int(bool(info.get("no_path")))

        for idx in range(200):
            group = "with_no_path" if no_path[idx] else "without_no_path"
            episode_groups[group].append({
                "reference_success": ref_scene["success"][idx],
                "snn_success": strict["success"][idx],
                "reference_spl": ref_scene["spl"][idx],
                "snn_spl": strict["spl"][idx],
            })
        scenes[scene] = {
            "episodes": 200,
            "reference": {key: avg(ref_scene[key]) for key in ("success", "spl", "dts")},
            "strict_snn": {key: avg(strict[key]) for key in ("success", "spl", "dts")},
            "delta": {
                key: avg(strict[key]) - avg(ref_scene[key]) for key in ("success", "spl", "dts")
            },
            "plans": sum(plans),
            "no_path_queries": sum(no_path),
            "no_path_query_rate": sum(no_path) / sum(plans),
            "episodes_with_no_path": sum(value > 0 for value in no_path),
        }

    reference_means = {key: avg(reference[key]) for key in ("success", "spl", "dts")}
    strict_means = {key: avg(strict_all[key]) for key in ("success", "spl", "dts")}
    paired_stats = {}
    for offset, key in enumerate(paired):
        values = np.asarray(paired[key])
        paired_stats[key] = {
            "mean_delta": float(values.mean()),
            "bootstrap_ci95": bootstrap_ci(values, seed=20260923 + offset),
            "wins": int((values > 1e-12).sum()),
            "ties": int((np.abs(values) <= 1e-12).sum()),
            "losses": int((values < -1e-12).sum()),
        }
    paired_stats["success_mcnemar"] = exact_mcnemar_p(
        reference["success"], strict_all["success"]
    )

    grouped = {}
    for group, rows in episode_groups.items():
        grouped[group] = {
            "episodes": len(rows),
            **means(rows, ("reference_success", "snn_success", "reference_spl", "snn_spl")),
        }
    return {
        "episodes": len(strict_all["success"]),
        "reference": reference_means,
        "strict_snn": strict_means,
        "delta": {key: strict_means[key] - reference_means[key] for key in reference_means},
        "paired_statistics": paired_stats,
        "scenes": scenes,
        "no_path_episode_groups": grouped,
        "strict_audit": audit_summary(glob.glob(os.path.join(
            root, "gibson_strict123_gpu*/*/audit/*.jsonl"
        ))),
        "legacy_audit": audit_summary(glob.glob(os.path.join(
            root, "pilot_wave_gpu0/neuron/audit/*.jsonl"
        ))),
    }


def load_mp3d(pattern):
    result = {}
    files = glob.glob(pattern)
    for path in files:
        part = load_json(path)
        overlap = set(result) & set(part)
        if overlap:
            raise RuntimeError("Duplicate MP3D episode IDs: " + str(len(overlap)))
        result.update(part)
    return result, files


def mp3d_summary(root):
    reference, reference_files = load_mp3d(os.path.join(
        root, "mp3d_ref123_gpu6/val_part_*/tb/stats.json"
    ))
    strict, strict_files = load_mp3d(os.path.join(
        root, "mp3d_strict123_gpu*/val_part_*/tb/stats.json"
    ))
    if set(reference) != set(strict):
        raise RuntimeError("Reference and strict MP3D episode IDs differ")
    keys = sorted(reference)
    metric_names = ("success", "spl", "softspl", "distance_to_goal", "reward")
    reference_means = means(list(reference.values()), metric_names)
    strict_means = means(list(strict.values()), metric_names)
    paired = {}
    for offset, metric in enumerate(metric_names):
        delta = np.asarray([strict[key][metric] - reference[key][metric] for key in keys])
        paired[metric] = {
            "mean_delta": float(delta.mean()),
            "bootstrap_ci95": bootstrap_ci(delta, seed=20261000 + offset),
            "wins": int((delta > 1e-12).sum()),
            "ties": int((np.abs(delta) <= 1e-12).sum()),
            "losses": int((delta < -1e-12).sum()),
        }
    paired["success_mcnemar"] = exact_mcnemar_p(
        [reference[key]["success"] for key in keys],
        [strict[key]["success"] for key in keys],
    )

    parts = {}
    exact_legacy_parts = 0
    for part_id in range(11):
        ref_path = os.path.join(root, f"mp3d_ref123_gpu6/val_part_{part_id}/tb/stats.json")
        strict_path = glob.glob(os.path.join(
            root, f"mp3d_strict123_gpu*/val_part_{part_id}/tb/stats.json"
        ))[0]
        legacy_path = os.path.join(
            root, f"mp3d_wave123_gpu7/val_part_{part_id}/tb/stats.json"
        )
        ref_part = load_json(ref_path)
        strict_part = load_json(strict_path)
        parts[str(part_id)] = {
            "episodes": len(ref_part),
            "reference": means(list(ref_part.values()), ("success", "spl", "softspl")),
            "strict_snn": means(list(strict_part.values()), ("success", "spl", "softspl")),
        }
        if open(strict_path, "rb").read() == open(legacy_path, "rb").read():
            exact_legacy_parts += 1
    return {
        "episodes": len(keys),
        "reference_parts": len(reference_files),
        "strict_parts": len(strict_files),
        "same_episode_ids": True,
        "reference": reference_means,
        "strict_snn": strict_means,
        "delta": {key: strict_means[key] - reference_means[key] for key in metric_names},
        "paired_statistics": paired,
        "parts": parts,
        "strict_vs_legacy_byte_identical_parts": exact_legacy_parts,
        "strict_audit": audit_summary(glob.glob(os.path.join(
            root, "mp3d_strict123_gpu*/val_part_*/audit/*.jsonl"
        ))),
        "legacy_audit": audit_summary(glob.glob(os.path.join(
            root, "mp3d_wave123_gpu7/val_part_*/audit/*.jsonl"
        ))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = {
        "schema": "independent_snn_wavefront_summary_v1",
        "root": os.path.abspath(args.root),
        "gibson": gibson_summary(args.root),
        "mp3d": mp3d_summary(args.root),
    }
    with open(args.output, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
