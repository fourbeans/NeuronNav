"""Official-val comparison that never counts exact scene reruns as new episodes."""
import argparse
import json
from pathlib import Path
from .report_dend import compare, load_episodes


def collapse_exact_runs(pairs):
    """Require exact within-ID reruns, then retain one independent episode.

    A cycling scene shard can restart from episode zero for every actor. Its
    500 records may therefore contain only 125 distinct evaluation episodes.
    If repeats differ, refuse to silently treat them as independent samples.
    """
    baseline, dendritic = {}, {}
    count = 0
    fields = ("scene", "success", "spl", "trajectory_sha256")
    for a, b in pairs:
        if a["episode_id"] != b["episode_id"] or a["scene"] != b["scene"]:
            raise ValueError("Baseline and dendritic actor streams are not paired")
        key = a["episode_id"]
        count += 1
        for seen, row in ((baseline, a), (dendritic, b)):
            if key in seen:
                if any(seen[key][field] != row[field] for field in fields):
                    raise ValueError("Repeated episode ID has different trajectory or metrics")
            else:
                seen[key] = row
    return list(baseline.values()), list(dendritic.values()), dict(
        recorded_pair_runs=count,
        distinct_episode_ids=len(baseline),
        exact_duplicate_pair_runs=count - len(baseline),
    )


def load_paired_actor_runs(root):
    baseline = sorted(root.glob("val_baseline_*/episodes.jsonl"))
    dendritic = sorted(root.glob("val_dend_*/episodes.jsonl"))
    if not baseline or len(baseline) != len(dendritic):
        raise ValueError("Official-val actor arms are incomplete")
    pairs = []
    for base_path in baseline:
        actor_id = base_path.parent.name.rsplit("_", 1)[1]
        dend_path = root / ("val_dend_" + actor_id) / "episodes.jsonl"
        if not dend_path.is_file():
            raise ValueError("Missing dendritic actor " + actor_id)
        if not (base_path.parent / "completion.json").is_file() or not (
            dend_path.parent / "completion.json"
        ).is_file():
            raise ValueError("Official-val actor not complete: " + actor_id)
        a = load_episodes([base_path])
        b = load_episodes([dend_path])
        if len(a) != len(b):
            raise ValueError("Different episode counts for actor " + actor_id)
        pairs.extend(zip(a, b))
    return collapse_exact_runs(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    baseline, dendritic, sample_audit = load_paired_actor_runs(root)
    result = compare(baseline, dendritic)
    result.update(
        split="gibson_official_val",
        checkpoint=str(checkpoint),
        sample_audit=sample_audit,
        interpretation=(
            "Paired official validation uses distinct episode IDs only; "
            "exact actor reruns are not independent samples."
        ),
    )
    with (root / "official_val_comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
