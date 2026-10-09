"""Compare deterministic MP3D pilot action/waypoint traces across code versions."""

import argparse
import glob
import json
from pathlib import Path


def digest(root, part):
    files = glob.glob(str(root / f"audit_val_part_{part}/*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one audit for part {part}, got {files}")
    records = []
    for line in open(files[0]):
        event = json.loads(line)
        if event["type"] == "episode_start":
            records.append(("episode", event["episode_index"]))
        elif event["type"] == "plan":
            chosen = event["selected"]
            info = chosen["info"]
            records.append(("plan", chosen["stg"], chosen["stop"], chosen["replan"],
                            chosen["turn_action"], info.get("readout_applied"),
                            info.get("readout_path_index"), info.get("no_path")))
        elif event["type"] == "action":
            records.append(("action", event["action"]))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    counts = {}
    for part in range(11):
        first, second = digest(args.before, part), digest(args.after, part)
        counts[str(part)] = {
            "before_records": len(first), "after_records": len(second),
            "exact_match": first == second,
            "first_difference": next((i for i, (a, b) in enumerate(zip(first, second)) if a != b),
                                     None),
        }
    result = {
        "all_parts_match": all(row["exact_match"] for row in counts.values()),
        "per_part": counts,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
