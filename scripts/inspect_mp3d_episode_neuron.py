"""Read-only single-episode MP3D plan/action audit summary."""

import argparse
import glob
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = glob.glob(str(args.run_root / f"audit_val_part_{args.part}/*.jsonl"))
    if len(files) != 1:
        parser.error(f"expected one audit, got {files}")
    active = False
    plans = []
    actions = []
    poses = []
    for line in open(files[0]):
        event = json.loads(line)
        if event["type"] == "episode_start":
            active = int(event["episode_index"]) == args.episode
            continue
        if not active:
            continue
        if event["type"] == "plan":
            item = event["selected"]
            info = item["info"]
            plans.append({
                "seq": event["seq"], "stg": item["stg"], "stop": item["stop"],
                "replan": item["replan"], "turn_action": item["turn_action"],
                "old_stg": info.get("tie_old_stg"),
                "tie_changed_parents": info.get("tie_changed_parents"),
                "tie_old_bends": info.get("tie_old_bends"),
                "tie_new_bends": info.get("tie_new_bends"),
                "no_path": info.get("no_path"),
            })
        elif event["type"] == "action":
            actions.append({"seq": event["seq"], "action": event["action"]})
        elif event["type"] == "pose":
            poses.append({"seq": event["seq"], "xy": event["pose_pred"][:2]})
    result = {
        "part": args.part, "episode": args.episode,
        "plans": len(plans), "actions": len(actions), "poses": len(poses),
        "stop_calls": sum(p["stop"] for p in plans),
        "replan_calls": sum(p["replan"] for p in plans),
        "no_path_calls": sum(bool(p["no_path"]) for p in plans),
        "action_counts": {str(k): sum(a["action"] == k for a in actions) for k in (0, 1, 2, 3)},
        "first_plans": plans[:8], "last_plans": plans[-8:],
        "first_pose": poses[0] if poses else None,
        "last_pose": poses[-1] if poses else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
