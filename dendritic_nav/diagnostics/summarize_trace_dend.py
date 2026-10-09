"""Compare captured decision traces against the original pilot identities/hashes."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    with np.load(path,allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    active = data["action"] >= 0
    indices = np.flatnonzero(active)
    selected = data["action"][active].astype(int)
    goal_steps = np.linalg.norm(np.diff(data["goal_xy"][active],axis=0),axis=1)
    candidates = data["candidates"][indices,selected]
    agent = data["agent_xy"]
    cells = np.floor(agent/.25).astype(int)
    separation = np.linalg.norm(data["rc"][indices,selected]-data["rc"][indices,0],axis=1)*.05
    anchor_gap = data["spatial"][indices,0]-data["spatial"][indices,selected]
    return dict(steps=len(agent), decisions=int(active.sum()),
                non_anchor=int((selected!=0).sum()), switches=int(data["goal_switch"].sum()),
                large_goal_jumps=int((goal_steps>2).sum()),collisions=int(data["collision"].sum()),
                unique_25cm_agent_cells=len(set(map(tuple,cells))),
                selected_explored_gt_half=int((candidates[:,7]>=.5).sum()),
                selected_obstacle_gt_half=int((candidates[:,6]>=.5).sum()),
                mean_anchor_separation=float(separation.mean()) if len(separation) else 0.,
                mean_spatial_sacrifice=float(anchor_gap.mean()) if len(anchor_gap) else 0.,
                visible_override_steps=int(data["override"].sum()),
                final_xy=agent[-1].tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--experiment",required=True)
    parser.add_argument("--traces",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();old=Path(args.experiment);root=Path(args.traces)
    rows=[]
    for folder in sorted(root.glob("*_actor_*")):
        if not folder.is_dir():continue
        baseline=folder.name.startswith("baseline")
        actor=int(folder.name.rsplit("_",1)[1])
        prior=old/("dev_baseline_{:03d}" if baseline else "dev_dend_{:03d}").format(actor)
        original={e["episode_id"]:e for e in map(json.loads,(prior/"episodes.jsonl").read_text().splitlines())}
        metrics_file=folder/"episodes.jsonl"
        if not metrics_file.exists():continue
        for e in map(json.loads,metrics_file.read_text().splitlines()):
            previous=original[e["episode_id"]]
            path=folder/"trace_{:04d}.npz".format(e["episode"])
            row=dict(actor=actor,baseline=baseline,episode=e["episode"],episode_id=e["episode_id"],
                     success=e["success"],spl=e["spl"],
                     same_trajectory_hash=e["trajectory_sha256"]==previous["trajectory_sha256"],
                     original_success=previous["success"],statistics=summarize(path))
            rows.append(row)
    result=dict(rows=rows,all_reproduced_exactly=all(e["same_trajectory_hash"] for e in rows))
    with open(args.output,"x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
