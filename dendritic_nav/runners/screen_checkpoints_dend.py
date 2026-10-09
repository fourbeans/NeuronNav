"""Evaluate saved v3 snapshots on exactly the existing paired dev episodes.

Runs checkpoint groups in bounded waves. It never trains, changes a model, or
stops another process; each evaluator has a fresh, auditable output directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from dendritic_nav.core.cell_dend import MODEL_FAMILY
from .common_dend import new_run, sha256
from .launch_dend import launch
from .pipeline_dend import live
from .report_dend import compare, load_episodes


def free_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"], text=True
    )
    return {int(index): int(free) for index, free in
            (line.split(",") for line in output.strip().splitlines())}


def assign_gpus(count, gpus, budget, reserve):
    remaining = free_memory()
    assignments = []
    for _ in range(count):
        available = [g for g in gpus if remaining[g] >= budget + reserve]
        if not available:
            raise RuntimeError("Insufficient free VRAM for the checkpoint wave")
        gpu = max(available, key=lambda g: remaining[g])
        assignments.append(gpu)
        remaining[gpu] -= budget
    return assignments


def wait_jobs(jobs):
    while True:
        pending = []
        for job in jobs:
            complete = Path(job["completion"]).is_file()
            running = live(job["pid"])
            # Wait for the process to release CUDA memory, not just for its
            # completion marker, before admitting the next evaluator wave.
            if complete and not running:
                continue
            if not complete and not running:
                raise RuntimeError("Evaluator exited without completion: " + job["log"])
            pending.append(job["pid"])
        if not pending:
            return
        time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--parallel-checkpoints", type=int, default=3)
    parser.add_argument("--actor-budget-mib", type=int, default=14000)
    parser.add_argument("--reserve-mib", type=int, default=10000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.parallel_checkpoints < 1:
        raise ValueError("parallel-checkpoints must be positive")
    experiment = Path(args.experiment).resolve()
    if args.resume:
        output = Path(args.output).resolve()
        if not output.is_dir() or (output / "completion.json").exists():
            raise ValueError("Resume requires an unfinished existing output directory")
        inventory = json.loads((output / "launch.json").read_text())
        results = json.loads((output / "results.json").read_text())
        with (output / "resumes.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(time=time.time(), args=vars(args))) + "\n")
    else:
        output = new_run(args.output, args,
                         "dendritic_nav.runners.screen_checkpoints_dend")
        inventory = dict(created=time.time(), jobs=[])
        results = {}
    root = Path(__file__).resolve().parents[2]
    baseline = load_episodes(sorted(experiment.glob("dev_baseline_*/episodes.jsonl")))
    if len(baseline) != 100:
        raise ValueError("Expected exactly 100 already-paired baseline episodes")
    checkpoints = sorted((experiment / "learner/checkpoints").glob("update_*.ckpt"))
    if not checkpoints:
        raise ValueError("No intermediate checkpoints found")
    import torch
    for checkpoint in checkpoints:
        metadata = torch.load(str(checkpoint), map_location="cpu")
        if metadata.get("model_family") != MODEL_FAMILY:
            raise ValueError("Wrong checkpoint family: " + str(checkpoint))
    gpus = [int(item) for item in args.gpus.split(",")]
    inventory["checkpoints"] = [str(p) for p in checkpoints]
    remaining = [p for p in checkpoints if p.stem not in results]
    for start in range(0, len(remaining), args.parallel_checkpoints):
        wave = remaining[start:start + args.parallel_checkpoints]
        assigned = assign_gpus(10 * len(wave), gpus, args.actor_budget_mib,
                               args.reserve_mib)
        wave_jobs = []
        for group_index, checkpoint in enumerate(wave):
            group = output / checkpoint.stem
            group.mkdir(exist_ok=False)
            for actor in range(10):
                name = "actor_{:03d}".format(actor)
                command = [
                    sys.executable, "-u", "-m", "dendritic_nav.runners.collect_dend",
                    "--experiment", str(experiment), "--output", str(group / name),
                    "--role", "dev", "--actor-id", str(actor), "--actors", "10",
                    "--episodes", "10", "--steps", "5100", "--seed", str(123 + actor),
                    "--checkpoint", str(checkpoint), "--visualize-every", "1000000",
                ]
                gpu = assigned[10 * group_index + actor]
                job = launch(command, gpu, group / (name + ".log"), root)
                job["completion"] = str(group / name / "completion.json")
                wave_jobs.append(job)
                inventory["jobs"].append(job)
                (output / "launch.json").write_text(json.dumps(inventory, indent=2))
                time.sleep(0.5)
        wait_jobs(wave_jobs)
        for checkpoint in wave:
            episodes = load_episodes(sorted((output / checkpoint.stem).glob(
                "actor_*/episodes.jsonl")))
            result = compare(baseline, episodes)
            result["checkpoint"] = str(checkpoint)
            result["sha256"] = sha256(checkpoint)
            results[checkpoint.stem] = result
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print("SCREEN " + json.dumps(dict(name=checkpoint.stem, result=result)),
                  flush=True)
    (output / "completion.json").write_text(json.dumps(dict(time=time.time(),
                                                            results=results), indent=2))


if __name__ == "__main__":
    main()
