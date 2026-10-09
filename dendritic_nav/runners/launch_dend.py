"""Launch tracked child jobs, with a conservative GPU admission check.

Uses available memory for real actors, not allocation-only occupancy jobs.
Does not stop, kill or reconfigure any existing process.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def launch(command, gpu, log, cwd):
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        GLOG_minloglevel="2",
        MAGNUM_LOG="quiet",
        PYTHONPATH=str(cwd) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    with open(log, "xb") as stream:
        child = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return dict(pid=child.pid, gpu=gpu, command=command, log=str(log))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--gpus", default="1,2,3,4,5,7,0,6")
    p.add_argument("--actors", type=int, default=16)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--learner-gpu", type=int, default=7)
    p.add_argument("--actor-budget-mib", type=int, default=14000)
    p.add_argument("--reserve-mib", type=int, default=10000)
    p.add_argument("--no-learner", action="store_true")
    p.add_argument("--role", choices=("train", "dev", "val"), default="train")
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--with-dev-pipeline", action="store_true")
    p.add_argument("--dev-actors", type=int, default=2)
    p.add_argument("--baseline-first", action="store_true")
    args = p.parse_args()
    experiment = Path(args.experiment).resolve()
    root = Path(__file__).resolve().parents[2]
    launch_file = experiment / "launch_{}.json".format(
        args.role + ("_baseline" if args.baseline else "")
    )
    if launch_file.exists():
        raise FileExistsError(str(launch_file))
    scenes = json.loads((experiment / "scene_split.json").read_text())
    if args.role == "train" and args.actors > len(scenes[args.role]):
        raise ValueError("Too many actors for split")
    if args.role != "train" and not args.no_learner:
        raise ValueError("Evaluation must set --no-learner")
    if args.with_dev_pipeline and (args.role != "train" or args.no_learner):
        raise ValueError("Dev pipeline requires a real training learner")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = {
        int(line.split(",")[0]): int(line.split(",")[1])
        for line in output.strip().splitlines()
    }
    gpus = [int(g) for g in args.gpus.split(",")]
    assignments = []
    remaining = dict(free)
    # Account for a learner even though its parameter footprint is tiny.
    if not args.no_learner:
        remaining[args.learner_gpu] -= 4000
    for i in range(args.actors):
        usable = [
            g for g in gpus if remaining[g] >= args.actor_budget_mib + args.reserve_mib
        ]
        if not usable:
            raise RuntimeError(
                "Insufficient free VRAM for requested real actors; lower --actors"
            )
        gpu = max(usable, key=lambda g: remaining[g])
        remaining[gpu] -= args.actor_budget_mib
        assignments.append(gpu)
    jobs = []
    logs = experiment / ("logs_" + args.role + ("_baseline" if args.baseline else ""))
    logs.mkdir(exist_ok=False)
    # Incremental, durable launch inventory: partial launches remain auditable.
    def record():
        launch_file.write_text(
            json.dumps(
                dict(created=time.time(), initial_free_mib=free, jobs=jobs), indent=2
            )
        )

    record()
    if not args.no_learner:
        command = [
            sys.executable,
            "-u",
            "-m",
            "dendritic_nav.runners.train_dend",
            "--experiment",
            str(experiment),
            "--steps",
            str(args.steps),
        ]
        jobs.append(launch(command, args.learner_gpu, logs / "learner.log", root))
        record()
    for i, gpu in enumerate(assignments):
        name = (
            "actor_{:03d}".format(i)
            if args.role == "train"
            else "{}_{}_{:03d}".format(
                args.role, "baseline" if args.baseline else "dend", i
            )
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "dendritic_nav.runners.collect_dend",
            "--experiment",
            str(experiment),
            "--output",
            str(experiment / name),
            "--actor-id",
            str(i),
            "--actors",
            str(args.actors),
            "--steps",
            str((args.steps + args.actors - 1) // args.actors),
            "--role",
            args.role,
            "--seed",
            str(123 + i),
        ]
        if args.episodes:
            command.extend(
                ["--episodes", str((args.episodes + args.actors - 1) // args.actors)]
            )
        if args.checkpoint:
            command.extend(["--checkpoint", str(Path(args.checkpoint).resolve())])
        if args.baseline:
            command.append("--baseline")
        jobs.append(launch(command, gpu, logs / (name + ".log"), root))
        record()
        time.sleep(1)
    if args.with_dev_pipeline:
        command = [
            sys.executable,
            "-u",
            "-m",
            "dendritic_nav.runners.pipeline_dend",
            "--experiment",
            str(experiment),
            "--gpus",
            args.gpus,
            "--dev-actors",
            str(args.dev_actors),
        ]
        if args.baseline_first:
            command.append("--baseline-first")
        jobs.append(launch(command, args.learner_gpu, logs / "pipeline.log", root))
        record()
    print(
        json.dumps(
            dict(launch=str(launch_file), jobs=len(jobs), assignments=assignments)
        )
    )


if __name__ == "__main__":
    main()
