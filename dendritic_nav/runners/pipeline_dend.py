"""Finite pilot -> paired dev -> report pipeline; no indefinite occupancy jobs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .report_dend import compare, load_episodes


def live(pid):
    path = Path("/proc") / str(pid) / "stat"
    if not path.exists():
        return False
    try:
        return path.read_text().split(")", 1)[1].strip().split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_phase(root, launch_name):
    launch = json.loads((root / launch_name).read_text())
    while True:
        pending = []
        for job in launch["jobs"]:
            cmd = job["command"]
            if "dendritic_nav.runners.pipeline_dend" in cmd:
                continue
            if "dendritic_nav.runners.train_dend" in cmd:
                completion = root / "learner/completion.json"
            else:
                completion = Path(cmd[cmd.index("--output") + 1]) / "completion.json"
            if completion.exists():
                continue
            if not live(job["pid"]):
                raise RuntimeError("Job exited without completion: " + job["log"])
            pending.append(job["pid"])
        (root / "pipeline_progress.json").write_text(
            json.dumps(dict(phase=launch_name, pending=pending, time=time.time()))
        )
        if not pending:
            return
        time.sleep(30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,7")
    p.add_argument("--dev-episodes", type=int, default=100)
    p.add_argument("--dev-actors", type=int, default=2)
    p.add_argument("--baseline-first", action="store_true")
    args = p.parse_args()
    root = Path(args.experiment).resolve()
    try:
        base = [
            sys.executable,
            "-m",
            "dendritic_nav.runners.launch_dend",
            "--experiment",
            str(root),
            "--gpus",
            args.gpus,
            "--role",
            "dev",
            "--no-learner",
            "--actors",
            str(args.dev_actors),
            "--steps",
            str(args.dev_episodes * 500 + 1000),
            "--episodes",
            str(args.dev_episodes),
        ]
        def ensure_eval(baseline=False):
            name = "launch_dev_baseline.json" if baseline else "launch_dev.json"
            if (root / name).exists():
                jobs = json.loads((root / name).read_text())["jobs"]
                if len(jobs) != args.dev_actors:
                    raise ValueError("Existing evaluation has a different actor count")
                return
            extra = ["--baseline"] if baseline else ["--checkpoint", str(checkpoint)]
            subprocess.run(base + extra, check=True)

        checkpoint = root / "learner/checkpoints/pilot_final.ckpt"
        if args.baseline_first:
            # The initial admission check sees only resident VRAM. Wait until
            # all training actors have executed an episode before admitting a
            # second wave, so still-loading actors cannot be overcommitted.
            training_jobs = json.loads((root / "launch_train.json").read_text())["jobs"]
            collectors = [job for job in training_jobs
                          if "dendritic_nav.runners.collect_dend" in job["command"]]
            while True:
                ready = True
                for job in collectors:
                    command = job["command"]
                    directory = Path(command[command.index("--output") + 1])
                    if (directory / "episodes.jsonl").exists():
                        continue
                    if not live(job["pid"]):
                        raise RuntimeError("Actor failed before first episode: " + job["log"])
                    ready = False
                if ready:
                    break
                time.sleep(30)
            ensure_eval(baseline=True)
        wait_phase(root, "launch_train.json")
        ensure_eval()
        ensure_eval(baseline=True)
        wait_phase(root, "launch_dev.json")
        wait_phase(root, "launch_dev_baseline.json")
        result = compare(
            load_episodes(sorted(root.glob("dev_baseline_*/episodes.jsonl"))),
            load_episodes(sorted(root.glob("dev_dend_*/episodes.jsonl"))),
        )
        result["checkpoint"] = str(checkpoint)
        if result["point_estimate_non_regression"]:
            destination = root / "learner/checkpoints/best_dev.ckpt"
            shutil.copy2(checkpoint, destination)
            result["qualified_best_dev"] = str(destination)
        result[
            "next_step"
        ] = "Review pilot and visualizations before long training / MP3D integration."
        with (root / "development_comparison.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2), flush=True)
        (root / "pipeline_complete.json").write_text(
            json.dumps(dict(time=time.time(), result=result), indent=2)
        )
    except Exception as exc:
        (root / "pipeline_failure.json").write_text(
            json.dumps(dict(time=time.time(), error=str(exc)))
        )
        raise


if __name__ == "__main__":
    main()
