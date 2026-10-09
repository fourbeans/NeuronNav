"""Launch a traced official-val episode prefix without altering the policy."""
import argparse
import json
from pathlib import Path
import sys

from dendritic_nav.runners.launch_dend import launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--actors", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--gpus", required=True, help="Two comma-separated GPU indices")
    args = parser.parse_args()
    gpus = [int(item) for item in args.gpus.split(",")]
    if len(gpus) != 2 or args.episodes < 1:
        raise ValueError("Need two GPUs and a positive episode prefix")
    experiment = Path(args.experiment).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for baseline, gpu in zip((True, False), gpus):
        label = "baseline" if baseline else "dend"
        name = "{}_actor_{:03d}".format(label, args.actor_id)
        command = [
            sys.executable, "-u", "-m", "dendritic_nav.diagnostics.trace_dend",
            "--experiment", str(experiment), "--output", str(output / name),
            "--role", "val", "--actor-id", str(args.actor_id),
            "--actors", str(args.actors), "--episodes", str(args.episodes),
            "--steps", str(args.episodes * 500 + 500),
            "--seed", str(123 + args.actor_id),
            "--checkpoint", str(checkpoint), "--visualize-every", "1",
        ]
        if baseline:
            command.append("--baseline")
        jobs.append(launch(command, gpu, output / (name + ".log"), root))
        (output / "launch.json").write_text(json.dumps(dict(jobs=jobs), indent=2))
    print(json.dumps(dict(output=str(output), jobs=jobs), indent=2))


if __name__ == "__main__":
    main()
