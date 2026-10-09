"""Launch exact paired baseline/dendritic traces for chosen dev episode prefixes."""
import argparse
import json
from pathlib import Path
import sys

from dendritic_nav.runners.launch_dend import launch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", required=True,
                        help="Actor and last episode, e.g. 0:2")
    parser.add_argument("--actors", type=int, default=2,
                        help="Original dev-evaluation actor count")
    parser.add_argument("--gpus", required=True, help="One GPU per trace job")
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    output = Path(args.output).resolve()
    gpus = [int(g) for g in args.gpus.split(",")]
    cases = [tuple(int(x) for x in item.split(":")) for item in args.case]
    if len(gpus) < 2 * len(cases):
        raise ValueError("Need two GPU assignments per paired case")
    if any(actor < 0 or actor >= args.actors or episodes < 1
           for actor, episodes in cases):
        raise ValueError("Each actor must belong to the dev evaluation")
    checkpoint = experiment / "learner/checkpoints/pilot_final.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    root = Path(__file__).resolve().parents[2]
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for index, (actor, episodes) in enumerate(cases):
        for baseline in (True, False):
            name = ("baseline" if baseline else "dend") + "_actor_{:03d}".format(actor)
            command = [
                sys.executable, "-u", "-m", "dendritic_nav.diagnostics.trace_dend",
                "--experiment", str(experiment), "--output", str(output / name),
                "--role", "dev", "--actor-id", str(actor),
                "--actors", str(args.actors),
                "--episodes", str(episodes), "--steps", str(episodes * 500 + 500),
                "--seed", str(123 + actor), "--checkpoint", str(checkpoint),
                "--visualize-every", "1",
            ]
            if baseline:
                command.append("--baseline")
            gpu = gpus[2 * index + int(not baseline)]
            jobs.append(launch(command, gpu, output / (name + ".log"), root))
            (output / "launch.json").write_text(json.dumps(dict(jobs=jobs), indent=2))
    print(json.dumps(dict(output=str(output), jobs=jobs), indent=2))


if __name__ == "__main__":
    main()
