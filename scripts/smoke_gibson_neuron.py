"""Bounded, sequential Gibson baseline/guarded/neuron runs in NEW directories."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--episodes-per-process", type=int, default=1)
    parser.add_argument("--scenes", nargs="+", help="Run these Gibson scenes sequentially with one process")
    parser.add_argument("--modes", nargs="+", choices=["reference", "guarded", "neuron", "shadow"],
                        default=["reference", "guarded", "neuron"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(root),
               GLOG_minloglevel="2", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    results = []
    if args.scenes and args.processes != 1:
        parser.error("--scenes requires --processes 1")
    tasks = [(mode, scene) for mode in args.modes for scene in (args.scenes or [""])]
    for mode, scene in tasks:
        dest = out / mode / scene if scene else out / mode
        dest.mkdir(parents=True)
        cmd = [sys.executable, "eval_poni_neuron.py", "--neuron-mode", mode,
               "--neuron-log-dir", str(dest / "audit"), "--neuron-capture-limit", "8",
               "--neuron-capture-every", "50", "--split", "val", "--seed", "100", "--eval", "1",
               "--task_config", "tasks/objectnav_gibson_box.yaml", "--version", "v1.1",
               "--auto_gpu_config", "0", "--num_processes", str(args.processes),
               "--num_processes_on_first_gpu", str(args.processes),
               "--num_processes_per_gpu", "1", "--sim_gpu_id", "0",
               "--pf_model_path", str(root / "pretrained_models/gibson_models/poni_seed_123.ckpt"),
               "-d", str(dest), "--num_eval_episodes", str(args.episodes_per_process),
               "--num_local_steps", "1", "--exp_name", "seed_100", "--global_downscaling", "1",
               "--mask_nearest_locations", "--pf_masking_opt", "unexplored", "--seg_interval", "3"]
        if scene:
            cmd += ["--neuron-scene", scene]
        with (dest / "command.json").open("x") as fp:
            json.dump({"command": cmd, "cwd": str(root / "semexp"),
                       "gpu": args.gpu, "episodes_per_process": args.episodes_per_process}, fp, indent=2)
        print("Starting " + mode + " -> " + str(dest), flush=True)
        tick = time.time()
        with (dest / "eval.log").open("x") as fp:
            process = subprocess.run(cmd, cwd=str(root / "semexp"), env=env, stdout=fp,
                                     stderr=subprocess.STDOUT)
        result = {"mode": mode, "scene": scene, "returncode": process.returncode, "elapsed_seconds": time.time() - tick}
        results.append(result)
        with (dest / "run_status.json").open("x") as fp:
            json.dump(result, fp, indent=2)
        print(json.dumps(result), flush=True)
        if process.returncode != 0:
            raise SystemExit("Failed; see " + str(dest / "eval.log"))
    with (out / "run_status.json").open("x") as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
