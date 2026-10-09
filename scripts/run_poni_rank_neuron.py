"""Bounded sequential capture/smoke jobs. Only creates NEW output directories."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["capture", "smoke"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--capture-limit", type=int, default=48)
    parser.add_argument("--capture-every", type=int, default=10)
    args = parser.parse_args()
    if args.phase == "smoke" and not args.checkpoint:
        parser.error("Smoke requires a trained --checkpoint")
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(root),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               GLOG_minloglevel="2", MAGNUM_LOG="quiet", HABITAT_SIM_LOG="quiet", PYTHONUNBUFFERED="1")
    modes = ["reference"] if args.phase == "capture" else ["reference", "guarded", "neuron"]
    statuses = []
    for mode in modes:
        for scene in args.scenes:
            dest = out / mode / scene
            dest.mkdir(parents=True)
            original = ["--split", "val", "--seed", "100", "--eval", "1",
                        "--task_config", "tasks/objectnav_gibson_box.yaml", "--version", "v1.1",
                        "--auto_gpu_config", "0", "--num_processes", "1", "--num_processes_on_first_gpu", "1",
                        "--num_processes_per_gpu", "1", "--sim_gpu_id", "0",
                        "--pf_model_path", str(root / "pretrained_models/gibson_models/poni_seed_123.ckpt"),
                        "-d", str(dest), "--num_eval_episodes", str(args.episodes),
                        "--num_local_steps", "1", "--exp_name", "seed_100", "--global_downscaling", "1",
                        "--mask_nearest_locations", "--pf_masking_opt", "unexplored", "--seg_interval", "3"]
            cmd = [sys.executable, "eval_poni_rank_neuron.py", "--rank-mode", mode,
                   "--rank-log-dir", str(dest / "rank_audit"), "--rank-scene", scene,
                   "--rank-capture-limit", str(args.capture_limit if args.phase == "capture" else 0),
                   "--rank-capture-every", str(args.capture_every)] + original
            if mode != "reference":
                cmd += ["--rank-checkpoint", str(Path(args.checkpoint).resolve())]
            with (dest / "command.json").open("x") as fp:
                json.dump({"command": cmd, "cwd": str(root / "semexp"), "gpu": args.gpu,
                           "original_entry_same_flags": [sys.executable, "eval_poni.py"] + original,
                           "note": "Pure original entry has no rank-scene override; use provided reference for exact scene pairing."}, fp, indent=2)
            tick = time.time()
            print("Starting " + mode + "/" + scene, flush=True)
            with (dest / "eval.log").open("x") as fp:
                completed = subprocess.run(cmd, cwd=str(root / "semexp"), env=env, stdout=fp, stderr=subprocess.STDOUT)
            status = {"mode": mode, "scene": scene, "returncode": completed.returncode,
                      "elapsed_seconds": time.time() - tick}
            with (dest / "run_status.json").open("x") as fp:
                json.dump(status, fp, indent=2)
            statuses.append(status)
            print(json.dumps(status), flush=True)
            if completed.returncode:
                raise SystemExit("Failed: " + str(dest / "eval.log"))
    with (out / "run_status.json").open("x") as fp:
        json.dump(statuses, fp, indent=2)


if __name__ == "__main__":
    main()
