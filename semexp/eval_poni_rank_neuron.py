"""Independent PONI rank experiment; low-level Gibson planner stays original FMM."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank-mode", choices=["reference", "shadow", "guarded", "neuron"], default="guarded")
    parser.add_argument("--rank-checkpoint", default="")
    parser.add_argument("--rank-log-dir", required=True)
    parser.add_argument("--rank-capture-limit", type=int, default=0)
    parser.add_argument("--rank-capture-every", type=int, default=10)
    parser.add_argument("--rank-scene", default="")
    parser.add_argument("--rank-stride", type=int, default=16)
    parser.add_argument("--rank-refine-topk", type=int, default=4)
    options, original = parser.parse_known_args()
    if options.rank_mode != "reference" and not options.rank_checkpoint:
        parser.error("--rank-checkpoint is required for a trained student")
    if options.rank_capture_limit and not options.rank_scene:
        parser.error("Data capture requires an explicit --rank-scene for leak-free splits")
    root = Path(options.rank_log_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for key, value in vars(options).items():
        os.environ["PONI_" + key.upper()] = str(value)
    os.environ["PONI_RANK_LOG_DIR"] = str(root)
    # Reuse ONLY the previous branch's audit/factory, force original FMM.
    os.environ.update(PONI_NEURON_MODE="reference", PONI_NEURON_CAPTURE_LIMIT="0",
                      PONI_NEURON_LOG_DIR=str(root / "low_level_fmm"),
                      PONI_NEURON_SCENE=options.rank_scene)
    with (root / "launch.json").open("x") as fp:
        json.dump({"argv": sys.argv, "cwd": os.getcwd(), "options": vars(options),
                   "low_level_planner": "original_FMM"}, fp, indent=2)
    sys.argv = [sys.argv[0]] + original
    import semexp.envs.habitat as environment
    from semexp.envs.habitat.sem_exp_neuron import make_env_fn_neuron
    import semexp.eval_poni as evaluator
    from semexp.model_pf_rank_neuron import RankPolicyNeuron
    environment.make_env_fn = make_env_fn_neuron
    evaluator.RL_Policy = RankPolicyNeuron
    return evaluator.main()


if __name__ == "__main__":
    main()
