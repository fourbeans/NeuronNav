"""Gibson actor/evaluator. Reuses the original mapper, target override and FMM."""
import argparse
import json
import os
from pathlib import Path
import sys
from .common_dend import new_run


def official_val_scenes(root):
    """List scene episodes, excluding val's auxiliary *_episodes files."""
    content = root / "data/datasets/objectnav/gibson/v1.1/val/content"
    scenes = sorted(path.name[:-8] for path in content.glob("*.glb.json.gz"))
    if not scenes:
        raise FileNotFoundError("No Gibson official val scene files in " + str(content))
    assets = root / "data/scene_datasets/gibson_semantic"
    missing = [scene for scene in scenes if not (assets / scene).is_file()]
    if missing:
        raise FileNotFoundError("Missing Gibson official val scene assets: " + repr(missing))
    return scenes


def scene_shard(scenes, actor_id, actors, role):
    """Cycle held-out scenes for independent parallel episode streams."""
    if role in ("dev", "val") and actors > len(scenes):
        return [scenes[actor_id % len(scenes)]]
    return scenes[actor_id::actors]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", choices=("train", "dev", "val"), default="train")
    p.add_argument("--actor-id", type=int, default=0)
    p.add_argument("--actors", type=int, default=1)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--visualize-every", type=int, default=10)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = Path(args.experiment).resolve()
    args.output = str(Path(args.output).resolve())
    out = new_run(args.output, args, "dendritic_nav.runners.collect_dend")
    split = json.loads((experiment / "scene_split.json").read_text())
    if args.role == "val":
        scenes = official_val_scenes(root)
    else:
        scenes = split[args.role]
    # Independent deterministic episode streams; cycle held-out scenes for
    # parallelism while preserving actor seed/scene baseline-dend pairing.
    shard = scene_shard(scenes, args.actor_id, args.actors, args.role)
    if not shard:
        raise ValueError("Actor count exceeds available scenes")
    spatial = json.loads((experiment / "spatial_checkpoint.json").read_text())["path"]
    checkpoint = args.checkpoint or str(experiment / "checkpoints/published.ckpt")
    if args.role != "train" and not args.baseline and not args.checkpoint:
        raise ValueError(
            "Evaluation must explicitly pin --checkpoint (never evaluate moving weights)"
        )
    os.chdir(str(root / "semexp"))
    from semexp.arguments import get_args

    saved = sys.argv
    sys.argv = [
        saved[0],
        "--split",
        "val" if args.role == "val" else "train",
        "--seed",
        str(args.seed),
        "--eval",
        "1",
        "--pf_model_path",
        spatial,
        "-d",
        str(out / "navigation"),
        "--exp_name",
        "dend",
        "--num_local_steps",
        "1",
        "--global_downscaling",
        "1",
        "--mask_nearest_locations",
        "--pf_masking_opt",
        "unexplored",
        "--auto_gpu_config",
        "0",
        "--num_processes",
        "1",
        "--num_processes_on_first_gpu",
        "1",
        "--num_processes_per_gpu",
        "1",
        "--sim_gpu_id",
        "0",
        "--num_train_episodes",
        "10",
        "--num_eval_episodes",
        str(args.episodes or 1000000),
        "--num_training_frames",
        str(args.steps + 10000),
    ]
    try:
        habitat_args = get_args()
    finally:
        sys.argv = saved
    habitat_args.dend_scenes = shard
    (out / "resolved_habitat_args.json").write_text(
        json.dumps(vars(habitat_args), indent=2, default=str)
    )
    from semexp import eval_poni as evaluator
    import semexp.envs.habitat as environment_module
    from semexp.envs import make_vec_envs
    from dendritic_nav.adapters.gibson_env_dend import make_env_dend
    from dendritic_nav.adapters.gibson_policy_dend import DendriticGibsonPolicy
    from dendritic_nav.adapters.bridge_dend import (
        EpisodeBridge,
        VecBridge,
        BudgetReached,
    )

    bridge = EpisodeBridge(
        out, args.role == "train", args.steps, args.episodes, args.visualize_every
    )

    def policy_factory(a, pf_model_path):
        return DendriticGibsonPolicy(
            a, pf_model_path, bridge, checkpoint, args.baseline
        )

    environments = []

    def env_factory(a, **kwargs):
        env = VecBridge(make_vec_envs(a, **kwargs), bridge)
        environments.append(env)
        return env

    # Registration is process-local and restored even on exceptions. No files
    # or installed modules are altered. Spawned workers import make_env_dend.
    old_policy, old_env, old_make = (
        evaluator.RL_Policy,
        evaluator.make_vec_envs,
        environment_module.make_env_fn,
    )
    evaluator.RL_Policy = policy_factory
    evaluator.make_vec_envs = env_factory
    environment_module.make_env_fn = make_env_dend
    try:
        evaluator.main(habitat_args)
    except BudgetReached:
        print("DEND_BUDGET_COMPLETE", flush=True)
    finally:
        for env in environments:
            env.close()
        evaluator.RL_Policy = old_policy
        evaluator.make_vec_envs = old_env
        environment_module.make_env_fn = old_make
    (out / "completion.json").write_text(
        json.dumps(dict(steps=bridge.steps, episodes=bridge.episodes))
    )


if __name__ == "__main__":
    main()
