"""Capture terminal transitions BEFORE the upstream vector-env auto reset."""
import copy
import hashlib
import json
import random
import numpy as np
from habitat import make_dataset
from semexp.envs.habitat.sem_exp import Sem_Exp_Env_Agent


class DendriticGibsonEnv(Sem_Exp_Env_Agent):
    def reset(self):
        terminal = getattr(self, "_dend_terminal", None)
        obs, info = super().reset()
        identity = dict(
            scene=str(self.scene_name),
            pose=list(map(float, self.get_sim_location())),
            categories=list(map(int, self.goal_idxs)),
            episode=int(self.episode_no),
        )
        self._dend_episode_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self._dend_terminal = None
        info = copy.deepcopy(info)
        info["dend_initial_distance"] = float(self.prev_distance)
        if terminal is not None:
            info["dend_transition"] = terminal
            # The evaluator's metrics must still describe the finished episode.
            for key in (
                "success",
                "spl",
                "dts",
                "ppl",
                "progress",
                "gspl",
                "gppl",
                "goal_distance",
            ):
                if key in terminal["metrics"]:
                    info[key] = terminal["metrics"][key]
        else:
            info.pop("dend_transition", None)
        return obs, info

    def plan_act_and_preprocess(self, planner_inputs):
        before = float(self.prev_distance)
        obs, native_reward, done, info = super().plan_act_and_preprocess(planner_inputs)
        info = copy.deepcopy(info)
        executed = not planner_inputs["wait"] and self.last_action is not None
        delta = list(info["sensor_pose"])
        collision = (
            executed
            and self.last_action == 1
            and np.hypot(*delta[:2]) < self.args.collision_threshold
        )
        transition = dict(
            executed=executed,
            before=before,
            after=float(self.prev_distance),
            delta=delta,
            action=-1 if self.last_action is None else int(self.last_action),
            collision=bool(collision),
            terminal=bool(done),
            success=float(info.get("success", 0)) if done else 0.0,
            spl=float(info.get("spl", 0)) if done else 0.0,
            scene=str(self.scene_name),
            episode_id=self._dend_episode_id,
            metrics={
                k: info[k]
                for k in (
                    "success",
                    "spl",
                    "dts",
                    "ppl",
                    "progress",
                    "gspl",
                    "gppl",
                    "goal_distance",
                )
                if k in info
            },
        )
        info["dend_transition"] = transition
        if done:
            self._dend_terminal = transition
        return obs, native_reward, done, info


def make_env_dend(args, config_env, rank):
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    config_env = config_env.clone()
    config_env.defrost()
    config_env.DATASET.CONTENT_SCENES = list(args.dend_scenes)
    config_env.freeze()
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    if not dataset.episodes:
        raise ValueError("No episodes in selected scene shard")
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    env = DendriticGibsonEnv(
        args=args, rank=rank, config_env=config_env, dataset=dataset
    )
    env.seed(args.seed + rank)
    return env
