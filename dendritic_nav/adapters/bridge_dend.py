"""Step/episode lifecycle. Only this bridge sees privileged reward metadata."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from dendritic_nav.learning.reward_dend import reward
from dendritic_nav.learning.replay_dend import validate_episode


class BudgetReached(Exception):
    pass


class EpisodeBridge:
    def __init__(
        self, directory, training, max_steps, max_episodes=0, visualize_every=10
    ):
        self.directory = Path(directory)
        (self.directory / "replay").mkdir(parents=True, exist_ok=True)
        (self.directory / "visualizations").mkdir(exist_ok=True)
        self.training = training
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.visualize_every = visualize_every
        self.new_episode = True
        self.advance = True
        self.motion = {}
        self.pending = None
        self.records = []
        self.steps = self.episodes = 0
        self.started = time.time()
        self.policy = None

    def consume(self, planner_input, info):
        transition = info["dend_transition"]
        self.advance = bool(transition["executed"])
        if not self.advance:
            self.policy.rollback_observation()
            self.pending = None
            return
        if self.pending is None:
            raise RuntimeError("Real action without a matching observation record")
        record = self.pending
        self.pending = None
        overridden = bool(planner_input["found_goal"])
        record["action"] = -1 if overridden else record["action"]
        switches_before = self.policy.history.switches
        if not overridden:
            self.policy.history.choose(record["goal_xy"])
        record["goal_switch"] = self.policy.history.switches > switches_before
        record["reward"] = reward(
            transition["before"],
            transition["after"],
            transition["collision"],
            transition["terminal"],
            transition["success"],
            transition["spl"],
            goal_switch=record["goal_switch"],
        )
        record["terminal"] = transition["terminal"]
        record["override"] = overridden
        record["collision"] = transition["collision"]
        self.records.append(record)
        self.steps += 1
        self.motion = {k: transition[k] for k in ("delta", "action", "collision")}
        # Advance coordinate statistics at the observed pre-action pose; next
        # observation carries measured displacement, never GT goal distance.
        self.policy.history.advance(
            record["agent_xy"], transition["collision"], record["new_target"]
        )
        if transition["terminal"]:
            self.policy.history.finish(transition["success"])
            self.finish_episode(transition)
            self.new_episode = True
            self.advance = True
            self.motion = {}

    def finish_episode(self, transition):
        keys = (
            "features",
            "candidates",
            "spatial",
            "valid",
            "action",
            "reward",
            "terminal",
            "rc",
            "agent_xy",
            "goal_xy",
            "override",
            "collision",
            "goal_switch",
            "activity",
            "residual",
            "selected_q",
            "weight_version",
        )
        episode = {k: np.asarray([row[k] for row in self.records]) for k in keys}
        for key in (
            "features",
            "candidates",
            "spatial",
            "reward",
            "residual",
            "selected_q",
        ):
            episode[key] = episode[key].astype(np.float32)
        episode["action"] = episode["action"].astype(np.int16)
        validate_episode(episode)
        if self.training:
            path = (
                self.directory / "replay" / "episode_{:07d}.npz".format(self.episodes)
            )
            with open(str(path) + ".tmp", "xb") as stream:
                np.savez_compressed(stream, **episode)
            os.replace(str(path) + ".tmp", path)
        if self.episodes % self.visualize_every == 0:
            from dendritic_nav.visualization.plots_dend import plot_episode

            plot_episode(
                episode,
                self.directory
                / "visualizations"
                / "episode_{:07d}.png".format(self.episodes),
                map_snapshot=self.policy.current_maps[:2],
                resolution=self.policy.args.map_resolution / 100.0,
            )
        self.episodes += 1
        metrics = dict(
            episode=self.episodes,
            steps=self.steps,
            length=len(self.records),
            episode_id=transition["episode_id"],
            success=transition["success"],
            spl=transition["spl"],
            scene=transition["scene"],
            return_sum=float(episode["reward"].sum()),
            decisions=int((episode["action"] >= 0).sum()),
            seconds=time.time() - self.started,
            weight_version=int(episode["weight_version"][0]),
            trajectory_sha256=hashlib.sha256(
                episode["agent_xy"].tobytes()
                + episode["goal_xy"].tobytes()
                + episode["override"].tobytes()
            ).hexdigest(),
        )
        with open(self.directory / "episodes.jsonl", "a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        print("DEND_EPISODE " + json.dumps(metrics), flush=True)
        self.records = []
        if self.steps >= self.max_steps or (
            self.max_episodes and self.episodes >= self.max_episodes
        ):
            raise BudgetReached()


class VecBridge:
    def __init__(self, env, bridge):
        self.env, self.bridge = env, bridge

    def __getattr__(self, name):
        return getattr(self.env, name)

    def plan_act_and_preprocess(self, inputs):
        result = self.env.plan_act_and_preprocess(inputs)
        self.bridge.consume(inputs[0], result[3][0])
        return result
