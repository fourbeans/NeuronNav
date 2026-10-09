"""Gibson opt-in subclass. No modifications to original agent or GT metrics."""
import time
import os

import numpy as np
from habitat import make_dataset

from poni.planner_adapter_neuron import PlanNeuron, gibson_plan
from poni.runtime_neuron import AuditNeuron
from semexp.envs.habitat.sem_exp import Sem_Exp_Env_Agent


class SemExpEnvAgentNeuron(Sem_Exp_Env_Agent):
    def _audit(self):
        if not hasattr(self, "audit_neuron"):
            self.audit_neuron = AuditNeuron("gibson")
        return self.audit_neuron

    def _get_stg(self, grid, start, goal, planning_window):
        gx1, gx2, gy1, gy2 = map(int, planning_window)
        collision = self.collision_map[gx1:gx2, gy1:gy2]
        visited = self.visited[gx1:gx2, gy1:gy2]
        audit = self._audit()
        def reference():
            tick = time.perf_counter()
            stg, stop = super(SemExpEnvAgentNeuron, self)._get_stg(grid, start, goal.copy(), planning_window)
            return PlanNeuron(stg, stop, info={"elapsed_ms": (time.perf_counter() - tick) * 1000})
        plan = audit.choose(
            lambda: gibson_plan(grid, start, goal, collision, visited, self.selem,
                                self.stg_selem, audit.quantum),
            reference, start, self.curr_loc[2], self.args.turn_angle,
            snapshot={"grid": grid, "goal": goal, "start": start, "collision": collision,
                      "visited": visited, "obstacle_kernel": self.selem,
                      "goal_kernel": self.stg_selem, "heading": self.curr_loc[2],
                      "turn_angle": self.args.turn_angle, "map_resolution": self.args.map_resolution})
        return plan.stg, plan.stop

    def _plan(self, planner_inputs):
        action = super()._plan(planner_inputs)
        self._audit().write({"type": "action", "seq": self._audit().seq,
                             "episode_no": int(self.episode_no), "step": int(self.timestep),
                             "scene": str(self.scene_name), "action": int(action),
                             "found_goal": bool(planner_inputs["found_goal"])})
        return action

    def reset(self):
        obs, info = super().reset()
        # PONI's manually loaded episode is not necessarily Habitat.current_episode.
        self._audit().write({"type": "episode_start", "episode_no": int(self.episode_no),
                             "scene": str(self.scene_name),
                             "position": np.asarray(self.episode_start_position).tolist(),
                             "rotation": str(self.episode_start_rotation),
                             "goal_ids": list(map(int, self.goal_idxs))})
        return obs, info


def make_env_fn_neuron(args, config_env, rank):
    """Top-level importable factory: safe under forkserver/spawn pickling."""
    selected_scene = os.environ.get("PONI_NEURON_SCENE", "")
    if selected_scene:
        if args.num_processes != 1:
            raise ValueError("Explicit scene selection requires --num_processes 1")
        config_env.defrost()
        config_env.DATASET.CONTENT_SCENES = [selected_scene if selected_scene.endswith(".glb") else selected_scene + ".glb"]
        config_env.freeze()
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    if args.agent != "sem_exp":
        raise ValueError("The neuron entry point supports the sem_exp agent only")
    env = SemExpEnvAgentNeuron(args=args, rank=rank, config_env=config_env, dataset=dataset)
    env.seed(rank + args.seed)
    return env
