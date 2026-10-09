"""Strict Gibson SNN-wave environment with no FMM fallback."""

import os

from habitat import make_dataset

from poni.planner_adapter_strict_neuron import gibson_plan_strict
from poni.runtime_strict_neuron import AuditStrictNeuron
from semexp.envs.habitat.sem_exp_neuron import SemExpEnvAgentNeuron


class SemExpEnvAgentStrictNeuron(SemExpEnvAgentNeuron):
    def _audit(self):
        if not hasattr(self, "audit_neuron"):
            self.audit_neuron = AuditStrictNeuron("gibson")
        return self.audit_neuron

    def _get_stg(self, grid, start, goal, planning_window):
        gx1, gx2, gy1, gy2 = map(int, planning_window)
        collision = self.collision_map[gx1:gx2, gy1:gy2]
        visited = self.visited[gx1:gx2, gy1:gy2]
        audit = self._audit()
        plan = audit.choose(
            lambda: gibson_plan_strict(
                grid, start, goal, collision, visited, self.selem,
                self.stg_selem, audit.quantum,
            ),
            None,
            start,
            self.curr_loc[2],
            self.args.turn_angle,
            snapshot={
                "grid": grid,
                "goal": goal,
                "start": start,
                "collision": collision,
                "visited": visited,
                "obstacle_kernel": self.selem,
                "goal_kernel": self.stg_selem,
                "heading": self.curr_loc[2],
                "turn_angle": self.args.turn_angle,
                "map_resolution": self.args.map_resolution,
            },
        )
        return plan.stg, plan.stop


def make_env_fn_strict_neuron(args, config_env, rank):
    selected_scene = os.environ.get("PONI_NEURON_SCENE", "")
    if selected_scene:
        if args.num_processes != 1:
            raise ValueError("Explicit scene selection requires --num_processes 1")
        config_env.defrost()
        config_env.DATASET.CONTENT_SCENES = [
            selected_scene if selected_scene.endswith(".glb") else selected_scene + ".glb"
        ]
        config_env.freeze()
    dataset = make_dataset(config_env.DATASET.TYPE, config=config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    config_env.freeze()
    if args.agent != "sem_exp":
        raise ValueError("The strict neuron entry point supports the sem_exp agent only")
    env = SemExpEnvAgentStrictNeuron(args=args, rank=rank, config_env=config_env, dataset=dataset)
    env.seed(rank + args.seed)
    return env
