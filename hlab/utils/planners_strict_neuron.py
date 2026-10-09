"""Strict MP3D SNN-wave planners with no weighted-A* fallback."""

from multiprocessing import Pipe, Process

from poni.planner_adapter_strict_neuron import mp3d_plan_strict
from poni.runtime_strict_neuron import AuditStrictNeuron
from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_neuron import (
    PlannerActorNeuron,
    PlannerActorSequentialNeuron,
    PlannerActorVectorNeuron,
)


class PlannerActorStrictNeuron(PlannerActorNeuron):
    def _audit(self):
        if not hasattr(self, "audit_neuron"):
            self.audit_neuron = AuditStrictNeuron("mp3d")
        return self.audit_neuron

    def _get_stg(self, grid, start, goal, planning_window):
        gx1, gx2, gy1, gy2 = map(int, planning_window)
        collision = self.collision_map[gx1:gx2, gy1:gy2]
        visited = self.visited[gx1:gx2, gy1:gy2]
        kernel = self.stg_selem_close if self.cfg.move_as_close_as_possible and self.is_close_to_goal else self.stg_selem
        audit = self._audit()
        plan = audit.choose(
            lambda: mp3d_plan_strict(
                grid, start, goal, collision, visited, self.selem, kernel,
                map_resolution=self.cfg.map_resolution,
                wscale=self.cfg.weighted_scale,
                niters=self.cfg.weighted_niters,
                downsampling=self.cfg.stg_downsampling,
                delay_quantum=audit.quantum,
            ),
            None,
            start,
            self.curr_loc[2],
            self.cfg.turn_angle,
            snapshot={
                "grid": grid,
                "goal": goal,
                "start": start,
                "collision": collision,
                "visited": visited,
                "obstacle_kernel": self.selem,
                "goal_kernel": kernel,
                "heading": self.curr_loc[2],
                "turn_angle": self.cfg.turn_angle,
                "map_resolution": self.cfg.map_resolution,
                "wscale": self.cfg.weighted_scale,
                "niters": self.cfg.weighted_niters,
            },
        )
        return plan.stg, plan.replan, plan.stop


def worker_strict_neuron(remote, parent_remote, planner_cfg, worker_id):
    del worker_id
    parent_remote.close()
    planner = PlannerActorStrictNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialStrictNeuron(PlannerActorSequentialNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorStrictNeuron(cfg) for _ in range(cfg.n_planners)]

class PlannerActorVectorStrictNeuron(PlannerActorVectorNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_strict_neuron, args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
