"""MP3D adapters preserving original PlannerActor controller/state machine."""
import time
from multiprocessing import Pipe, Process

import numpy as np

from poni.planner_adapter_neuron import PlanNeuron, mp3d_plan
from poni.runtime_neuron import AuditNeuron
from utils.planners import (PlannerActor, PlannerActorSequential, PlannerActorVector,
                            worker_fn, CLOSE_COMMAND)


class PlannerActorNeuron(PlannerActor):
    def _audit(self):
        if not hasattr(self, "audit_neuron"):
            self.audit_neuron = AuditNeuron("mp3d")
        return self.audit_neuron

    def _get_stg(self, grid, start, goal, planning_window):
        gx1, gx2, gy1, gy2 = map(int, planning_window)
        collision = self.collision_map[gx1:gx2, gy1:gy2]
        visited = self.visited[gx1:gx2, gy1:gy2]
        kernel = self.stg_selem_close if self.cfg.move_as_close_as_possible and self.is_close_to_goal else self.stg_selem
        audit = self._audit()
        def reference():
            tick = time.perf_counter()
            stg, replan, stop = super(PlannerActorNeuron, self)._get_stg(grid, start, goal.copy(), planning_window)
            return PlanNeuron(stg, stop, replan, info={"elapsed_ms": (time.perf_counter() - tick) * 1000})
        plan = audit.choose(
            lambda: mp3d_plan(grid, start, goal, collision, visited, self.selem, kernel,
                              map_resolution=self.cfg.map_resolution,
                              wscale=self.cfg.weighted_scale, niters=self.cfg.weighted_niters,
                              downsampling=self.cfg.stg_downsampling, delay_quantum=audit.quantum),
            reference, start, self.curr_loc[2], self.cfg.turn_angle,
            snapshot={"grid": grid, "goal": goal, "start": start, "collision": collision,
                      "visited": visited, "obstacle_kernel": self.selem, "goal_kernel": kernel,
                      "heading": self.curr_loc[2], "turn_angle": self.cfg.turn_angle,
                      "map_resolution": self.cfg.map_resolution,
                      "wscale": self.cfg.weighted_scale, "niters": self.cfg.weighted_niters})
        return plan.stg, plan.replan, plan.stop

    def plan_and_act(self, planner_inputs):
        action, replan = super().plan_and_act(planner_inputs)
        if not planner_inputs["wait"]:
            self._audit().write({"type": "action", "seq": self._audit().seq,
                                 "action": int(np.asarray(action).reshape(-1)[0]),
                                 "replan": bool(replan),
                                 "found_goal": bool(planner_inputs["found_goal"])})
        return action, replan


def worker_neuron(remote, parent_remote, planner_cfg, worker_id):
    parent_remote.close()
    planner = PlannerActorNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialNeuron(PlannerActorSequential):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorNeuron(cfg) for _ in range(cfg.n_planners)]


class PlannerActorVectorNeuron(PlannerActorVector):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [Process(target=worker_neuron, args=(work, remote, cfg, i))
                   for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()

    def close(self):
        if self.closed:
            return
        super().close()
        for process in self.ps:
            process.join(timeout=5)
        for remote in self.remotes:
            remote.close()
        self.closed = True
