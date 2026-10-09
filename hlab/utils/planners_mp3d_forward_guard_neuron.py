"""MP3D strict SNN planner with an equal-cost, forward-preserving readout."""

from multiprocessing import Pipe, Process

from poni.planner_adapter_mp3d_action_readout_neuron import mp3d_plan_action_readout
from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_opt_neuron import PlannerActorOptNeuron
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron, PlannerActorVectorStrictNeuron,
)


class PlannerActorMP3DForwardGuardNeuron(PlannerActorOptNeuron):
    READOUT_KWARGS = {"preserve_forward": True}

    def _get_stg(self, grid, start, goal, planning_window):
        gx1, gx2, gy1, gy2 = map(int, planning_window)
        collision = self.collision_map[gx1:gx2, gy1:gy2]
        visited = self.visited[gx1:gx2, gy1:gy2]
        kernel = (self.stg_selem_close if self.cfg.move_as_close_as_possible
                  and self.is_close_to_goal else self.stg_selem)
        audit = self._audit()
        heading, turn_angle = self.curr_loc[2], self.cfg.turn_angle
        plan = audit.choose(
            lambda: mp3d_plan_action_readout(
                grid, start, goal, collision, visited, self.selem, kernel,
                map_resolution=self.cfg.map_resolution,
                wscale=self.cfg.weighted_scale,
                niters=self.cfg.weighted_niters,
                downsampling=self.cfg.stg_downsampling,
                delay_quantum=audit.quantum,
                heading=heading,
                turn_angle=turn_angle,
                **self.READOUT_KWARGS,
            ),
            None, start, heading, turn_angle,
            snapshot={
                "grid": grid, "goal": goal, "start": start,
                "collision": collision, "visited": visited,
                "obstacle_kernel": self.selem, "goal_kernel": kernel,
                "heading": heading, "turn_angle": turn_angle,
                "map_resolution": self.cfg.map_resolution,
                "wscale": self.cfg.weighted_scale,
                "niters": self.cfg.weighted_niters,
            },
        )
        return plan.stg, plan.replan, plan.stop


def worker_mp3d_forward_guard_neuron(remote, parent_remote, planner_cfg, worker_id):
    del worker_id
    parent_remote.close()
    planner = PlannerActorMP3DForwardGuardNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialMP3DForwardGuardNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorMP3DForwardGuardNeuron(cfg)
                         for _ in range(cfg.n_planners)]


class PlannerActorVectorMP3DForwardGuardNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_mp3d_forward_guard_neuron,
                    args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
