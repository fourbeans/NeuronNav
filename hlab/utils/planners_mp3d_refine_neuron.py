"""Isolated MP3D planner workers for bounded SNN-path visibility."""

from functools import partial
import os
from multiprocessing import Pipe, Process

from poni.planner_adapter_mp3d_refine_neuron import mp3d_plan_refine
from utils.planners_opt_neuron import PlannerActorOptNeuron
from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron,
    PlannerActorVectorStrictNeuron,
)
import utils.planners_strict_neuron as strict_planners


def _install():
    direction_limit = os.environ.get("PONI_MP3D_REFINE_ANGLE_DEG")
    strict_planners.mp3d_plan_strict = partial(
        mp3d_plan_refine,
        cap_cm=float(os.environ["PONI_MP3D_REFINE_CAP_CM"]),
        cost_guard=os.environ["PONI_MP3D_REFINE_COST_GUARD"] == "1",
        max_direction_delta_deg=(float(direction_limit) if direction_limit else None),
    )


def worker_mp3d_refine_neuron(remote, parent_remote, planner_cfg, worker_id):
    del worker_id
    parent_remote.close()
    _install()
    planner = PlannerActorOptNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialMP3DRefineNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        _install()
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorOptNeuron(cfg) for _ in range(cfg.n_planners)]


class PlannerActorVectorMP3DRefineNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_mp3d_refine_neuron, args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
