"""Separate MP3D worker binding for the equal-arrival SNN tie-break pilot."""

from multiprocessing import Pipe, Process

from poni.planner_adapter_mp3d_tiebreak_neuron import mp3d_plan_tiebreak
from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_opt_neuron import PlannerActorOptNeuron
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron, PlannerActorVectorStrictNeuron,
)
import utils.planners_strict_neuron as strict_planners


def _install():
    strict_planners.mp3d_plan_strict = mp3d_plan_tiebreak


def worker_mp3d_tiebreak_neuron(remote, parent_remote, planner_cfg, worker_id):
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


class PlannerActorSequentialMP3DTiebreakNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        _install()
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorOptNeuron(cfg) for _ in range(cfg.n_planners)]


class PlannerActorVectorMP3DTiebreakNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_mp3d_tiebreak_neuron, args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
