"""Old strict MP3D SNN with episode/pose audit, no readout patch."""

from multiprocessing import Pipe, Process

from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_opt_neuron import PlannerActorOptNeuron
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron,
    PlannerActorVectorStrictNeuron,
)


def worker_trace_neuron(remote, parent_remote, planner_cfg, worker_id):
    del worker_id
    parent_remote.close()
    planner = PlannerActorOptNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialTraceNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorOptNeuron(cfg) for _ in range(cfg.n_planners)]


class PlannerActorVectorTraceNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[
            Pipe() for _ in range(cfg.n_planners)
        ])
        self.ps = [
            Process(target=worker_trace_neuron, args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
