"""Independent worker binding for turn-to-forward-only equal-cost SNN paths."""

from multiprocessing import Pipe, Process

from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_mp3d_forward_guard_neuron import PlannerActorMP3DForwardGuardNeuron
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron, PlannerActorVectorStrictNeuron,
)


class PlannerActorMP3DForwardGainNeuron(PlannerActorMP3DForwardGuardNeuron):
    READOUT_KWARGS = {"forward_gain_only": True}


def worker_mp3d_forward_gain_neuron(remote, parent_remote, planner_cfg, worker_id):
    del worker_id
    parent_remote.close()
    planner = PlannerActorMP3DForwardGainNeuron(planner_cfg)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == CLOSE_COMMAND:
                break
            remote.send(worker_fn(planner, cmd, data))
    finally:
        remote.close()


class PlannerActorSequentialMP3DForwardGainNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorMP3DForwardGainNeuron(cfg)
                         for _ in range(cfg.n_planners)]


class PlannerActorVectorMP3DForwardGainNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_mp3d_forward_gain_neuron,
                    args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
