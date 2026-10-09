"""Install the selected strict SNN-wave readout in each MP3D planner worker."""

from functools import partial
import os
from multiprocessing import Pipe, Process

import numpy as np

from poni.planner_adapter_opt_neuron import mp3d_plan_opt
from utils.planners import CLOSE_COMMAND, worker_fn
from utils.planners_strict_neuron import (
    PlannerActorSequentialStrictNeuron,
    PlannerActorStrictNeuron,
    PlannerActorVectorStrictNeuron,
)
import utils.planners_strict_neuron as strict_planners


def _install():
    strict_planners.mp3d_plan_strict = partial(
        mp3d_plan_opt,
        mode=os.environ["PONI_WAVEOPT_READOUT"],
        lookahead_cm=float(os.environ["PONI_WAVEOPT_LOOKAHEAD_CM"]),
    )


class PlannerActorOptNeuron(PlannerActorStrictNeuron):
    def reset(self):
        super().reset()
        if os.environ.get("PONI_WAVEOPT_TRACE") == "1":
            self.waveopt_episode_index = getattr(self, "waveopt_episode_index", -1) + 1
            self._audit().write({"type": "episode_start", "episode_index": self.waveopt_episode_index})

    def plan_and_act(self, planner_inputs):
        result = super().plan_and_act(planner_inputs)
        if os.environ.get("PONI_WAVEOPT_TRACE") == "1" and not planner_inputs["wait"]:
            self._audit().write({
                "type": "pose",
                "seq": self._audit().seq,
                "episode_index": getattr(self, "waveopt_episode_index", 0),
                "pose_pred": np.asarray(planner_inputs["pose_pred"], dtype=float).tolist(),
            })
        return result


def worker_opt_neuron(remote, parent_remote, planner_cfg, worker_id):
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


class PlannerActorSequentialOptNeuron(PlannerActorSequentialStrictNeuron):
    def __init__(self, cfg):
        _install()
        self.cfg = cfg
        self.n_planners = cfg.n_planners
        self.closed = False
        self.planners = [PlannerActorOptNeuron(cfg) for _ in range(cfg.n_planners)]


class PlannerActorVectorOptNeuron(PlannerActorVectorStrictNeuron):
    def __init__(self, cfg):
        self.cfg = cfg
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(cfg.n_planners)])
        self.ps = [
            Process(target=worker_opt_neuron, args=(work, remote, cfg, i))
            for i, (work, remote) in enumerate(zip(self.work_remotes, self.remotes))
        ]
        for process in self.ps:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()
