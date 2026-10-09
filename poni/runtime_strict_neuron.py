"""Strict SNN-wave audit path: candidate failures never call a reference planner."""

import time

import numpy as np

from poni.planner_adapter_neuron import turn_action
from poni.runtime_neuron import AuditNeuron


class AuditStrictNeuron(AuditNeuron):
    def __init__(self, dataset):
        super().__init__(dataset)
        if self.mode != "neuron":
            raise ValueError("Strict runtime only supports PONI_NEURON_MODE=neuron")

    def choose(self, candidate_fn, reference_fn, start, heading, turn_angle, snapshot=None):
        del reference_fn
        self.seq += 1
        tick = time.perf_counter()
        try:
            candidate = candidate_fn()
        except Exception as exc:
            self.write({
                "type": "plan_failure",
                "seq": self.seq,
                "mode": "neuron_strict",
                "used": "none",
                "fallback": False,
                "error": type(exc).__name__ + ": " + str(exc),
                "elapsed_ms": (time.perf_counter() - tick) * 1000,
            })
            raise
        if candidate is None:
            raise RuntimeError("Strict SNN planner returned None")

        record = {
            "type": "plan",
            "seq": self.seq,
            "mode": "neuron_strict",
            "used": "neuron",
            "fallback": False,
            "reason": "",
            "error": None,
            "candidate": {
                "stg": list(map(float, candidate.stg)),
                "stop": bool(candidate.stop),
                "replan": bool(candidate.replan),
                "turn_action": turn_action(candidate.stg, start, heading, turn_angle),
                "info": candidate.info,
            },
            "reference": None,
            "elapsed_ms": (time.perf_counter() - tick) * 1000,
        }
        record["selected"] = record["candidate"]
        if snapshot is not None and self.captured < self.capture_limit and (self.seq - 1) % self.capture_every == 0:
            path = self.root / (self.dataset + "_" + str(self.pid) + "_" + str(self.seq) + ".npz")
            np.savez_compressed(path, **snapshot)
            record["snapshot"] = path.name
            self.captured += 1
        self.write(record)
        return candidate
