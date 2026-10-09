"""Opt-in experiment modes, action-preserving guard, per-process audit logs."""
import json
import os
from pathlib import Path
import time

import numpy as np

from poni.planner_adapter_neuron import PlanNeuron, turn_action


MODES = ("reference", "shadow", "guarded", "neuron")


def add_neuron_arguments(parser):
    parser.add_argument("--neuron-mode", choices=MODES, default="guarded")
    parser.add_argument("--neuron-log-dir", required=True)
    parser.add_argument("--neuron-delay-quantum", type=float, default=0.0)
    parser.add_argument("--neuron-capture-limit", type=int, default=12)
    parser.add_argument("--neuron-capture-every", type=int, default=25)
    parser.add_argument("--neuron-scene", default="", help="Gibson only: one scene, requires one environment")


def export_options(options):
    for key, value in vars(options).items():
        if key.startswith("neuron_"):
            os.environ["PONI_" + key.upper()] = str(value)


class AuditNeuron:
    def __init__(self, dataset):
        self.dataset = dataset
        self.mode = os.environ.get("PONI_NEURON_MODE", "guarded")
        if self.mode not in MODES:
            raise ValueError(self.mode)
        self.quantum = float(os.environ.get("PONI_NEURON_DELAY_QUANTUM", "0"))
        root = os.environ.get("PONI_NEURON_LOG_DIR")
        if not root:
            raise ValueError("PONI_NEURON_LOG_DIR is required for an auditable experiment")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pid = os.getpid()
        self.path = self.root / (dataset + "_" + str(self.pid) + ".jsonl")
        self.seq = 0
        self.capture_limit = int(os.environ.get("PONI_NEURON_CAPTURE_LIMIT", "12"))
        self.capture_every = max(1, int(os.environ.get("PONI_NEURON_CAPTURE_EVERY", "25")))
        self.captured = 0
        self.write({"type": "init", "mode": self.mode, "delay_quantum": self.quantum,
                    "pid": self.pid, "dataset": dataset})

    def write(self, record):
        with self.path.open("a") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def choose(self, candidate_fn, reference_fn, start, heading, turn_angle, snapshot=None):
        self.seq += 1
        tick = time.perf_counter()
        candidate, reference, error = None, None, None
        if self.mode != "reference":
            try:
                candidate = candidate_fn()
            except Exception as exc:
                # Never silently hide a candidate failure; reference exceptions propagate.
                error = type(exc).__name__ + ": " + str(exc)
        if self.mode != "neuron" or candidate is None:
            reference = reference_fn()
        reason = ""
        if self.mode in ("reference", "shadow"):
            selected = reference
            used = "reference"
        elif candidate is None:
            selected, used, reason = reference, "reference", "candidate_failure"
        elif self.mode == "neuron":
            selected, used = candidate, "neuron"
        else:
            # Same controller action + same state-changing flags => same next step.
            if candidate.stop != reference.stop:
                reason = "stop_disagreement"
            elif self.dataset == "mp3d" and candidate.replan != reference.replan:
                reason = "replan_disagreement"
            elif turn_action(candidate.stg, start, heading, turn_angle) != turn_action(reference.stg, start, heading, turn_angle):
                reason = "action_disagreement"
            selected, used = (reference, "reference") if reason else (candidate, "neuron")
        def pack(plan):
            if plan is None:
                return None
            return {"stg": list(map(float, plan.stg)), "stop": bool(plan.stop),
                    "replan": bool(plan.replan),
                    "turn_action": turn_action(plan.stg, start, heading, turn_angle),
                    "info": plan.info}
        record = {"type": "plan", "seq": self.seq, "mode": self.mode,
                  "used": used, "fallback": bool(reason), "reason": reason, "error": error,
                  "candidate": pack(candidate), "reference": pack(reference),
                  "selected": pack(selected), "elapsed_ms": (time.perf_counter() - tick) * 1000}
        if snapshot is not None and self.captured < self.capture_limit and (self.seq - 1) % self.capture_every == 0:
            path = self.root / (self.dataset + "_" + str(self.pid) + "_" + str(self.seq) + ".npz")
            np.savez_compressed(path, **snapshot)
            record["snapshot"] = path.name
            self.captured += 1
        self.write(record)
        return selected
