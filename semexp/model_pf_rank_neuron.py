"""Opt-in PONI teacher/student ranking; original low-level planner is untouched."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from poni.rank_features_neuron import (FEATURE_VERSION, FeatureBankNeuron, decode_goal,
                                        encode_goal, teacher_scores)
from poni.rank_model_neuron import RankSNNNeuron, rank_candidates
from semexp.model_pf import RL_Policy

CONTRACT_KEYS = ("pf_masking_opt", "mask_nearest_locations", "mask_size", "map_resolution",
                 "area_weight_coef", "dist_weight_coef", "add_agent2loc_distance",
                 "add_agent2loc_distance_v2", "use_egocentric_transform", "num_sem_categories")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_value(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


class RankPolicyNeuron(RL_Policy):
    def __init__(self, args, pf_model_path):
        super().__init__(args, pf_model_path)
        self.rank_mode = os.environ.get("PONI_RANK_MODE", "guarded")
        if self.rank_mode not in ("reference", "shadow", "guarded", "neuron"):
            raise ValueError("Unknown rank mode")
        self.rank_root = Path(os.environ["PONI_RANK_LOG_DIR"])
        self.rank_root.mkdir(parents=True, exist_ok=True)
        self.rank_log = self.rank_root / ("rank_" + str(os.getpid()) + ".jsonl")
        self.contract = {key: getattr(args, key) for key in CONTRACT_KEYS}
        self.teacher_hash = file_sha256(pf_model_path)
        self.scene = os.environ.get("PONI_RANK_SCENE", "unspecified")
        self.capture_limit = int(os.environ.get("PONI_RANK_CAPTURE_LIMIT", "0"))
        self.capture_every = max(1, int(os.environ.get("PONI_RANK_CAPTURE_EVERY", "10")))
        self.stride = int(os.environ.get("PONI_RANK_STRIDE", "16"))
        self.refine_topk = int(os.environ.get("PONI_RANK_REFINE_TOPK", "4"))
        self.seq, self.captured = 0, 0
        self._bypass_rank = False
        self.student = None
        self.supported_categories = ()
        if self.has_action_output or self.needs_egocentric_transform or args.use_nearest_frontier:
            raise ValueError("Initial rank branch supports allocentric map output, not actions/egocentric/frontier-only modes")
        if self.rank_mode != "reference":
            checkpoint = os.environ.get("PONI_RANK_CHECKPOINT", "")
            if not checkpoint:
                raise ValueError("A trained --rank-checkpoint is required; random weights are never deployed")
            state = torch.load(checkpoint, map_location="cpu")
            if state["feature_version"] != FEATURE_VERSION or state["contract"] != self.contract:
                raise ValueError("Student feature/configuration contract mismatch")
            if state["teacher_sha256"] != self.teacher_hash:
                raise ValueError("Student was distilled from a different teacher checkpoint")
            self.supported_categories = tuple(map(int, state["supported_categories"]))
            with torch.random.fork_rng(devices=[]):
                self.student = RankSNNNeuron(**state["model_config"])
                self.student.load_state_dict(state["state_dict"])
                self.student.eval()
        self.write({"type": "init", "mode": self.rank_mode, "scene": self.scene,
                    "teacher_sha256": self.teacher_hash, "contract": self.contract,
                    "feature_version": FEATURE_VERSION, "stride": self.stride,
                    "refine_topk": self.refine_topk,
                    "supported_categories": self.supported_categories,
                    "checkpoint": os.environ.get("PONI_RANK_CHECKPOINT", "")})

    def write(self, record):
        with self.rank_log.open("a") as fp:
            fp.write(json.dumps(record, allow_nan=False) + "\n")

    def context(self, inputs, extra_maps, extras, index):
        umap = numpy_value(extra_maps["umap"][index])
        # The evaluator sets dmap=None after the first call unless distance terms
        # are enabled. Ignore the one-off initial dmap too, avoiding a train/serve shift.
        uses_distance = self.contract["add_agent2loc_distance"] or self.contract["add_agent2loc_distance_v2"]
        if uses_distance:
            if extra_maps["dmap"] is None:
                raise ValueError("Enabled distance term requires dmap")
            dmap = numpy_value(extra_maps["dmap"][index])
        else:
            dmap = np.full(umap.shape, np.inf, dtype=np.float32)
        return {"semantic_map": numpy_value(self.do_proc(inputs)[index]),
                "umap": umap, "dmap": dmap,
                "agent": numpy_value(extra_maps["agent_locations"][index]),
                "category": int(extras[index, 1].item()), "contract": self.contract}

    def candidate(self, context):
        tick = time.perf_counter()
        if context["category"] not in self.supported_categories:
            raise ValueError("unseen_goal_category: " + str(context["category"]))
        bank = FeatureBankNeuron(**context)
        cell, info, _, _ = rank_candidates(self.student, bank, self.stride, self.refine_topk)
        info["elapsed_ms"] = (time.perf_counter() - tick) * 1000
        return cell, encode_goal(cell, (bank.h, bank.w)), info

    def act(self, inputs, rnn_hxs, masks, extras=None, extra_maps=None, deterministic=False):
        self.seq += 1
        self._rank_context = (inputs, extra_maps, extras)
        if self.rank_mode != "neuron":
            return super().act(inputs, rnn_hxs, masks, extras, extra_maps, deterministic)
        # This route does NOT call teacher forward unless candidate computation fails.
        records, actions = [], []
        try:
            if self.args.visualize or self.args.print_images:
                raise ValueError("Teacher-free mode does not provide dense PF visualizations")
            for b in range(inputs.shape[0]):
                cell, action, info = self.candidate(self.context(inputs, extra_maps, extras, b))
                actions.append(action)
                records.append({"type": "decision", "seq": self.seq, "batch": b, "mode": "neuron",
                                "used": "neuron", "teacher_called": False, "candidate_cell": cell,
                                "fallback": False, "info": info})
            for record in records:
                self.write(record)
            zeros = torch.zeros(inputs.shape[0], device=inputs.device)
            return zeros, torch.tensor(np.stack(actions), device=inputs.device), zeros, rnn_hxs, {}
        except Exception as exc:
            self.write({"type": "student_failure", "seq": self.seq, "error": type(exc).__name__ + ": " + str(exc)})
            self._bypass_rank = True
            try:
                return super().act(inputs, rnn_hxs, masks, extras, extra_maps, deterministic)
            finally:
                self._bypass_rank = False

    def get_action(self, pfs, goal_cat_id, umap, dmap, agent_locs):
        reference = super().get_action(pfs, goal_cat_id, umap, dmap, agent_locs)
        inputs, extra_maps, extras = self._rank_context
        actions = reference.clone()
        for b in range(len(reference)):
            context = self.context(inputs, extra_maps, extras, b)
            shape = context["umap"].shape
            reference_cell = decode_goal(numpy_value(reference[b]), shape)
            scores = teacher_scores(numpy_value(pfs[b]), context["category"], context["umap"], context["agent"], self.contract)
            record = {"type": "decision", "seq": self.seq, "batch": b, "mode": self.rank_mode,
                      "teacher_called": True, "reference_cell": reference_cell, "used": "reference",
                      "fallback": self._bypass_rank, "reason": "student_failure" if self._bypass_rank else "",
                      "target_visible": bool(context["semantic_map"][context["category"]+2].sum() > 0)}
            if self.capture_limit > self.captured and (self.seq-1) % self.capture_every == 0:
                self.capture(context, scores, reference[b], record)
            if self.rank_mode in ("guarded", "shadow") and not self._bypass_rank:
                try:
                    cell, action, info = self.candidate(context)
                    same = cell == reference_cell
                    record.update(candidate_cell=cell, goal_equal=same, info=info,
                                  teacher_score_at_candidate=float(scores[cell]),
                                  teacher_best_score=float(scores.max()))
                    if self.rank_mode == "guarded":
                        if same:
                            actions[b] = torch.from_numpy(action).to(actions.device)
                            record["used"] = "neuron"
                        else:
                            record.update(fallback=True, reason="goal_cell_disagreement")
                except Exception as exc:
                    record.update(error=type(exc).__name__ + ": " + str(exc),
                                  fallback=self.rank_mode == "guarded", reason="candidate_failure")
            self.write(record)
        return actions

    def capture(self, context, scores, action, record):
        sample = self.rank_root / ("sample_%06d_%d.npz" % (self.seq, record["batch"]))
        if sample.exists():
            raise FileExistsError(sample)
        np.savez_compressed(str(sample), semantic_map=context["semantic_map"].astype(np.float32),
                            umap=context["umap"], dmap=context["dmap"], agent=context["agent"],
                            category=context["category"], teacher_scores=scores,
                            teacher_action=numpy_value(action), scene=np.asarray(self.scene),
                            teacher_sha256=np.asarray(self.teacher_hash),
                            contract_json=np.asarray(json.dumps(self.contract, sort_keys=True)),
                            target_visible=record["target_visible"], seq=self.seq)
        self.captured += 1
        record["sample"] = sample.name
