"""CPU checks for true spike forward, reset/gradients, teacher-free features,
upstream mask/goal decoding, student bypass, and guard/fallback behavior.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch import nn
from poni.rank_features_neuron import (FeatureBankNeuron, teacher_scores, decode_goal, encode_goal)
from poni.rank_model_neuron import SpikeNeuron, RankSNNNeuron, distillation_loss, rank_candidates
from semexp.model_pf import RL_Policy
from semexp.model_pf_rank_neuron import RankPolicyNeuron, CONTRACT_KEYS
from train_poni_rank_neuron import split_checks


def args():
    return NS(pf_masking_opt="unexplored", mask_nearest_locations=False, mask_size=1.0,
              map_resolution=5, area_weight_coef=0.5, dist_weight_coef=0.3,
              add_agent2loc_distance=False, add_agent2loc_distance_v2=False,
              use_egocentric_transform=False, num_sem_categories=16,
              visualize=False, print_images=False, use_nearest_frontier=False)


class TeacherFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.pf_model = NS(cfg=NS(MODEL=NS(output_type="map")))
        self.calls = 0

    def forward(self, inputs, *unused):
        self.calls += 1
        fields = torch.zeros_like(inputs)
        fields[:, :, 10, 10] = 0.9
        return fields, None


def fixture(cls, mode="reference", cell=(10, 10), fail=False):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    obj.args = args()
    obj.network = TeacherFixture()
    obj._cached_visualizations = None
    if cls == RankPolicyNeuron:
        obj.contract = {k: getattr(obj.args, k) for k in CONTRACT_KEYS}
        obj.rank_mode, obj.seq, obj.captured = mode, 0, 0
        obj.capture_limit, obj.capture_every = 0, 10
        obj._bypass_rank = False
        obj.records = []
        obj.write = obj.records.append
        def candidate(context):
            assert np.isinf(context["dmap"]).all()  # default dmap missing/ignored
            if fail:
                raise RuntimeError("injected_failure")
            return cell, encode_goal(cell, (32, 32)), {}
        obj.candidate = candidate
    return obj


def controller_checks():
    inp = torch.zeros(2, 24, 32, 32)
    extras = torch.tensor([[0., 0.], [0., 1.]])
    maps = {"umap": torch.ones(2, 32, 32), "dmap": None,
            "agent_locations": [[16, 16], [16, 16]], "ego_agent_poses": None}
    base = fixture(RL_Policy)
    reference = base.act(inp, None, torch.ones(2), extras, maps)
    cells = [decode_goal(x, (32, 32)) for x in reference[1].numpy()]
    results = []
    for mode, cell, fail in [("reference", (11, 10), False), ("shadow", (11, 10), False),
                              ("guarded", (10, 10), False), ("guarded", (11, 10), False),
                              ("guarded", (10, 10), True), ("neuron", (11, 10), False),
                              ("neuron", (10, 10), True)]:
        obj = fixture(RankPolicyNeuron, mode, cell, fail)
        out = obj.act(inp, None, torch.ones(2), extras, maps)
        actual = [decode_goal(x, (32, 32)) for x in out[1].numpy()]
        expected = [(11, 10)] * 2 if mode == "neuron" and not fail else cells
        assert actual == expected, (mode, cell, fail, actual)
        assert obj.network.calls == (0 if mode == "neuron" and not fail else 1)
        assert torch.equal(out[0], reference[0]) and torch.equal(out[2], reference[2])
        assert len([r for r in obj.records if r["type"] == "decision"]) == 2
        results.append({"mode": mode, "candidate": cell, "injected_failure": fail, "passed": True})
    # Initial populated dmap must have identical features to later None.
    obj = fixture(RankPolicyNeuron)
    a = obj.context(inp, maps, extras, 0)
    maps["dmap"] = torch.ones(2, 32, 32)
    b = obj.context(inp, maps, extras, 0)
    assert np.array_equal(a["dmap"], b["dmap"])
    obj.supported_categories = (4,)
    try:
        RankPolicyNeuron.candidate(obj, {"category": 0})
    except ValueError as exc:
        assert "unseen_goal_category" in str(exc)
    else:
        raise AssertionError("Untrained category accepted")
    return results


def mask_checks():
    rng = np.random.RandomState(11)
    policy = fixture(RL_Policy)
    for case in range(32):
        h, w = 32, 40
        pfs = rng.rand(1, 17, h, w).astype(np.float32)
        if case % 8 == 0:
            pfs.fill(0)  # all-zero tie has exact upstream row-major behavior
        umap = rng.rand(h, w).astype(np.float32)
        cat = case % 15
        agent = (case % h, (case*3) % w)
        policy.args.mask_nearest_locations = bool(case % 2)
        policy.args.pf_masking_opt = "unexplored" if case % 3 else "none"
        contract = {k: getattr(policy.args, k) for k in CONTRACT_KEYS}
        action = policy.get_action(torch.tensor(pfs), torch.tensor([cat]), torch.tensor(umap[None]), None, [agent])
        scores = teacher_scores(pfs[0], cat, umap, agent, contract)
        r, c = np.unravel_index(scores.argmax(), scores.shape)
        expected_action = np.array([r/h, c/w], np.float32)
        assert np.array_equal(action.numpy()[0], expected_action)
    # Actual decoder round-trip including float32 truncation boundaries.
    for n in (32, 40, 480, 481):
        for r in range(n):
            cell = (r, n-1-r)
            assert decode_goal(encode_goal(cell, (n, n)), (n, n)) == cell
    return {"original_mask_cases": 32, "coordinate_roundtrips": 2*(32+40+480+481)}


def core_checks():
    torch.manual_seed(7)
    voltage = torch.linspace(-2, 2, 21, requires_grad=True)
    spike = SpikeNeuron.apply(voltage)
    assert set(spike.tolist()) == {0.0, 1.0}
    spike.sum().backward()
    assert torch.isfinite(voltage.grad).all() and (voltage.grad > 0).all()
    model = RankSNNNeuron(12, hidden=24, steps=8)
    x = torch.rand(64, 12)
    target = (0.7*x[:, 0] + 0.3*x[:, 1]).detach()
    first, activity = model(x, return_activity=True)
    model(torch.rand(33, 12))
    assert torch.equal(first, model(x)), "membrane leaked across calls"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    initial = float(distillation_loss(first, target).detach())
    for step in range(120):
        optimizer.zero_grad()
        loss = distillation_loss(model(x), target)
        loss.backward()
        for parameter in (model.fc1.weight, model.fc2.weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        optimizer.step()
    final = float(distillation_loss(model(x), target).detach())
    assert final < initial - 0.1, (initial, final)
    config = {k: getattr(args(), k) for k in CONTRACT_KEYS}
    semantic = np.zeros((17, 48, 48), np.float32)
    semantic[0, 10:30, 12:32] = 1
    semantic[4, 20:22, 20:22] = 1
    bank = FeatureBankNeuron(semantic, np.ones((48,48)), np.full((48,48), np.inf), (24,24), 2, config)
    points = bank.coarse_candidates(8)
    features = bank.features(points)
    assert features.shape[1] == bank.dim and np.isfinite(features).all()
    # No teacher argument exists; arbitrary teacher values cannot alter features/candidates.
    unrelated_teacher = np.ones((17,48,48))
    unrelated_teacher *= 99
    assert np.array_equal(points, bank.coarse_candidates(8)) and np.array_equal(features, bank.features(points))
    probe = RankSNNNeuron(bank.dim)
    a = rank_candidates(probe, bank, 8, 2)
    b = rank_candidates(probe, bank, 8, 2)
    assert a[0] == b[0] and np.array_equal(a[2], b[2])
    empty = FeatureBankNeuron(semantic, np.zeros((48,48)), np.full((48,48), np.inf), (24,24), 2, config)
    try:
        rank_candidates(probe, empty)
    except ValueError:
        pass
    else:
        raise AssertionError("Empty candidate set accepted")
    same = [{"scene": "same", "path": "a", "teacher_sha256": "teacher", "contract_json": "{}"}]
    try:
        split_checks(same, same)
    except ValueError:
        pass
    else:
        raise AssertionError("Scene leakage accepted")
    return {"binary_spikes_and_surrogate": "passed", "state_reset": "passed",
            "initial_synthetic_loss": initial, "final_synthetic_loss": final,
            "spike_activity": activity, "teacher_free_candidates": "passed",
            "empty_candidates_and_split_rejection": "passed"}


def reference_equivalence(capture, previous):
    def read_trace(folder, pattern):
        actions, episodes = {}, {}
        for path in sorted(Path(folder).glob(pattern)):
            for row in map(json.loads, path.read_text().splitlines()):
                if row["type"] == "action":
                    actions[(row["scene"], row["episode_no"], row["step"])] = row["action"]
                elif row["type"] == "episode_start":
                    episodes[(row["scene"], row["episode_no"])] = (row["position"], row["rotation"], row["goal_ids"])
        return actions, episodes
    report = []
    for scene in ("Corozal", "Darden"):
        a, ea = read_trace(Path(capture) / "reference" / scene, "rank_audit/low_level_fmm/*.jsonl")
        b, eb = read_trace(Path(previous) / "reference" / scene, "audit/*.jsonl")
        if not a or not b or not ea or not eb:
            raise ValueError("Missing reference comparison traces")
        row = {"scene": scene, "same_action_keys": set(a) == set(b), "actions": len(a),
               "action_mismatches": sum(a[k] != b[k] for k in set(a) & set(b)),
               "same_episode_keys": set(ea) == set(eb),
               "episode_mismatches": sum(ea[k] != eb[k] for k in set(ea) & set(eb))}
        assert row["same_action_keys"] and row["action_mismatches"] == 0
        assert row["same_episode_keys"] and row["episode_mismatches"] == 0
        report.append(row)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-capture")
    parser.add_argument("--previous-reference")
    options = parser.parse_args()
    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    report = {"core": core_checks(), "upstream_masks": mask_checks(), "controller": controller_checks(), "status": "passed"}
    if options.reference_capture or options.previous_reference:
        if not options.reference_capture or not options.previous_reference:
            parser.error("Both reference paths are required")
        report["live_reference_equivalence"] = reference_equivalence(options.reference_capture, options.previous_reference)
    with (out / "validation.json").open("x") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
