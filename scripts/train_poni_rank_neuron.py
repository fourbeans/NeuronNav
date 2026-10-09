"""Small-scene ranking distillation. Explicit scene-disjoint train/validation.

CPU is the default. Teacher scores label training only; candidate generation at
validation/inference is teacher-free. Fixed epochs; no held-out checkpoint tuning.
"""
import argparse
import glob
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from poni.rank_features_neuron import FEATURE_VERSION, FeatureBankNeuron, decode_goal
from poni.rank_model_neuron import RankSNNNeuron, distillation_loss, rank_candidates


def read_sample(path):
    with np.load(path, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    sample["path"] = str(Path(path).resolve())
    sample["contract"] = json.loads(str(sample["contract_json"]))
    return sample


def make_bank(sample):
    return FeatureBankNeuron(sample["semantic_map"], sample["umap"], sample["dmap"],
                             sample["agent"], int(sample["category"]), sample["contract"])


def split_checks(train, val):
    if not train or not val:
        raise ValueError("Both nonempty train and held-out sets are required")
    train_scenes = {str(x["scene"]) for x in train}
    val_scenes = {str(x["scene"]) for x in val}
    if train_scenes & val_scenes:
        raise ValueError("Scene leakage: " + str(train_scenes & val_scenes))
    if {x["path"] for x in train} & {x["path"] for x in val}:
        raise ValueError("Sample leakage")
    signatures = {(str(x["teacher_sha256"]), str(x["contract_json"])) for x in train + val}
    if len(signatures) != 1:
        raise ValueError("Mixed teacher checkpoints or preprocessing contracts")
    return sorted(train_scenes), sorted(val_scenes)


@torch.no_grad()
def evaluate(model, samples, stride=16, refine_topk=4):
    model.eval()
    rows = []
    for sample in samples:
        bank = make_bank(sample)
        cell, info, candidates, scores = rank_candidates(model, bank, stride, refine_topk)
        teacher = sample["teacher_scores"]
        teacher_at_candidates = teacher[candidates[:, 0], candidates[:, 1]]
        actual_goal = decode_goal(sample["teacher_action"], teacher.shape)
        peak = np.unravel_index(teacher.argmax(), teacher.shape)
        selected_score = float(teacher[cell])
        sorted_student = np.argsort(-scores, kind="stable")
        topk = sorted_student[:min(5, len(scores))]
        row = {"sample": sample["path"], "scene": str(sample["scene"]),
               "goal_category": int(sample["category"]), "student_cell": cell,
               "teacher_decoded_cell": actual_goal, "teacher_peak_cell": list(map(int, peak)),
               "goal_equal": cell == actual_goal,
               "teacher_peak_in_candidates": bool(np.any(np.all(candidates == np.asarray(peak), axis=1))),
               "regret": max(0.0, float(teacher.max()) - selected_score),
               "candidate_oracle_regret": max(0.0, float(teacher.max() - teacher_at_candidates.max())),
               "candidate_top1_equal": bool(selected_score >= teacher_at_candidates.max() - 1e-6),
               "top5_contains_candidate_teacher_best": bool(teacher_at_candidates[topk].max() >= teacher_at_candidates.max() - 1e-6),
               "goal_distance_m": float(np.linalg.norm(np.asarray(cell) - actual_goal) * sample["contract"]["map_resolution"] / 100),
               "info": info}
        rows.append(row)
    return {"samples": len(rows), "exact_goal_agreement": float(np.mean([r["goal_equal"] for r in rows])),
            "mean_pf_regret": float(np.mean([r["regret"] for r in rows])),
            "mean_candidate_oracle_regret": float(np.mean([r["candidate_oracle_regret"] for r in rows])),
            "candidate_top1_agreement": float(np.mean([r["candidate_top1_equal"] for r in rows])),
            "rows": rows}


def training_example(sample, rng, count=512, stride=16):
    bank = make_bank(sample)
    pool = bank.coarse_candidates(stride)
    if not len(pool):
        raise ValueError("No eligible training candidates")
    if len(pool) > count:
        pool = pool[rng.choice(len(pool), count, replace=False)]
    scores = sample["teacher_scores"]
    peak = np.asarray(np.unravel_index(scores.argmax(), scores.shape)).reshape(1, 2)
    # Training-only positive / near-peak negatives. NEVER used in inference/evaluate.
    near = bank.refine_candidates(peak, radius=3)
    if len(near):
        pool = np.unique(np.concatenate([pool, near]), axis=0)
    x = bank.features(pool)
    y = scores[pool[:, 0], pool[:, 1]].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    # Once a goal object is observed, the evaluator ignores PONI's exploration goal.
    train_all = [read_sample(p) for p in sorted(glob.glob(args.train_glob))]
    val_all = [read_sample(p) for p in sorted(glob.glob(args.val_glob))]
    train_scenes, val_scenes = split_checks(train_all, val_all)
    train = [s for s in train_all if not bool(s["target_visible"])]
    val = [s for s in val_all if not bool(s["target_visible"])]
    if not train or not val:
        raise ValueError("No exploration-only examples in one split")
    examples = [training_example(s, rng) for s in train]
    dim = examples[0][0].shape[-1]
    model = RankSNNNeuron(dim, args.hidden, args.steps).to(args.device)
    before = evaluate(model, val)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    tick = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for index in rng.permutation(len(examples)):
            x, target = [t.to(args.device) for t in examples[index]]
            optimizer.zero_grad()
            loss = distillation_loss(model(x), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch+1, "mean_ranking_loss": float(np.mean(losses))}
        history.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    after = evaluate(model, val)
    state = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "model_config": model.config, "feature_version": FEATURE_VERSION,
             "contract": train[0]["contract"], "teacher_sha256": str(train[0]["teacher_sha256"]),
             "train_scenes": train_scenes, "validation_scenes": val_scenes,
             "supported_categories": sorted({int(s["category"]) for s in train}),
             "training_args": vars(args), "navigation_noninferiority_validated": False}
    torch.save(state, out / "student.pt")
    report = {"args": vars(args), "train_scenes": train_scenes, "validation_scenes": val_scenes,
              "supported_categories": state["supported_categories"],
              "train_samples": len(train), "validation_samples": len(val),
              "excluded_target_visible": {"train": len(train_all)-len(train), "validation": len(val_all)-len(val)},
              "train_paths": [s["path"] for s in train], "validation_paths": [s["path"] for s in val],
              "before": before, "after": after, "history": history,
              "training_and_final_validation_seconds": time.time()-tick,
              "checkpoint": str((out / "student.pt").resolve()),
              "warning": "Small-scene feasibility only; offline ranking is not ObjectNav noninferiority."}
    with (out / "training.json").open("x") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps({"checkpoint": report["checkpoint"], "held_out_samples": len(val),
                      "exact_goal_agreement": after["exact_goal_agreement"]}), flush=True)


if __name__ == "__main__":
    main()
