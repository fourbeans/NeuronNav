"""Independent recurrent dendritic learner consuming completed actor episodes."""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dendritic_nav.core.cell_dend import DendriticScorer, MODEL_FAMILY
from dendritic_nav.learning.td_v4_dend import training_loss
from dendritic_nav.learning.replay_dend import EpisodeReplay
from .common_dend import new_run, atomic_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--update-every", type=int, default=64)
    p.add_argument("--target-sync-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--idle-timeout", type=int, default=3600)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    experiment = Path(args.experiment)
    out = experiment / "learner"
    if not args.resume:
        new_run(out, args, "dendritic_nav.runners.train_dend")
    online = DendriticScorer().to(args.device)
    initial = torch.load(
        str(experiment / "checkpoints/initial.ckpt"), map_location=args.device
    )
    if initial.get("model_family") != MODEL_FAMILY:
        raise ValueError("Training checkpoint has wrong model family")
    online.load_state_dict(initial["state_dict"])
    target = copy.deepcopy(online).requires_grad_(False).eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    updates = 0
    if args.resume:
        saved = torch.load(str(out / "checkpoints/last.ckpt"), map_location=args.device)
        online.load_state_dict(saved["state_dict"])
        target.load_state_dict(saved["target_state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        updates = int(saved["updates"])
        rng.set_state(saved["sample_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    replay = EpisodeReplay()
    writer = SummaryWriter(str(out / "tensorboard"))
    start = time.time()
    last_change = start
    last_seen = -1
    last_publish = 0

    def save(final=False):
        nonlocal last_publish
        payload = dict(
            model_family=MODEL_FAMILY,
            state_dict=online.state_dict(),
            target_state_dict=target.state_dict(),
            optimizer=optimizer.state_dict(),
            updates=updates,
            environment_steps=replay.total_seen,
            config=vars(args),
            sample_rng=rng.get_state(),
            torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            replay_files=sorted(replay.seen),
        )
        atomic_checkpoint(payload, out / "checkpoints/last.ckpt")
        if updates and updates % 100 == 0:
            atomic_checkpoint(
                payload, out / "checkpoints/update_{:05d}.ckpt".format(updates)
            )
        if final:
            atomic_checkpoint(payload, out / "checkpoints/pilot_final.ckpt")
        if final or replay.total_seen - last_publish >= 2000:
            atomic_checkpoint(payload, experiment / "checkpoints/published.ckpt")
            last_publish = replay.total_seen

    try:
        while True:
            replay.ingest(experiment)
            if replay.total_seen != last_seen:
                last_seen = replay.total_seen
                last_change = time.time()
            eligible = max(
                0,
                (min(replay.total_seen, args.steps) - args.warmup) // args.update_every,
            )
            if updates >= eligible:
                if replay.total_seen >= args.steps:
                    save(final=True)
                    break
                if time.time() - last_change > args.idle_timeout:
                    save()
                    raise RuntimeError(
                        "No new complete episodes within idle timeout; inspect actors"
                    )
                time.sleep(2)
                continue
            samples = replay.sample(rng, args.batch, args.window)
            loss, metrics = training_loss(online, target, samples, args.device)
            if loss is None:
                raise RuntimeError(
                    "Sampled batch has no executed residual decisions; inspect override rate"
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite dendritic TD loss")
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            optimizer.step()
            updates += 1
            if updates % args.target_sync_every == 0:
                target.load_state_dict(online.state_dict())
            metrics.update(
                update=updates,
                environment_steps=replay.total_seen,
                replay_steps=replay.steps,
                grad_norm=float(grad),
                target_syncs=updates // args.target_sync_every,
                seconds=time.time() - start,
            )
            with open(out / "metrics.jsonl", "a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            for name, value in metrics.items():
                writer.add_scalar(name, value, updates)
            if updates % 10 == 0:
                print("DEND_TRAIN " + json.dumps(metrics), flush=True)
            if updates % 50 == 0:
                save()
        (out / "completion.json").write_text(
            json.dumps(
                dict(
                    updates=updates,
                    steps=replay.total_seen,
                    seconds=time.time() - start,
                )
            )
        )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
