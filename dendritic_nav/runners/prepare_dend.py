"""Initialize a new experiment and deterministic scene-disjoint train/dev split."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import torch
from dendritic_nav.core.cell_dend import DendriticScorer, MODEL_FAMILY
from .common_dend import new_run, atomic_checkpoint, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spatial-checkpoint", required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--dend-checkpoint", help="Optional v2 initialization for a new run")
    args = p.parse_args()
    root = Path(args.root)
    content = root / "data/datasets/objectnav/gibson/v1.1/train/content"
    scenes = sorted(path.name[:-8] for path in content.glob("*.json.gz"))
    if len(scenes) < 4:
        raise ValueError("Need at least four installed training scenes")
    missing = [
        scene
        for scene in scenes
        if not (root / "data/scene_datasets/gibson_semantic" / scene).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing training scene assets: " + repr(missing))
    ranked = sorted(
        scenes, key=lambda s: hashlib.sha256(("dend-v1:" + s).encode()).hexdigest()
    )
    count = max(2, round(0.1 * len(scenes)))
    dev = sorted(ranked[:count])
    train = sorted(ranked[count:])
    spatial = Path(args.spatial_checkpoint).resolve()
    if not spatial.exists():
        raise FileNotFoundError(str(spatial))
    out = new_run(
        args.output,
        args,
        __name__.replace("__main__", "dendritic_nav.runners.prepare_dend"),
    )
    with zipfile.ZipFile(
        out / "source_snapshot.zip", "x", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted((root / "dendritic_nav").rglob("*.py")):
            archive.write(path, str(path.relative_to(root)))
    (out / "scene_split.json").write_text(
        json.dumps(
            dict(
                train=train,
                dev=dev,
                official_source="train",
                rule="sha256(dend-v1:scene), first rounded 10%, min 2",
            ),
            indent=2,
        )
    )
    (out / "spatial_checkpoint.json").write_text(
        json.dumps(dict(path=str(spatial), sha256=sha256(spatial)), indent=2)
    )
    torch.manual_seed(args.seed)
    model = DendriticScorer()
    if args.dend_checkpoint:
        source = torch.load(args.dend_checkpoint, map_location="cpu")
        if source.get("model_family") != MODEL_FAMILY:
            raise ValueError("Initialization checkpoint has wrong model family")
        model.load_state_dict(source["state_dict"])
        (out / "dend_initialization.json").write_text(
            json.dumps(dict(path=str(Path(args.dend_checkpoint).resolve()),
                            sha256=sha256(Path(args.dend_checkpoint))), indent=2)
        )
    atomic_checkpoint(
        dict(
            model_family=MODEL_FAMILY,
            state_dict=model.state_dict(),
            environment_steps=0,
            updates=0,
            seed=args.seed,
        ),
        out / "checkpoints/initial.ckpt",
    )
    atomic_checkpoint(
        dict(
            model_family=MODEL_FAMILY,
            state_dict=model.state_dict(),
            environment_steps=0,
            updates=0,
            seed=args.seed,
        ),
        out / "checkpoints/published.ckpt",
    )
    print(json.dumps(dict(output=str(out), train_scenes=len(train), dev_scenes=dev)))


if __name__ == "__main__":
    main()
