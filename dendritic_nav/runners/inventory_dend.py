"""Hash old sources/selected checkpoints, and validate they remain unchanged."""
import argparse
import json
from pathlib import Path
from .common_dend import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    output = Path(args.output)
    if args.verify:
        hashes = json.loads(output.read_text())
        changed = [
            name
            for name, value in hashes.items()
            if not (root / name).exists() or sha256(root / name) != value
        ]
        print(json.dumps(dict(checked=len(hashes), changed=changed), indent=2))
        if changed:
            raise SystemExit(1)
    else:
        files = []
        for folder in ("poni", "semexp", "hlab", "configs"):
            files.extend(
                path
                for path in (root / folder).rglob("*")
                if path.suffix in (".py", ".yaml")
            )
        for relative in (
            "dairy/sigma_d_unet_20260924/experiments/gibson_full_q4095_lr1e5_seed234_w4_from30k/checkpoints/best.ckpt",
            "dairy/sigma_d_unet_20260924/experiments/mp3d_full_q4095_lr1e5_seed123_w4_from10k/checkpoints/best.ckpt",
        ):
            files.append(root / relative)
        hashes = {str(path.relative_to(root)): sha256(path) for path in sorted(files)}
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(hashes, stream, indent=2)
        print("Recorded {} immutable source/checkpoint hashes".format(len(hashes)))


if __name__ == "__main__":
    main()
