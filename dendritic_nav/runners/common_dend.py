import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def new_run(directory, args, kind):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.sh").write_text(
        "#!/bin/sh\n" + shlex.join([sys.executable, "-m", kind] + sys.argv[1:]) + "\n"
    )
    (directory / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    manifest = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        kind=kind,
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        schema="dendritic_v3",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    package = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(path.relative_to(package)): sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    (directory / "code_sha256.json").write_text(json.dumps(code_hashes, indent=2))
    return directory
