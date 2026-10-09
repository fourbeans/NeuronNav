#!/usr/bin/env python3
"""Create a Sigma-D resume checkpoint normalized to the next epoch.

Changing DDP world size changes the number of per-rank loader batches.  A
checkpoint written after the final optimizer update of an epoch still records
the old loader's terminal batch index because saving happens before the loop
increments ``epoch``.  This utility validates that exact boundary and writes a
new checkpoint with ``epoch + 1`` and ``next_batch_in_epoch = 0``.  The source
checkpoint is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--expected-next-batch", type=int, required=True)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if source == destination:
        raise RuntimeError("The derived checkpoint must not overwrite its source")
    if destination.exists():
        raise FileExistsError(destination)

    checkpoint = torch.load(str(source), map_location="cpu")
    if checkpoint.get("model_family") != "sigma_d":
        raise RuntimeError("Refusing a non-Sigma-Delta checkpoint")
    if int(checkpoint["global_update"]) != args.expected_update:
        raise RuntimeError(
            f"update mismatch: {checkpoint['global_update']} != {args.expected_update}"
        )
    if int(checkpoint.get("next_batch_in_epoch", -1)) != args.expected_next_batch:
        raise RuntimeError(
            "checkpoint is not at the asserted old-loader epoch boundary: "
            f"{checkpoint.get('next_batch_in_epoch')} != {args.expected_next_batch}"
        )

    old_epoch = int(checkpoint["epoch"])
    checkpoint["epoch"] = old_epoch + 1
    checkpoint["next_batch_in_epoch"] = 0
    checkpoint["resume_normalization"] = {
        "source": str(source),
        "reason": "completed old-world-size epoch boundary",
        "old_epoch": old_epoch,
        "new_epoch": old_epoch + 1,
        "old_next_batch_in_epoch": args.expected_next_batch,
        "new_next_batch_in_epoch": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(destination))
    print(
        f"wrote {destination} update={checkpoint['global_update']} "
        f"epoch={checkpoint['epoch']} next_batch_in_epoch=0"
    )


if __name__ == "__main__":
    main()
