"""Plot a named MP3D episode even when one run stops before 20 steps."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_paired_trace_neuron import _poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = (str(args.part), args.episode)
    new = _poses(args.new_root, "mp3d")[key]
    old = _poses(args.old_root, "mp3d")[key]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, limit, title in ((axes[0], 40, "First 40 steps"),
                             (axes[1], max(len(new), len(old)), "Whole episode")):
        a, b = new[:limit], old[:limit]
        ax.plot(b[:, 0], b[:, 1], "-", color="#2474b5", label=f"old SNN ({len(old)} steps)")
        ax.plot(a[:, 0], a[:, 1], "-", color="#278163", label=f"tie-break ({len(new)} steps)")
        ax.scatter([a[0, 0]], [a[0, 1]], marker="*", s=100, c="#d18b2a", label="shared start")
        ax.scatter([a[-1, 0]], [a[-1, 1]], s=25, c="#278163")
        ax.scatter([b[-1, 0]], [b[-1, 1]], s=25, c="#2474b5")
        ax.set(xlabel="map x (m)", ylabel="map y (m)", title=title)
        ax.axis("equal")
        ax.legend(fontsize=7)
    fig.suptitle(f"MP3D part{args.part}/episode{args.episode}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(str(args.output))


if __name__ == "__main__":
    main()
