import numpy as np


def plot_episode(episode, output, map_snapshot=None, resolution=0.05):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    xy = episode["agent_xy"]
    if map_snapshot is not None:
        obstacle, explored = map_snapshot
        background = np.where(explored >= 0.5, 0.0, 0.5)
        background[obstacle >= 0.5] = 1.0
        height, width = background.shape
        axes[0, 0].imshow(
            background,
            cmap="gray_r",
            vmin=0,
            vmax=1,
            extent=(0, width * resolution, height * resolution, 0),
        )
    axes[0, 0].plot(xy[:, 0], xy[:, 1], linewidth=1)
    goals = episode["goal_xy"][episode["action"] >= 0]
    if len(goals):
        axes[0, 0].scatter(
            goals[:, 0], goals[:, 1], s=5, alpha=0.4, label="Selected goals"
        )
    axes[0, 0].set(
        title="Episode trajectory (metric coordinates)", xlabel="x (m)", ylabel="y (m)"
    )
    axes[0, 0].axis("equal")
    activity = episode["activity"].reshape(len(xy), 16).T
    axes[0, 1].imshow(activity, aspect="auto", interpolation="nearest")
    axes[0, 1].set(
        title="4 branches x 4 compartments: mean |event|", xlabel="Real navigation step"
    )
    axes[0, 1].set_yticks([1.5, 5.5, 9.5, 13.5])
    axes[0, 1].set_yticklabels(["Semantic", "Explore", "Motion", "Decision"])
    for name, color in (("collision", "red"), ("goal_switch", "orange")):
        for step in np.flatnonzero(episode.get(name, np.zeros(len(xy), dtype=bool))):
            axes[1, 0].axvline(step, color=color, alpha=0.15, linewidth=0.5)
    axes[1, 0].plot(
        np.cumsum(episode["reward"]), label="Undiscounted cumulative reward"
    )
    axes[1, 0].plot(
        episode["override"].astype(float), alpha=0.4, label="Target-visible override"
    )
    axes[1, 0].set_title("Reward; red=collision, orange=goal switch")
    axes[1, 0].legend(fontsize=8)
    decision = np.flatnonzero(episode["action"] >= 0)
    t = int(decision[len(decision) // 2]) if len(decision) else 0
    valid = np.flatnonzero(episode["valid"][t])
    top = valid[np.argsort(-episode["spatial"][t, valid], kind="stable")[:10]]
    x = np.arange(len(top))
    for offset, key, label in [
        (-0.25, "spatial", "Spatial"),
        (0, "residual", "Dendritic residual"),
    ]:
        axes[1, 1].bar(x + offset, episode[key][t, top], width=0.25, label=label)
    axes[1, 1].bar(
        x + 0.25,
        (episode["spatial"] + episode["residual"])[t, top],
        width=0.25,
        label="Final",
    )
    axes[1, 1].set(
        title="Top spatial candidates at step {}".format(t), xlabel="Candidate"
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output), dpi=130)
    plt.close(fig)
