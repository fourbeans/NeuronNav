"""Complete-episode replay: exact prefix reconstruction, no cached hidden states."""
from collections import deque
from pathlib import Path
import numpy as np


class EpisodeReplay:
    def __init__(self, capacity=500000):
        self.capacity = capacity
        self.episodes = deque()
        self.steps = 0
        self.seen = set()
        self.total_seen = 0

    def ingest(self, directory):
        for path in sorted(Path(directory).glob("actor_*/replay/episode_*.npz")):
            if str(path) in self.seen:
                continue
            with np.load(path, allow_pickle=False) as archive:
                episode = {k: archive[k] for k in archive.files}
            validate_episode(episode)
            self.seen.add(str(path))
            n = len(episode["reward"])
            self.total_seen += n
            self.episodes.append(episode)
            self.steps += n
            while self.steps > self.capacity and len(self.episodes) > 1:
                self.steps -= len(self.episodes.popleft()["reward"])

    def sample(self, rng, batch=16, length=32):
        # Uniform across real environment steps, not across episode lengths.
        lengths = np.asarray([len(e["reward"]) for e in self.episodes])
        if not len(lengths):
            raise ValueError("empty replay")
        cumsum = lengths.cumsum()
        samples = []
        for _ in range(batch):
            flat = int(rng.randint(cumsum[-1]))
            index = int(np.searchsorted(cumsum, flat, side="right"))
            start = flat - (int(cumsum[index - 1]) if index else 0)
            samples.append(
                (self.episodes[index], start, min(start + length, int(lengths[index])))
            )
        return samples


def validate_episode(e):
    n = len(e["reward"])
    if n == 0 or not bool(e["terminal"][-1]) or e["terminal"][:-1].any():
        raise ValueError(
            "Replay requires exactly one final terminal, not a collection cutoff"
        )
    if not np.isfinite(e["features"]).all() or not np.isfinite(e["reward"]).all():
        raise ValueError("nonfinite replay")
    active = e["action"] >= 0
    if np.any(e["action"][active] >= 128):
        raise ValueError("action outside candidates")
    if not e["valid"][np.flatnonzero(active), e["action"][active]].all():
        raise ValueError("action selects padding")


def return_target(episode, start, gamma=0.995, decisions=3):
    """Three decision transitions, discounting every real step (including overrides)."""
    total, discount, count = 0.0, 1.0, 0
    for t in range(start, len(episode["reward"])):
        if t > start and episode["action"][t] >= 0:
            count += 1
            if count == decisions:
                return total, discount, t
        total += discount * float(episode["reward"][t])
        discount *= gamma
        if episode["terminal"][t]:
            return total, 0.0, None
    raise ValueError("Incomplete episode")
