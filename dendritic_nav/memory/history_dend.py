"""Sparse global metric-coordinate statistics; independent of local window."""
from collections import defaultdict
import numpy as np


class CoordinateHistory:
    def __init__(self, cell_size=0.25):
        self.cell_size = cell_size
        self.cells = defaultdict(lambda: np.zeros(6, dtype=np.float32))
        self.entries = defaultdict(int)
        self.last_agent_key = None
        self.goals = []
        self.goal = None
        self.age = self.switches = self.steps = self.stall = 0
        self.best_distance = float("inf")
        self.stall_reported = False

    def key(self, xy):
        return tuple(np.floor(np.asarray(xy) / self.cell_size).astype(int))

    def lookup(self, xy):
        return self.cells.get(self.key(xy), np.zeros(6, dtype=np.float32)).copy()

    def entry_count(self, xy):
        """Distinct visits to a cell, excluding consecutive dwell steps."""
        return self.entries.get(self.key(xy), 0)

    def choose(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        if self.goal is None or self.key(xy) != self.key(self.goal):
            self.switches += int(self.goal is not None)
            self.goal = xy.copy()
            self.age = self.stall = 0
            self.stall_reported = False
            self.best_distance = float("inf")
            self.goals = [xy.copy()] + self.goals[:15]
            self.cells[self.key(xy)][3] += 1

    def advance(self, xy, collision=False, new_target=False):
        self.steps += 1
        self.age += int(self.goal is not None)
        key = self.key(xy)
        if key != self.last_agent_key:
            self.entries[key] += 1
            self.last_agent_key = key
        cell = self.cells[key]
        cell[0] += 1
        cell[1] = self.steps
        cell[2] += int(collision)
        if self.goal is not None:
            distance = float(np.linalg.norm(np.asarray(xy) - self.goal))
            if new_target or self.best_distance - distance >= 0.1:
                self.best_distance = distance
                self.stall = 0
                self.stall_reported = False
            else:
                self.stall += 1
            if self.stall >= 10 and not self.stall_reported:
                target = self.cells[self.key(self.goal)]
                target[4] += 1
                target[5] = -1
                self.stall_reported = True

    def finish(self, success):
        if self.goal is not None:
            self.cells[self.key(self.goal)][5] = 1 if success else -1


def rc_to_xy(rc, origin, resolution):
    return np.asarray(origin) + np.asarray(rc)[..., ::-1] * resolution


def xy_to_rc(xy, origin, resolution):
    return np.rint((np.asarray(xy) - np.asarray(origin)) / resolution)[
        ..., ::-1
    ].astype(int)
