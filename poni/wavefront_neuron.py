"""Single-spike, delay-coded wavefront; software event simulator, not chip code.

Each free cell is an integrate-and-fire neuron with threshold one, no leak,
and inhibition until the next planning query. A unit input triggers its first
spike; axon delay is geometric length times destination traversal weight.
The event queue coalesces arrivals already dominated by an earlier scheduled
arrival. This is an exact simulation optimization for this single-spike model.
Mathematically the first-arrival times solve the positive-weight graph shortest
path problem (Dijkstra/min-plus). No E-prop or hardware speedup is claimed.
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

    def njit(**kwargs):
        return lambda fn: fn


@njit(cache=True)
def _events(free, weights, seeds, targets, reverse, quantum, max_spikes):
    h, w = free.shape
    times = np.full(h * w, np.inf)
    scheduled = np.full(h * w, np.inf)
    parents = np.full(h * w, -1, dtype=np.int64)
    fired = np.zeros(h * w, dtype=np.uint8)
    queue = [(0.0, np.int64(-1), np.int64(-1))]
    heapq.heappop(queue)
    for src in seeds:
        scheduled[src] = 0.0
        heapq.heappush(queue, (0.0, src, np.int64(-1)))
    nspikes, emitted, popped = 0, 0, 0
    goal = -1
    exhausted_budget = False
    while queue:
        t, node, parent = heapq.heappop(queue)
        popped += 1
        if fired[node] or t > scheduled[node]:
            continue
        if max_spikes > 0 and nspikes >= max_spikes:
            exhausted_budget = True
            break
        fired[node] = 1
        times[node] = t
        parents[node] = parent
        nspikes += 1
        r, c = node // w, node % w
        if targets[r, c]:
            goal = node
            break
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or not free[nr, nc]:
                    continue
                nxt = nr * w + nc
                emitted += 1
                if fired[nxt]:
                    continue
                length = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
                # Reverse propagation must use the reversed DIRECTED edge.
                weight = weights[r, c] if reverse else weights[nr, nc]
                delay = length * weight
                if quantum > 0:
                    delay = max(quantum, math.floor(delay / quantum + 0.5) * quantum)
                arrival = t + delay
                if arrival < scheduled[nxt]:
                    scheduled[nxt] = arrival
                    heapq.heappush(queue, (arrival, np.int64(nxt), np.int64(node)))
    return times, parents, nspikes, emitted, popped, goal, exhausted_budget


@dataclass
class WaveResult:
    times: np.ndarray
    parents: np.ndarray
    reached: int
    spikes: int
    emitted: int
    popped: int
    budget_exhausted: bool

    def path_to(self, cell):
        """Return seed -> cell, or an empty path if the cell did not spike."""
        h, w = self.times.shape
        r, c = map(int, cell)
        if not (0 <= r < h and 0 <= c < w) or not np.isfinite(self.times[r, c]):
            return []
        node = r * w + c
        path = []
        for _ in range(h * w + 1):
            path.append((node // w, node % w))
            node = int(self.parents.flat[node])
            if node == -1:
                return path[::-1]
        raise RuntimeError("Cycle in wavefront predecessor chain")


def propagate(traversible, seeds, weights=None, targets=None, reverse=False,
              delay_quantum=0.0, max_spikes=0):
    free = np.ascontiguousarray(traversible, dtype=np.bool_)
    if free.ndim != 2 or 0 in free.shape:
        raise ValueError("Expected a nonempty 2-D traversibility map")
    weights = np.ones(free.shape) if weights is None else np.asarray(weights, dtype=np.float64)
    targets = np.zeros(free.shape, dtype=np.bool_) if targets is None else np.asarray(targets, dtype=np.bool_)
    if weights.shape != free.shape or targets.shape != free.shape:
        raise ValueError("Map, weights and targets must have identical shapes")
    if not np.all(np.isfinite(weights[free])) or np.any(weights[free] <= 0):
        raise ValueError("Free-cell traversal weights must be finite and positive")
    if not np.isfinite(delay_quantum) or delay_quantum < 0:
        raise ValueError("delay_quantum must be nonnegative and finite")
    if max_spikes < 0:
        raise ValueError("max_spikes must be nonnegative")
    source_ids = []
    for r, c in seeds:
        r, c = int(r), int(c)
        if not (0 <= r < free.shape[0] and 0 <= c < free.shape[1]) or not free[r, c]:
            raise ValueError("A source neuron is outside the free map")
        source_ids.append(r * free.shape[1] + c)
    if not source_ids:
        raise ValueError("At least one source neuron is required")
    out = _events(free, np.ascontiguousarray(weights),
                  np.asarray(sorted(set(source_ids)), dtype=np.int64),
                  np.ascontiguousarray(targets), bool(reverse), float(delay_quantum), int(max_spikes))
    t, p, ns, ne, npop, reached, budget = out
    return WaveResult(t.reshape(free.shape), p.reshape(free.shape),
                      int(reached), int(ns), int(ne), int(npop), bool(budget))


def warmup():
    propagate(np.ones((3, 3), dtype=bool), [(0, 0)])
