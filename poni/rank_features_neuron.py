"""Teacher-free candidate features and coarse-to-fine candidate generation.

All spatial preprocessing is fixed (not learned): box averages, coordinates,
goal category and map masks. No potential field is accepted by this module.
"""
import cv2
import numpy as np

FEATURE_VERSION = "semantic_box_1_17_65_global_v1"


def selection_mask(umap, agent, contract):
    mask = np.asarray(umap, np.float32).copy() if contract["pf_masking_opt"] == "unexplored" else np.ones_like(umap, dtype=np.float32)
    if contract["mask_nearest_locations"]:
        r, c = map(int, agent)
        size = int(contract["mask_size"] * 100.0 / contract["map_resolution"])
        # Match original slices, including the negative-index boundary behavior.
        mask[r-size:r+size+1, c-size:c+size+1] = 0
    return mask


def teacher_scores(pfs, category, umap, agent, contract):
    """Training/diagnostics ONLY; never part of candidate features."""
    return np.asarray(pfs[int(category)+2], np.float32) * selection_mask(umap, agent, contract)


class FeatureBankNeuron:
    def __init__(self, semantic_map, umap, dmap, agent, category, contract):
        self.maps = np.asarray(semantic_map, dtype=np.float32)
        self.umap = np.asarray(umap, dtype=np.float32)
        self.dmap = np.asarray(dmap, dtype=np.float32)
        self.agent = np.asarray(agent, dtype=np.float32)
        self.category = int(category)
        if self.maps.ndim != 3 or self.maps.shape[0] < 3:
            raise ValueError("Expected free/obstacle/semantic CHW map")
        self.channels, self.h, self.w = self.maps.shape
        if self.umap.shape != (self.h, self.w) or self.dmap.shape != self.umap.shape:
            raise ValueError("Map dimensions differ")
        if not np.isfinite(self.maps).all() or not np.isfinite(self.umap).all():
            raise ValueError("Nonfinite semantic or unexplored map")
        if not 0 <= self.category < self.channels - 2:
            raise ValueError("Invalid goal category")
        self.mask = selection_mask(self.umap, self.agent, contract)
        self.valid = np.isfinite(self.mask) & (self.mask > 0)
        self.scales = [self.maps]
        for size in (17, 65):
            self.scales.append(np.stack([cv2.boxFilter(ch, -1, (size, size), normalize=True,
                                                     borderType=cv2.BORDER_CONSTANT) for ch in self.maps]))
        self.global_means = self.maps.mean(axis=(1, 2))

    @property
    def dim(self):
        return 4 * self.channels + (self.channels - 2) + 8

    def features(self, coords):
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
        r, c = coords.T
        if (r < 0).any() or (c < 0).any() or (r >= self.h).any() or (c >= self.w).any():
            raise ValueError("Candidate outside map")
        n = len(coords)
        local = [scale[:, r, c].T for scale in self.scales]
        global_context = np.broadcast_to(self.global_means, (n, self.channels))
        category = np.zeros((n, self.channels - 2), np.float32)
        category[:, self.category] = 1
        dr, dc = (r - self.agent[0]) / self.h, (c - self.agent[1]) / self.w
        distance = self.dmap[r, c]
        finite = np.isfinite(distance)
        distance = np.log1p(np.clip(np.nan_to_num(distance, nan=1e4, posinf=1e4, neginf=0), 0, 1e4)) / np.log1p(1e4)
        scalar = np.column_stack([r / self.h, c / self.w, dr, dc, np.hypot(dr, dc),
                                  self.umap[r, c], distance, finite.astype(np.float32)])
        return np.ascontiguousarray(np.concatenate(local + [global_context, category, scalar], axis=1), dtype=np.float32)

    def coarse_candidates(self, stride=16, frontier_count=256):
        if stride < 1:
            raise ValueError("stride must be positive")
        rr, cc = np.meshgrid(np.arange(0, self.h, stride), np.arange(0, self.w, stride), indexing="ij")
        points = np.column_stack([rr.ravel(), cc.ravel()])
        # Map-derived boundaries, not teacher-derived peaks.
        known = (self.umap < 0.5).astype(np.uint8)
        border = cv2.dilate(known, np.ones((3, 3), np.uint8)) != cv2.erode(known, np.ones((3, 3), np.uint8))
        frontiers = np.argwhere(border & self.valid)
        if len(frontiers) > frontier_count:
            frontiers = frontiers[np.linspace(0, len(frontiers)-1, frontier_count, dtype=int)]
        points = np.concatenate([points, frontiers], axis=0)
        points = points[self.valid[points[:, 0], points[:, 1]]]
        if not len(points):
            remaining = np.argwhere(self.valid)
            if len(remaining):
                points = remaining[np.linspace(0, len(remaining)-1, min(256, len(remaining)), dtype=int)]
        return np.unique(points, axis=0)

    def refine_candidates(self, centers, radius=16):
        points = []
        for r, c in centers:
            rr, cc = np.meshgrid(np.arange(max(0, r-radius), min(self.h, r+radius+1)),
                                 np.arange(max(0, c-radius), min(self.w, c+radius+1)), indexing="ij")
            points.append(np.column_stack([rr.ravel(), cc.ravel()]))
        if not points:
            return np.empty((0, 2), np.int64)
        points = np.unique(np.concatenate(points), axis=0)
        return points[self.valid[points[:, 0], points[:, 1]]]


def decode_goal(action, shape):
    """Exact numpy.float32 -> int convention in original eval_poni.py."""
    action = np.asarray(action, dtype=np.float32)
    return tuple(min(int(action[i] * shape[i]), shape[i]-1) for i in range(2))


def encode_goal(cell, shape):
    # Small within-cell offset avoids original float32 division truncating a pixel.
    return np.asarray([(float(cell[i]) + 0.25) / shape[i] for i in range(2)], np.float32)
