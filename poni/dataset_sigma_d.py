"""Temporal semantic-map data for incremental Sigma-Delta PONI training.

The original PONI dataset samples one partially observed map from one shortest
path.  This module keeps that scene/floor and shortest path fixed, then emits
strictly increasing prefixes of the path as a short *real* observation
sequence.  A frame is therefore new accumulated map evidence; it is never a
copy of one static map repeated for an artificial simulation time dimension.

No temporal frames are written to disk.  Cropping, rotation, potential-field
targets, frontier masks, and unexplored-area targets are produced on demand.
"""

from __future__ import annotations

import contextlib
import math
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from poni.constants import FLOOR_ID
from poni.dataset import SemanticMapDataset, SemanticMapPrecomputedDataset
from poni.fmm_planner import FMMPlanner


class SigmaDTemporalSemanticMapDataset(SemanticMapDataset):
    """Generate fixed-view, cumulative semantic-map sequences on the fly.

    Args:
        cfg: The original PONI ``DATASET`` config node.  In particular,
            ``root`` must point to semantic-map H5 files, not the precomputed
            single-frame PBZ2 directory.
        split: Original PONI scene split.
        sequence_length: Number of real cumulative observations.  The first
            observation is a reset frame and later observations add longer
            prefixes of the same shortest path.  The Sigma-Delta baseline uses
            four.
        samples_per_epoch: Virtual epoch size.  Sampling remains on-the-fly and
            does not materialize this many sequences.  Defaults to the number
            of loaded scene/floor maps.
        seed: Base sampling seed.  Defaults to ``cfg.seed``.
        max_sampling_attempts: Number of path/view resamples allowed when a
            crop would make adjacent cumulative observations identical.
        require_strict_growth: If true, reject a sequence unless every frame
            contains new visible cells relative to its predecessor.

    Returns from ``__getitem__``:
        A dictionary containing:

        - ``frames``: ``[K,C,H,W]`` float32 cumulative partial semantic maps.
        - ``object_pfs``: ``[K,C,H,W]`` float32 masked object-PF targets.
        - ``area_pfs``: ``[K,1,H,W]`` float32 targets.  If area supervision is
          disabled in the config this is an all-zero tensor and
          ``metadata['has_area_pfs']`` is false.
        - ``loss_masks``: ``[K,C,H,W]`` float32 PONI loss masks.
        - ``valid_steps``: ``[K]`` bool; all true because this dataset only
          returns complete sequences.
        - ``reset_mask``: ``[K]`` bool; true only at the first step.
        - ``metadata``: deterministic sample/view provenance.

    ``set_epoch`` changes the deterministic sample stream without depending on
    DataLoader worker order.  The same ``(seed, epoch, index)`` always produces
    the same result.
    """

    _EPOCH_MIX = 0x9E3779B1
    _INDEX_MIX = 0x85EBCA77
    _ATTEMPT_MIX = 0xC2B2AE3D

    def __init__(
        self,
        cfg,
        split: str = "train",
        sequence_length: int = 4,
        samples_per_epoch: Optional[int] = None,
        seed: Optional[int] = None,
        max_sampling_attempts: int = 32,
        require_strict_growth: bool = True,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if max_sampling_attempts < 1:
            raise ValueError("max_sampling_attempts must be positive")

        # SemanticMapDataset seeds the process-global Python and NumPy RNGs.
        # Preserve the caller's RNG streams: item sampling below is deliberately
        # derived only from (base_seed, epoch, index).
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            super().__init__(cfg, split=split, seed=seed)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

        if not self.names:
            raise RuntimeError(
                f"No semantic scene/floor maps were loaded for split={split!r} "
                f"from root={cfg.root!r}"
            )

        if samples_per_epoch is None:
            samples_per_epoch = len(self.names)
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")

        self.split = split
        self.sequence_length = int(sequence_length)
        self.samples_per_epoch = int(samples_per_epoch)
        self.base_seed = int(cfg.seed if seed is None else seed)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.require_strict_growth = bool(require_strict_growth)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic virtual epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def _seed_for(self, index: int, attempt: int = 0) -> int:
        """Return a stable uint32 seed independent of Python hash randomization."""
        value = (
            self.base_seed
            + self.epoch * self._EPOCH_MIX
            + index * self._INDEX_MIX
            + attempt * self._ATTEMPT_MIX
        )
        return int(value & 0xFFFFFFFF)

    @staticmethod
    @contextlib.contextmanager
    def _scoped_legacy_rng(seed: int) -> Iterator[None]:
        """Seed legacy RNG users without leaking state into the training loop.

        ``SemanticMapDataset.get_random_shortest_path`` uses the global NumPy
        RNG.  Reusing it inside this scope retains the original path-generation
        behavior while keeping multi-worker samples reproducible.
        """
        py_state = random.getstate()
        np_state = np.random.get_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

    def _choose_scene_floor(self, index: int) -> str:
        # Shuffle scene/floor order per epoch, then cycle if samples_per_epoch is
        # larger than the number of maps.  This guarantees balanced coverage
        # while remaining invariant to DataLoader worker scheduling.
        scene_rng = np.random.RandomState(self._seed_for(0))
        order = scene_rng.permutation(len(self.names))
        cycle = index // len(self.names)
        offset = index % len(self.names)
        if cycle:
            cycle_rng = np.random.RandomState(self._seed_for(cycle))
            order = cycle_rng.permutation(len(self.names))
        return self.names[int(order[offset])]

    def _sample_path(
        self, name: str, seed: int
    ) -> List[Tuple[int, int]]:
        semmap = self.maps[name]
        nav_space = semmap[FLOOR_ID]
        nav_locs = self.nav_locs[name]
        if str(self.cfg.dset_name).lower() != "mp3d":
            # Gibson did not exhibit the tiny-floor/cyclic-STG pathology.  Keep
            # its original conditional path distribution exactly stable so a
            # full run resumed after this MP3D fix does not change data policy
            # midway through training.
            with self._scoped_legacy_rng(seed):
                path = self.get_random_shortest_path(nav_space, nav_locs)
            return [(int(y), int(x)) for y, x in path]

        with self._scoped_legacy_rng(seed):
            # The canonical sampler chooses any reachable endpoint.  For a
            # temporal sample that wastes repeated full-map FMM solves on
            # endpoints whose paths have fewer than K planner steps.  Compute
            # one distance field from a random start and condition the endpoint
            # on being in the farther half of reachable cells.  This is the
            # efficient form of the same "sample then reject short paths"
            # distribution required by a complete K-frame sequence.
            planner = FMMPlanner(nav_space)
            ys, xs = nav_locs
            best_path: List[Tuple[int, int]] = []
            for _ in range(16):
                start_ix = int(np.random.randint(0, xs.shape[0]))
                start = (int(ys[start_ix]), int(xs[start_ix]))
                planner.set_goal(start)
                distances = np.asarray(planner.fmm_dist)
                reachable = np.isfinite(distances) & (distances < distances.max())
                reachable_distances = distances[reachable]
                if reachable_distances.size < 20:
                    continue

                # Sampling from the farther half avoids degenerate adjacent
                # endpoints without always choosing the scene diameter.
                cutoff = float(np.percentile(reachable_distances, 50.0))
                far_y, far_x = np.where(reachable & (distances >= cutoff))
                if far_x.size == 0:
                    continue
                end_ix = int(np.random.randint(0, far_x.shape[0]))
                end = (int(far_y[end_ix]), int(far_x[end_ix]))
                candidate, reached_goal = self._plan_path_safely(
                    nav_space, start, end
                )
                if not reached_goal:
                    continue
                if len(candidate) > len(best_path):
                    best_path = candidate
                if len(candidate) >= self.sequence_length:
                    return candidate
        return best_path

    @staticmethod
    def _plan_path_safely(
        nav_space: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        max_steps: int = 4096,
    ) -> Tuple[List[Tuple[int, int]], bool]:
        """Plan like the original helper, rejecting a cycle without printing.

        ``SemanticMapDataset.plan_path`` only stops a pathological numerical
        STG descent after 10,000 iterations and returns that partial trace as
        if it were a path.  Temporal supervision must never consume it.  A
        shortest path cannot revisit a cell, so cycle detection is exact and
        avoids the long warning-producing loop.
        """

        planner = FMMPlanner(nav_space)
        planner.set_goal(end)
        current = (int(start[0]), int(start[1]))
        path = [current]
        visited = {current}
        for _ in range(max_steps):
            next_y, next_x, _, stop = planner.get_short_term_goal(current)
            if stop:
                return path, True
            next_location = (int(next_y), int(next_x))
            if next_location in visited:
                return path, False
            visited.add(next_location)
            path.append(next_location)
            current = next_location
        return path, False

    def _prefix_ends(self, path_length: int) -> np.ndarray:
        if path_length < self.sequence_length:
            raise ValueError(
                f"path has {path_length} locations but "
                f"sequence_length={self.sequence_length}"
            )
        ends = np.linspace(
            0, path_length - 1, num=self.sequence_length, dtype=np.int64
        )
        if np.any(ends[1:] <= ends[:-1]):
            raise AssertionError("prefix endpoints are not strictly increasing")
        return ends

    def _transform_prefix(
        self,
        out_semmap: torch.Tensor,
        out_fmm_dists: torch.Tensor,
        path: Sequence[Tuple[int, int]],
        prefix_end: int,
        translation: torch.Tensor,
        rotation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get_visibility_map returns a cumulative union over all prefix poses.
        visibility = self.get_visibility_map(out_semmap, path[: prefix_end + 1])
        in_semmap = out_semmap * visibility.unsqueeze(0)
        transformed = self.transform_input_output_pairs(
            in_semmap,
            out_semmap,
            out_fmm_dists,
            translation,
            rotation,
        )
        # The original helper also returns agent distance and its own base mask;
        # targets below are deliberately built by the precomputed-dataset helper
        # so the training labels match the canonical PONI path.
        t_input, t_semmap, t_fmm_dists = transformed[:3]
        # ``scripts/create_poni_dataset.py`` thresholds both semantic tensors
        # at 0.5 before the canonical label builder sees them.  Spatial
        # rotation returns floating tensors even though these are categorical
        # occupancy maps; retaining those floats breaks boolean area-map
        # operations and would also diverge from the original training data.
        t_input = t_input > 0.5
        t_semmap = t_semmap > 0.5
        return t_input.float(), t_semmap, t_fmm_dists

    def _make_targets(
        self,
        in_semmap: torch.Tensor,
        out_semmap: torch.Tensor,
        out_fmm_dists: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            loss_masks,
            potential_masks,
            _directions,
            _locations,
            area_pfs,
            _actions,
            _frontiers,
        ) = SemanticMapPrecomputedDataset.get_masks_and_labels(
            self, in_semmap, out_semmap, out_fmm_dists
        )

        object_pfs = self.compute_object_pfs(out_fmm_dists)
        if self.cfg.potential_function_masking:
            object_pfs = torch.clamp(object_pfs * potential_masks, 0.0, 1.0)

        if area_pfs is None:
            height, width = in_semmap.shape[-2:]
            area_pfs = torch.zeros((1, height, width), dtype=torch.float32)

        return object_pfs.float(), area_pfs.float(), loss_masks.float()

    @staticmethod
    def _strict_growth_statistics(frames: Sequence[torch.Tensor]) -> Tuple[bool, List[int]]:
        changed_cells: List[int] = [int(torch.count_nonzero(frames[0]).item())]
        strictly_growing = True
        for previous, current in zip(frames[:-1], frames[1:]):
            # Semantic observations are cumulative.  Check both that nothing
            # disappears and that at least one cell/channel is newly visible.
            if torch.any(current < previous):
                strictly_growing = False
            changed = int(torch.count_nonzero(current != previous).item())
            changed_cells.append(changed)
            if changed == 0:
                strictly_growing = False
        return strictly_growing, changed_cells

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        return self._getitem_with_scene_fallback(
            index=index,
            requested_index=index,
            scene_hop=0,
            failures=[],
        )

    def _getitem_with_scene_fallback(
        self,
        index: int,
        requested_index: int,
        scene_hop: int,
        failures: List[str],
    ) -> Dict[str, object]:
        """Sample one sequence, moving deterministically to another map if needed.

        Some MP3D floor maps are real but too small to contain ``K`` distinct
        locations (for example, every sampled shortest path can have length
        two).  Retrying that same floor forever cannot produce a valid temporal
        sequence.  After exhausting the ordinary path/view retries, advance in
        the epoch's deterministic scene order.  We never pad or duplicate
        frames: every returned item still satisfies the strict K-frame growth
        contract.
        """

        name = self._choose_scene_floor(index)
        semmap_np = self.maps[name]
        fmm_dists_np = self.fmm_dists[name]
        out_semmap = torch.from_numpy(semmap_np)
        # Original dynamic dataset converts grid cells to metres before forming
        # potential fields and labels.
        out_fmm_dists = torch.from_numpy(fmm_dists_np).float() * self.grid_size

        last_failure = "no sampling attempt was made"
        for attempt in range(self.max_sampling_attempts):
            sample_seed = self._seed_for(index, attempt)
            rng = np.random.RandomState(sample_seed)
            path = self._sample_path(name, sample_seed)
            if len(path) < self.sequence_length:
                last_failure = f"shortest path length {len(path)} < {self.sequence_length}"
                # A handful of independent endpoints is enough to identify the
                # tiny MP3D floor maps that cannot support a K-frame sequence.
                # Spending all view-resampling attempts here is both futile and
                # very expensive because shortest paths are FMM-generated.
                if attempt >= min(3, self.max_sampling_attempts - 1):
                    break
                continue

            prefix_ends = self._prefix_ends(len(path))

            # One center and one rotation are shared by every temporal frame.
            # Sampling the center from the path preserves the original PONI
            # view distribution while the strict-growth check rejects views in
            # which crop boundaries erase all incremental evidence.
            center_index = int(rng.randint(0, len(path)))
            center_y, center_x = path[center_index]
            rotation_radians = float(rng.uniform(-math.pi, math.pi))
            width_half = out_semmap.shape[2] // 2
            height_half = out_semmap.shape[1] // 2
            translation = torch.tensor(
                [[center_x - width_half, center_y - height_half, 0.0]],
                dtype=torch.float32,
            )
            rotation = torch.tensor(
                [[0.0, 0.0, rotation_radians]], dtype=torch.float32
            )

            frames: List[torch.Tensor] = []
            object_pfs: List[torch.Tensor] = []
            area_pfs: List[torch.Tensor] = []
            loss_masks: List[torch.Tensor] = []
            reference_semmap: Optional[torch.Tensor] = None
            reference_fmm: Optional[torch.Tensor] = None

            for prefix_end in prefix_ends.tolist():
                frame, transformed_semmap, transformed_fmm = self._transform_prefix(
                    out_semmap,
                    out_fmm_dists,
                    path,
                    int(prefix_end),
                    translation,
                    rotation,
                )
                if reference_semmap is None:
                    reference_semmap = transformed_semmap
                    reference_fmm = transformed_fmm
                else:
                    # The view is fixed, hence ground truth must be identical at
                    # every temporal step.  Fail loudly if an upstream transform
                    # ever violates this contract.
                    if not torch.equal(reference_semmap, transformed_semmap):
                        raise RuntimeError("fixed-view semantic target changed across prefixes")
                    if not torch.equal(reference_fmm, transformed_fmm):
                        raise RuntimeError("fixed-view FMM target changed across prefixes")

                obj_pf, area_pf, loss_mask = self._make_targets(
                    frame, transformed_semmap, transformed_fmm
                )
                frames.append(frame)
                object_pfs.append(obj_pf)
                area_pfs.append(area_pf)
                loss_masks.append(loss_mask)

            strict_growth, changed_cells = self._strict_growth_statistics(frames)
            if self.require_strict_growth and not strict_growth:
                last_failure = (
                    "fixed crop produced a non-incremental or repeated frame; "
                    f"changed counts={changed_cells}"
                )
                continue

            reset_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
            reset_mask[0] = True
            valid_steps = torch.ones(self.sequence_length, dtype=torch.bool)
            scene_name, separator, floor_id = name.rpartition("_")
            if not separator:
                scene_name, floor_id = name, ""

            return {
                "frames": torch.stack(frames, dim=0),
                "object_pfs": torch.stack(object_pfs, dim=0),
                "area_pfs": torch.stack(area_pfs, dim=0),
                "loss_masks": torch.stack(loss_masks, dim=0),
                "valid_steps": valid_steps,
                "reset_mask": reset_mask,
                "metadata": {
                    "scene_floor": name,
                    "scene_name": scene_name,
                    "floor_id": floor_id,
                    "split": self.split,
                    "epoch": torch.tensor(self.epoch, dtype=torch.int64),
                    "sample_index": torch.tensor(
                        requested_index, dtype=torch.int64
                    ),
                    "source_sample_index": torch.tensor(index, dtype=torch.int64),
                    "scene_fallback_hops": torch.tensor(
                        scene_hop, dtype=torch.int64
                    ),
                    "sample_seed": torch.tensor(sample_seed, dtype=torch.int64),
                    "path_length": torch.tensor(len(path), dtype=torch.int64),
                    "prefix_ends": torch.as_tensor(prefix_ends, dtype=torch.int64),
                    "prefix_lengths": torch.as_tensor(
                        prefix_ends + 1, dtype=torch.int64
                    ),
                    "crop_center_yx": torch.tensor(
                        [center_y, center_x], dtype=torch.int64
                    ),
                    "rotation_radians": torch.tensor(
                        rotation_radians, dtype=torch.float32
                    ),
                    "changed_elements": torch.tensor(
                        changed_cells, dtype=torch.int64
                    ),
                    "has_area_pfs": bool(self.cfg.enable_unexp_area),
                    "strict_growth": strict_growth,
                },
            }

        failures.append(f"{name}: {last_failure}")
        if scene_hop + 1 < len(self.names):
            # Move by one virtual index, retaining the original requested index
            # in metadata.  Since _choose_scene_floor is a permutation within
            # each cycle, at most len(names) hops inspect every scene/floor.
            next_index = (index + 1) % len(self)
            return self._getitem_with_scene_fallback(
                index=next_index,
                requested_index=requested_index,
                scene_hop=scene_hop + 1,
                failures=failures,
            )

        failure_summary = "; ".join(failures[:8])
        if len(failures) > 8:
            failure_summary += f"; ... ({len(failures)} scene/floors tried)"
        raise RuntimeError(
            f"Could not sample a valid K={self.sequence_length} temporal sequence "
            f"at epoch={self.epoch}, requested_index={requested_index}; "
            f"each scene/floor received {self.max_sampling_attempts} attempts. "
            f"Failures: {failure_summary}"
        )


__all__ = ["SigmaDTemporalSemanticMapDataset"]
