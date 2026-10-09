"""Subclass only goal selection; all Sigma-D and FMM code is reused unchanged."""
from pathlib import Path
import numpy as np
import torch
from semexp.model_pf_sigma_d import RL_PolicySigmaD
from dendritic_nav.core.cell_dend import DendriticScorer, MODEL_FAMILY
from dendritic_nav.core.candidates_dend import build_candidates, encode_action
from dendritic_nav.memory.features_dend import FeatureBuilder, candidate_features
from dendritic_nav.memory.history_dend import CoordinateHistory, rc_to_xy, xy_to_rc


class DendriticGibsonPolicy(RL_PolicySigmaD):
    def __init__(self, args, pf_model_path, bridge, checkpoint=None, baseline=False):
        super().__init__(args, pf_model_path)
        self.network.requires_grad_(False)
        self.dend = DendriticScorer().eval()
        self.bridge = bridge
        bridge.policy = self
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.baseline = baseline
        self.weight_version = -1
        self.rng = np.random.RandomState(args.seed)
        self.state = None
        self.history = CoordinateHistory()
        self.features = FeatureBuilder()

    def _begin_episode(self):
        self.state = None
        self.history = CoordinateHistory()
        self.features = FeatureBuilder()
        if self.checkpoint and self.checkpoint.exists() and not self.baseline:
            checkpoint = torch.load(str(self.checkpoint), map_location="cpu")
            if checkpoint.get("model_family") != MODEL_FAMILY:
                raise ValueError("Unexpected dendritic checkpoint family")
            self.dend.load_state_dict(checkpoint["state_dict"], strict=True)
            self.weight_version = int(checkpoint["environment_steps"])
        self.bridge.new_episode = False

    def act(
        self, inputs, rnn_hxs, masks, extras=None, extra_maps=None, deterministic=False
    ):
        if len(inputs) != 1:
            raise ValueError(
                "Each actor owns one episode stream; scale with independent actors"
            )
        if self.args.global_downscaling != 1 or self.args.num_local_steps != 1:
            raise ValueError(
                "Gibson v1 requires fixed global frame and one policy call per real step"
            )
        if self.bridge.new_episode:
            self._begin_episode()
        self.current_maps = inputs[0].detach().cpu().numpy()
        return super().act(inputs, rnn_hxs, masks, extras, extra_maps, deterministic)

    def get_action(self, pfs, goal_cat_id, umap, dmap, agent_locs):
        anchor = super().get_action(pfs, goal_cat_id, umap, dmap, agent_locs)
        self._rollback = (self.state, self.features.previous, self.features.origin)
        category = int(goal_cat_id[0])
        score = pfs[0, category + 2].detach().cpu().numpy().copy()
        allowed = np.ones_like(score, dtype=bool)
        if self.args.pf_masking_opt == "unexplored":
            unexplored = umap[0].detach().cpu().numpy()
            score *= unexplored
            allowed &= unexplored > 0
        if self.args.mask_nearest_locations:
            r, c = agent_locs[0]
            size = int(self.args.mask_size * 100 / self.args.map_resolution)
            # Deliberately identical slicing to the original get_action.
            score[r - size : r + size + 1, c - size : c + size + 1] = 0
            allowed[r - size : r + size + 1, c - size : c + size + 1] = False
        origin = np.zeros(2)
        resolution = self.args.map_resolution / 100.0
        hist = [xy_to_rc(g, origin, resolution) for g in self.history.goals]
        candidates = build_candidates(
            score, allowed, self.current_maps[1], hist, resolution
        )
        features, frontier, new_target = self.features.build(
            self.current_maps,
            category,
            agent_locs[0],
            origin,
            resolution,
            self.history,
            self.bridge.motion,
        )
        cf = candidate_features(
            candidates,
            self.current_maps,
            agent_locs[0],
            origin,
            resolution,
            self.history,
            frontier,
            category,
        )
        device = pfs.device

        def tensor(x):
            return torch.as_tensor(x, device=device).unsqueeze(0)

        self.state, events = self.dend.cell(tensor(features), self.state)
        q, residual = self.dend.score(
            tensor(cf), tensor(candidates.spatial), tensor(candidates.valid), self.state,
            tensor(features)
        )
        index = int(q[0].argmax())
        if self.baseline:
            index = 0
        if self.bridge.training and not self.baseline:
            published = max(0, self.weight_version)
            epsilon = 0.1 - 0.08 * min(max(published - 10000, 0) / 90000, 1.0)
            if self.rng.rand() < epsilon:
                index = int(self.rng.choice(np.flatnonzero(candidates.valid)))
        action = encode_action(
            candidates.rc[index], score.shape, anchor[0].cpu().numpy(), index
        )
        executed_rc = np.minimum(
            (action * np.array(score.shape)).astype(int), np.array(score.shape) - 1
        )
        self.bridge.pending = dict(
            features=features,
            candidates=cf,
            spatial=candidates.spatial,
            valid=candidates.valid,
            action=index,
            rc=candidates.rc,
            agent_xy=rc_to_xy(agent_locs[0], origin, resolution),
            goal_xy=rc_to_xy(executed_rc, origin, resolution),
            new_target=new_target,
            activity=events[0].abs().mean(-1).detach().cpu().numpy(),
            residual=residual[0].detach().cpu().numpy(),
            selected_q=float(q[0, index]),
            weight_version=self.weight_version,
        )
        return torch.as_tensor(action, device=device).unsqueeze(0)

    def rollback_observation(self):
        self.state, self.features.previous, self.features.origin = self._rollback
