"""MP3D HLab policy registration for the independent Sigma-D branch."""

import argparse

from habitat.config import Config

from semexp.model_pf_sigma_d import RL_PolicySigmaD
from policies.policy_registry import policy_registry


@policy_registry.register_policy
class PFExpSigmaD(RL_PolicySigmaD):
    def __init__(self, cfg, *args, **kwargs):
        self.cfg = cfg
        self.prev_maps = None
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg: Config):
        gcfg = cfg.GLOBAL_AGENT
        pcfg = cfg.PF_EXP_POLICY
        if not pcfg.pf_model_path:
            raise ValueError("PF_EXP_POLICY.pf_model_path is required")
        args = argparse.Namespace(
            use_egocentric_transform=pcfg.use_egocentric_transform,
            add_agent2loc_distance=pcfg.add_agent2loc_distance,
            add_agent2loc_distance_v2=pcfg.add_agent2loc_distance_v2,
            pf_masking_opt=pcfg.pf_masking_opt,
            mask_nearest_locations=pcfg.mask_nearest_locations,
            mask_size=pcfg.mask_size,
            visualize=gcfg.visualize,
            print_images=gcfg.visualize,
            map_resolution=gcfg.map_resolution,
            area_weight_coef=pcfg.area_weight_coef,
            dist_weight_coef=pcfg.dist_weight_coef,
            map_size_cm=cfg.PLANNER.map_size_cm,
        )
        return cls(cfg, args, pcfg.pf_model_path)

    def load_checkpoint(self):
        print(
            "=====> PFExpSigmaD: checkpoint already loaded by strict "
            f"Sigma-D adapter from {self.cfg.PF_EXP_POLICY.pf_model_path}"
        )

    def act(
        self,
        inputs,
        rnn_hxs,
        masks,
        extras=None,
        deterministic=False,
        extra_maps=None,
    ):
        outputs = super().act(
            inputs,
            rnn_hxs,
            masks,
            extras=extras,
            deterministic=deterministic,
            extra_maps=extra_maps,
        )
        value, action, action_log_probs, rnn_hxs, all_maps = outputs
        self.prev_maps = all_maps
        return value, action, action_log_probs, rnn_hxs

    @property
    def needs_dist_maps(self):
        return (
            self.cfg.PF_EXP_POLICY.add_agent2loc_distance
            or self.cfg.PF_EXP_POLICY.add_agent2loc_distance_v2
        )

    @property
    def needs_unexplored_maps(self):
        return self.cfg.PF_EXP_POLICY.pf_masking_opt == "unexplored"

    @property
    def needs_frontier_maps(self):
        return False


__all__ = ["PFExpSigmaD"]
