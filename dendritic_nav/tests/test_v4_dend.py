"""Focused invariants for v4 policy geometry and Bellman critic."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import torch
from dendritic_nav.core.cell_dend import DendriticScorer, MODEL_FAMILY
from dendritic_nav.core.ranking_v4_dend import locality_gain
from dendritic_nav.learning.td_v4_dend import training_loss
from dendritic_nav.learning.replay_dend import validate_episode
from dendritic_nav.runners.collect_dend import official_val_scenes, scene_shard
from dendritic_nav.runners.report_val_dend import collapse_exact_runs


class V4Invariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_chip_friendly_distance_bins(self):
        candidates = torch.zeros(1, 5, 24)
        candidates[0, 1:, 3] = torch.tensor([0.4, 0.75, 1.5, 3.0]) / 20
        self.assertEqual(
            locality_gain(candidates).tolist(), [[1.0, 1.0, 0.5, 0.25, 0.0625]]
        )

    def test_official_val_ignores_auxiliary_episode_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            content = root / "data/datasets/objectnav/gibson/v1.1/val/content"
            assets = root / "data/scene_datasets/gibson_semantic"
            content.mkdir(parents=True)
            assets.mkdir(parents=True)
            (content / "Scene.glb.json.gz").touch()
            (content / "Scene_episodes.json.gz").touch()
            (assets / "Scene.glb").touch()
            self.assertEqual(official_val_scenes(root), ["Scene.glb"])

    def test_official_val_parallel_scene_shards(self):
        scenes = ["a.glb", "b.glb", "c.glb", "d.glb", "e.glb"]
        self.assertEqual(scene_shard(scenes, 7, 20, "val"), ["c.glb"])
        self.assertEqual(scene_shard(scenes, 2, 5, "val"), ["c.glb"])

    def test_official_val_collapses_only_exact_reruns(self):
        a = dict(episode_id="same", scene="scene", success=1, spl=0.5,
                 trajectory_sha256="base")
        b = dict(episode_id="same", scene="scene", success=1, spl=0.4,
                 trajectory_sha256="dend")
        baseline, dendritic, audit = collapse_exact_runs([(a, b), (a, b)])
        self.assertEqual((len(baseline), len(dendritic)), (1, 1))
        self.assertEqual(audit["exact_duplicate_pair_runs"], 1)
        changed = dict(b, trajectory_sha256="different")
        with self.assertRaises(ValueError):
            collapse_exact_runs([(a, b), (a, changed)])

    def test_critic_excludes_fixed_spatial_potential(self):
        self.assertEqual(MODEL_FAMILY, "sigma_d_dendritic_v4")
        model = DendriticScorer()
        features = torch.randn(1, 4, 32) * 3
        features[0, 1, 9] = 3
        state, _ = model.cell(features)
        candidates = torch.randn(1, 8, 24)
        valid = torch.ones(1, 8, dtype=torch.bool)
        first = torch.linspace(0.1, 0.8, 8).unsqueeze(0)
        second = torch.flip(first, (1,))
        first_policy, first_residual, first_critic = model.evaluate(
            candidates, first, valid, state, features
        )
        second_policy, second_residual, second_critic = model.evaluate(
            candidates, second, valid, state, features
        )
        self.assertTrue(torch.equal(first_critic, second_critic))
        self.assertTrue(torch.equal(first_residual, second_residual))
        self.assertTrue(torch.allclose(
            second_policy - first_policy, second - first, atol=1e-6
        ))

    def test_v4_td_loss_updates_value_and_candidate_head(self):
        rng = np.random.RandomState(4)
        n = 6
        features = (rng.randn(n, 4, 32) * 3).astype(np.float32)
        features[:, 1, 9] = 3
        episode = dict(
            features=features,
            candidates=rng.randn(n, 128, 24).astype(np.float32),
            spatial=rng.rand(n, 128).astype(np.float32),
            valid=np.ones((n, 128), bool),
            action=np.array([0, 1, -1, 3, 4, 5]),
            reward=np.array([0.1, 0.2, 0.0, -0.1, 0.1, 1.0], np.float32),
            terminal=np.array([0, 0, 0, 0, 0, 1], bool),
        )
        validate_episode(episode)
        online = DendriticScorer()
        target = DendriticScorer()
        target.load_state_dict(online.state_dict())
        loss, metrics = training_loss(
            online, target, [(episode, 2, 6), (episode, 0, 4)], "cpu"
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["decisions"], 6)
        loss.backward()
        self.assertGreater(float(online.value_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(online.interaction_head.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
