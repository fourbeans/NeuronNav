import unittest
import numpy as np
import torch
from dendritic_nav.core.cell_dend import DendriticCell, DendriticScorer, spike
from dendritic_nav.core.candidates_dend import build_candidates, encode_action
from dendritic_nav.memory.history_dend import CoordinateHistory, rc_to_xy, xy_to_rc
from dendritic_nav.memory.features_dend import FeatureBuilder
from dendritic_nav.learning.reward_dend import reward
from dendritic_nav.learning.replay_dend import return_target, validate_episode
from dendritic_nav.learning.ddqn_dend import training_loss


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_state_size_and_no_runtime_weights(self):
        cell = DendriticCell()
        state = cell.initial(1, "cpu")
        self.assertEqual(sum(x.numel() for x in state), 2304)
        self.assertTrue(all("readout" not in name for name in cell.state_dict()))

    def test_signed_integer_spike_and_gradient(self):
        x = torch.tensor([-20.0, -0.6, 0.0, 0.7, 20.0], requires_grad=True)
        y = spike(x)
        self.assertEqual(y.tolist(), [-7, -1, 0, 1, 7])
        y.sum().backward()
        self.assertTrue((x.grad == 1).all())

    def test_wait_reset_and_leaks(self):
        cell = DendriticCell()
        x = torch.randn(2, 4, 32)
        state, _ = cell(x)
        waiting, _ = cell(x, state, advance=torch.zeros(2, dtype=torch.bool))
        for a, b in zip(state, waiting):
            self.assertTrue(torch.equal(a, b))
        reset, _ = cell(x, state, reset=torch.tensor([True, False]))
        fresh, _ = cell(x[:1])
        for a, b in zip(reset, fresh):
            self.assertTrue(torch.allclose(a[:1], b, atol=1e-6))
        self.assertEqual(cell.leaks.tolist(), [0.75, 0.9375, 0.984375, 0.99609375])

    def test_sparse_masks(self):
        model = DendriticScorer()
        for projection in list(model.cell.projections) + [model.encoder]:
            self.assertTrue((projection.mask.sum(-1) == 16).all())

    def test_zero_residual_exact_anchor_and_padding(self):
        model = DendriticScorer()
        state, _ = model.cell(torch.randn(1, 4, 32))
        c = torch.randn(1, 128, 24)
        s = torch.zeros(1, 128)
        s[0, 0] = 0.9
        valid = torch.zeros(1, 128, dtype=torch.bool)
        valid[0, :10] = True
        q, r = model.score(c, s, valid, state, torch.zeros(1, 4, 32))
        self.assertTrue(torch.equal(r, torch.zeros_like(r)))
        self.assertEqual(int(q.argmax()), 0)
        self.assertTrue(torch.isneginf(q[0, 10:]).all())

    def test_head_then_cell_gradients(self):
        model = DendriticScorer()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(1, 4, 32) * 3
        features[0, 1, 9] = 3
        c = torch.randn(1, 4, 24) * 3

        def loss():
            state, _ = model.cell(features)
            q, _ = model.score(
                c, torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), state,
                features
            )
            return (q[0, 1] - 1).square()

        loss().backward()
        self.assertGreater(float(model.interaction_head.weight.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad()
        loss().backward()
        self.assertGreater(float(model.cell.projections[0].weight.grad.abs().sum()), 0)

    def test_candidate_independent_value_and_bounded_correction(self):
        model = DendriticScorer()
        state, _ = model.cell(torch.randn(1, 4, 32) * 3)
        candidates = torch.randn(1, 16, 24) * 3
        spatial = torch.rand(1, 16)
        valid = torch.ones(1, 16, dtype=torch.bool)
        features = torch.zeros(1, 4, 32)
        features[0, 1, 9] = 3
        with torch.no_grad():
            model.interaction_head.weight.fill_(100.0)
            model.value_head.weight.fill_(2.0)
            q, correction = model.score(candidates, spatial, valid, state, features)
            self.assertLessEqual(float(correction.abs().max()), model.MAX_CORRECTION)
            shared = q - spatial - correction
            self.assertLess(float((shared - shared[:, :1]).abs().max()), 1e-5)

            inactive = torch.zeros_like(features)
            inactive_q, inactive_correction = model.score(
                candidates, spatial, valid, state, inactive
            )
            self.assertTrue(torch.equal(inactive_correction,
                                        torch.zeros_like(inactive_correction)))
            self.assertEqual(int(inactive_q.argmax()), int(spatial.argmax()))

    def test_evidence_gate_requires_revisit_stall_or_collision(self):
        features = torch.zeros(4, 4, 32)
        features[1, 1, 9] = 3
        features[2, 2, 12] = 0.1
        features[3, 1, 5] = 0.22
        self.assertEqual(DendriticScorer.evidence_gate(features).tolist(),
                         [False, True, True, True])

    def test_switch_penalty_only_when_actual_switch(self):
        base = reward(8.0, 7.0, False, False)
        switched = reward(8.0, 7.0, False, False, goal_switch=True)
        self.assertAlmostEqual(base - switched, 0.01)

    def test_candidate_determinism_and_anchor(self):
        rng = np.random.RandomState(12)
        score = rng.rand(64, 64).astype(np.float32)
        allowed = score > 0.2
        explored = (score > 0.7).astype(float)
        a = build_candidates(score, allowed, explored)
        b = build_candidates(score, allowed, explored)
        self.assertTrue(np.array_equal(a.rc, b.rc))
        self.assertLessEqual(a.valid.sum(), 128)
        self.assertEqual(tuple(a.rc[0]), np.unravel_index(score.argmax(), score.shape))
        self.assertEqual(len(set(map(tuple, a.rc[a.valid]))), int(a.valid.sum()))

    def test_empty_map_has_only_anchor(self):
        c = build_candidates(
            np.zeros((16, 16)), np.zeros((16, 16), bool), np.zeros((16, 16))
        )
        self.assertEqual(c.valid.sum(), 1)
        self.assertEqual(tuple(c.rc[0]), (0, 0))

    def test_anchor_float_semantics_and_new_coordinates(self):
        for shape in ((480, 480), (240, 320)):
            for rc in ((19, 21), (109, 111), (201, 209)):
                old = np.asarray(rc, dtype=np.float32) / np.asarray(
                    shape, dtype=np.float32
                )
                self.assertTrue(np.array_equal(encode_action(rc, shape, old, 0), old))
                self.assertTrue(
                    np.array_equal(
                        (encode_action(rc, shape, index=1) * shape).astype(int), rc
                    )
                )

    def test_coordinate_window_and_stall(self):
        h = CoordinateHistory()
        origin = np.array([4.0, 8.0])
        rc = np.array([20, 30])
        xy = rc_to_xy(rc, origin, 0.05)
        self.assertTrue(np.array_equal(xy_to_rc(xy, origin, 0.05), rc))
        h.choose([8.0, 8.0])
        for _ in range(30):
            h.advance(xy)
        self.assertEqual(h.lookup([8.0, 8.0])[4], 1)
        self.assertEqual(
            h.lookup(rc_to_xy(xy_to_rc(xy, [1.0, 2.0], 0.05), [1.0, 2.0], 0.05))[0], 30
        )

    def test_entry_count_ignores_contiguous_dwell(self):
        h = CoordinateHistory()
        a = np.array([1.0, 1.0])
        b = np.array([1.5, 1.0])
        for _ in range(8):
            h.advance(a)
        self.assertEqual(h.entry_count(a), 1)
        h.advance(b)
        h.advance(a)
        self.assertEqual(h.entry_count(a), 2)

    def test_frame_change_not_false_semantic_evidence(self):
        builder = FeatureBuilder()
        maps = np.zeros((24, 16, 16), dtype=np.float32)
        h = CoordinateHistory()
        builder.build(maps, 0, [8, 8], [0, 0], 0.05, h, {})
        maps[8] = 1
        f, _, _ = builder.build(maps, 0, [8, 8], [1, 0], 0.05, h, {})
        self.assertEqual(float(f[0, 0]), 0)
        self.assertEqual(float(f[1, 8]), 1)

    def test_terminal_reward_telescopes(self):
        gamma = 0.995
        distances = [10.0, 9.0, 7.0, 6.0]
        total = sum(
            gamma ** i
            * reward(distances[i], distances[i + 1], False, i == 2, gamma=gamma)
            for i in range(3)
        )
        self.assertAlmostEqual(
            total, 0.5 - 0.001 * sum(gamma ** i for i in range(3)), places=6
        )

    def test_overrides_discount_actual_steps(self):
        e = dict(
            action=np.array([0, -1, -1, 1, 0, 1]),
            reward=np.ones(6),
            terminal=np.array([0, 0, 0, 0, 0, 1], bool),
        )
        r, d, t = return_target(e, 0, gamma=0.5)
        self.assertEqual(t, 5)
        self.assertEqual(d, 0.5 ** 5)
        self.assertEqual(r, sum(0.5 ** i for i in range(5)))
        r, d, t = return_target(e, 4, gamma=0.5)
        self.assertIsNone(t)
        self.assertEqual(d, 0)
        self.assertEqual(r, 1.5)

    def test_recurrent_ddqn_loss_backward(self):
        rng = np.random.RandomState(4)
        n = 6
        e = dict(
            features=rng.randn(n, 4, 32).astype(np.float32),
            candidates=rng.randn(n, 128, 24).astype(np.float32),
            spatial=rng.rand(n, 128).astype(np.float32),
            valid=np.ones((n, 128), bool),
            action=np.array([0, 1, -1, 3, 4, 5]),
            reward=np.zeros(n, np.float32),
            terminal=np.array([0, 0, 0, 0, 0, 1], bool),
        )
        validate_episode(e)
        online = DendriticScorer()
        target = DendriticScorer()
        target.load_state_dict(online.state_dict())
        loss, metrics = training_loss(online, target, [(e, 2, 6), (e, 0, 4)], "cpu")
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(metrics["decisions"], 6)

    def test_temporal_gradients_survive_subtractive_reset(self):
        cell = DendriticCell()
        x = (torch.randn(1, 4, 32) * 2).requires_grad_()
        state, _ = cell(x)
        state, _ = cell(torch.zeros_like(x), state)
        state.readout.sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_fixed_point_leak_shift_error_bound(self):
        # Q16 reference: v - (v >> k), one arithmetic shift per compartment.
        v = np.arange(-150000, 150000, 137, dtype=np.int64)
        for k in (2, 4, 6, 8):
            fixed = (v - (v >> k)) / 65536.0
            floating = (v / 65536.0) * (1 - 2.0 ** -k)
            self.assertLessEqual(float(np.max(np.abs(fixed - floating))), 1 / 65536.0)

    def test_privileged_metadata_not_used_as_input(self):
        maps = np.zeros((24, 16, 16), dtype=np.float32)
        clean, _, _ = FeatureBuilder().build(
            maps, 0, [8, 8], [0, 0], 0.05, CoordinateHistory(), {}
        )
        extra, _, _ = FeatureBuilder().build(
            maps,
            0,
            [8, 8],
            [0, 0],
            0.05,
            CoordinateHistory(),
            {"gt_distance": 9999.0, "success": 1.0},
        )
        self.assertTrue(np.array_equal(clean, extra))


if __name__ == "__main__":
    unittest.main()
