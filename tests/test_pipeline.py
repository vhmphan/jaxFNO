import json
import tempfile
import unittest
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import FNOConfig
from data import Dataset, dataset_fingerprint, encode_source_features, generate_synthetic_dataset, inspect_npz, load_dataset
from evaluate import error_metrics, evaluate_model, linearity_diagnostics
from fno import FourierBlock, make_model, predict
from train import load_predictor, relative_l2_loss, save_checkpoint, split_dataset


class SourceIdentity(eqx.Module):
    def __call__(self, x):
        return x[:1]


class PipelineTests(unittest.TestCase):
    def test_spectral_quadrants_and_small_grids(self):
        block = FourierBlock(1, (8, 8, 8), key=jax.random.PRNGKey(0))
        block = eqx.tree_at(lambda b: (b.kernel_r, b.kernel_i), block,
                            (jnp.ones_like(block.kernel_r), jnp.zeros_like(block.kernel_i)))
        # All retained coefficients set to one must reconstruct arbitrary fields,
        # including negative frequencies, with no overlap for odd or tiny axes.
        for shape in ((1, 5, 7, 5), (1, 4, 4, 4), (1, 1, 2, 3)):
            x = jax.random.normal(jax.random.PRNGKey(2), shape)
            np.testing.assert_allclose(block.spectral(x), x, atol=1e-6)

    def test_shapes_finite_gradients_and_exact_fit(self):
        cfg = FNOConfig(width=2, modes=(3, 3, 3), padding=2)
        model = make_model(cfg)
        x = jnp.ones((4, 3, 4, 5))
        self.assertEqual(model(x).shape, (1, 3, 4, 5))
        loss, grads = eqx.filter_value_and_grad(
            lambda m: relative_l2_loss(jax.vmap(m)(x[None])[:, 0], jnp.ones((1, 3, 4, 5))))(model)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(grads)))
        zeros = jnp.zeros((1, 2, 2, 2))
        gradient = jax.grad(relative_l2_loss)(zeros, zeros)
        self.assertTrue(np.isfinite(gradient).all())

    def test_checkpoint_and_batch_axis(self):
        cfg = FNOConfig(width=2, modes=(2, 2, 2), padding=1)
        ds = generate_synthetic_dataset(10, 4, 4, 4)
        model = make_model(cfg)
        split = dict(zip(("train_idx", "val_idx", "test_idx"), split_dataset(10, seed=3)))
        scales = {"S_scale": 0.2, "u_scale": 0.03}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.npz"
            save_checkpoint(model, path, cfg, scales, ds, split)
            loaded = load_predictor(path)
            expected = predict(model, ds.S[:1], ds.x, ds.y, ds.z, 0.2, 0.03)
            self.assertEqual(expected.shape, (1, 4, 4, 4))
            np.testing.assert_array_equal(loaded.predict(ds.S[:1]), expected)
            self.assertEqual(loaded.predict(ds.S[0]).shape, (4, 4, 4))
            self.assertEqual(loaded.cfg.padding, 1)
            self.assertEqual(loaded.fingerprint, dataset_fingerprint(ds))
            ds.S[[0, 1]] = ds.S[[1, 0]]
            self.assertNotEqual(loaded.fingerprint, dataset_fingerprint(ds))
            np.testing.assert_array_equal(loaded.splits["test_idx"], split["test_idx"])

    def test_loader_mapping_and_validation(self):
        ds = generate_synthetic_dataset(10, 4, 5, 6)
        layout = dict(source_key="surface", target_key="solution", x_key="xx", y_key="yy", z_key="zz",
                      source_axes="ynx", target_axes="znxy", metadata={"units": "arbitrary",
                      "boundary_conditions": "specified in solver", "shared_bvp": True})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.npz"
            np.savez(path, surface=ds.S.transpose(2, 0, 1), solution=ds.u.transpose(3, 0, 1, 2),
                     xx=ds.x, yy=ds.y, zz=ds.z, label=np.array("test"))
            self.assertEqual(inspect_npz(path)["label"]["value"], "test")
            with self.assertRaises(ValueError):
                load_dataset(path)
            actual = load_dataset(path, layout=layout)
            np.testing.assert_array_equal(actual.S, ds.S)
            np.testing.assert_array_equal(actual.u, ds.u)
            with self.assertRaises(FileNotFoundError):
                load_dataset(Path(temp) / "missing.npz")
        with self.assertRaises(ValueError):
            Dataset(ds.S, ds.u, np.array([0, 1, 2, 4]), ds.y, ds.z, {})
        bad = ds.S.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            Dataset(bad, ds.u, ds.x, ds.y, ds.z, {})

    def test_grouped_split(self):
        groups = np.repeat(np.arange(10), 3)
        splits = split_dataset(len(groups), seed=17, groups=groups)
        np.testing.assert_array_equal(np.sort(np.concatenate(splits)), np.arange(len(groups)))
        self.assertEqual(tuple(map(len, splits)), (24, 3, 3))
        for i in range(3):
            np.testing.assert_array_equal(splits[i], split_dataset(len(groups), seed=17, groups=groups)[i])
            for j in range(i):
                self.assertFalse(set(groups[splits[i]]) & set(groups[splits[j]]))

    def test_sol3d_export_adapter(self):
        sx, sy = np.linspace(-2, 2, 5), np.linspace(-1, 1, 3)
        # An asymmetric bilinear field catches x/y swaps and wrong interpolation.
        surface = sx[:, None] + 3 * sy[None, :] + sx[:, None] * sy[None, :]
        original = np.stack([surface, 2 * surface, 3 * surface])
        x, y, z = np.linspace(-2, 2, 9), np.linspace(-1, 1, 5), np.linspace(-4, 4, 5)
        coords = {a: np.concatenate(([c[0] - (c[1]-c[0])], c, [c[-1] + (c[1]-c[0])]))
                  for a, c in zip("xyz", (x, y, z))}
        targets = np.zeros((3, len(z), len(y), len(x)))
        targets[:, 2, 1, 3] = [1, 2, 3]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.npz"
            source_path = Path(temp) / "sources.npz"
            np.savez(source_path, x=sx, y=sy, source=original)
            np.savez(path, **coords, S=original.transpose(0, 2, 1), u=targets, N=3, axis_order="zyx")
            ds = load_dataset(path, layout={"format": "sol3d"})
            expected = x[1:-1, None] + 3 * y[None, 1:-1] + x[1:-1, None] * y[None, 1:-1]
            np.testing.assert_allclose(ds.S[0, 1:-1, 1:-1], expected, atol=1e-7)
            np.testing.assert_array_equal(ds.u, targets.transpose(0, 3, 2, 1))
            np.testing.assert_array_equal(ds.x, x)
            self.assertTrue(np.all(ds.S[:, [0, -1]] == 0))
            self.assertTrue(np.all(ds.S[:, :, [0, -1]] == 0))
            self.assertTrue(ds.metadata["homogeneous_boundary_conditions"])
            np.savez(source_path, x=sx, y=sy, source=original[::-1])
            with self.assertRaisesRegex(ValueError, "sample order"):
                load_dataset(path, layout={"format": "sol3d"})

    def test_training_scales_used_for_evaluation(self):
        c = np.arange(3, dtype=np.float32)
        S = np.full((3, 3, 3), 7.0, dtype=np.float32)
        u = np.broadcast_to((S * 0.1)[..., None], (3, 3, 3, 3)).copy()
        ds = Dataset(S, u, c, c, c, {})
        result = evaluate_model(SourceIdentity(), ds, FNOConfig(), np.arange(3),
                                scales={"S_scale": 2.0, "u_scale": 0.2})
        self.assertLess(result["global_rmse"], 1e-6)
        zero = np.zeros_like(u)
        metrics = error_metrics(np.ones_like(u), zero, np.zeros_like(S))
        self.assertIsNone(metrics["mean_rel_l2"])
        self.assertEqual(metrics["zero_source"]["rmse"], 1.0)
        self.assertEqual(metrics["zero_target"]["count"], 3)

    def test_tiny_subset_overfit(self):
        ds = generate_synthetic_dataset(1, 4, 4, 4, seed=5)
        features = jnp.asarray(encode_source_features(ds.S, ds.x, ds.y, ds.z))
        target = jnp.asarray(ds.u)
        model = make_model(FNOConfig(width=4, modes=(2, 2, 2), padding=1))
        optimizer = optax.adamw(3e-3, weight_decay=0.0)
        state = optimizer.init(eqx.filter(model, eqx.is_array))
        def loss_fn(m):
            return relative_l2_loss(jax.vmap(m)(features)[:, 0], target)
        @eqx.filter_jit
        def step(m, state):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
            updates, state = optimizer.update(grads, state, eqx.filter(m, eqx.is_array))
            return eqx.apply_updates(m, updates), state, loss
        initial = float(loss_fn(model))
        for _ in range(180):
            model, state, loss = step(model, state)
        final = float(loss_fn(model))
        print(f"tiny-subset relative L2: {initial:.6f} -> {final:.6f}", flush=True)
        self.assertLess(final, 0.08)
        self.assertLess(final, initial * 0.1)

    def test_linearity_diagnostics(self):
        sources = np.random.default_rng(0).normal(size=(2, 3, 3))
        metrics = linearity_diagnostics(lambda S: np.repeat(S[..., None], 4, axis=-1), sources)
        self.assertEqual(metrics["zero_source_rmse"], 0.0)
        self.assertLess(metrics["amplitude_scaling_rel_l2"], 1e-14)
        self.assertLess(metrics["superposition_rel_l2"], 1e-14)


if __name__ == "__main__":
    unittest.main()
