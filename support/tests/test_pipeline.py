import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxfno.config import FNOConfig
from jaxfno.data import Dataset, load_dataset
from jaxfno.evaluation_metrics import error_metrics, evaluate_model, linearity_diagnostics
from evaluate import timed_prediction
from jaxfno.fno import FourierBlock, make_model
from train import Predictor, relative_l2_loss, split_dataset


class SourceIdentity(eqx.Module):
    def __call__(self, x):
        return x[:1]


class PipelineTests(unittest.TestCase):
    def test_prediction_on_new_grid(self):
        cfg = FNOConfig(width=2, modes=(2, 2, 2), padding=2)
        coords = tuple(np.linspace(-limit, limit, size, dtype=np.float32)
                       for limit, size in zip((10, 10, 4), (5, 5, 3)))
        model = make_model(cfg)
        operator = Predictor(model, cfg, {"S_scale": 2.0, "u_scale": 0.1}, coords, {}, {}, "")
        self.assertIs(operator.on_grid(*coords), operator)
        new_coords = tuple(np.linspace(-limit, limit, size, dtype=np.float32)
                           for limit, size in zip((10, 10, 4), (9, 9, 5)))
        refined = operator.on_grid(*new_coords)
        self.assertEqual(refined.inference_padding, (4, 4, 4))
        self.assertIs(refined.model, operator.model)
        self.assertIs(refined.scales, operator.scales)
        self.assertIsNone(operator.inference_padding)
        pred = refined.predict(np.ones((1, 9, 9), dtype=np.float32))
        self.assertEqual(pred.shape, (1, 9, 9, 5))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(operator.on_grid(new_coords[0], coords[1], new_coords[2]).inference_padding, (4, 2, 4))
        with self.assertRaisesRegex(ValueError, "domain"):
            operator.on_grid(new_coords[0] + 1, new_coords[1], new_coords[2])
        with self.assertRaisesRegex(ValueError, "orientation"):
            operator.on_grid(new_coords[0][::-1], new_coords[1], new_coords[2])
        uneven = new_coords[0].copy()
        uneven[1] += 0.1
        with self.assertRaisesRegex(ValueError, "uniformly"):
            operator.on_grid(uneven, new_coords[1], new_coords[2])

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


    def test_timed_predictions_are_returned(self):
        calls = []
        def predict_fn(sources):
            calls.append(len(sources))
            return np.repeat(sources[..., None], 2, axis=-1)
        predictor = SimpleNamespace(cfg=SimpleNamespace(batch_size=2), predict=predict_fn)
        sources = np.ones((3, 4, 4), dtype=np.float32)
        predictions, runtime = timed_prediction(predictor, sources)
        self.assertEqual(calls, [1, 2, 3])  # warm both shapes, then time all samples
        self.assertEqual(predictions.shape, (3, 4, 4, 2))
        self.assertEqual(runtime["realizations"], 3)
        self.assertGreater(runtime["total_seconds"], 0)

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
            only = load_dataset(path, layout={"format": "sol3d"}, sources_only=True)
            self.assertIsNone(only.u)
            np.testing.assert_array_equal(only.S, ds.S)
            # Reconstruct from this file alone, with a deliberately missing companion.
            standalone = load_dataset(path, layout={"format": "sol3d", "source_grid": "uniform_domain",
                                                     "source_path": "does_not_exist.npz"}, sources_only=True)
            np.testing.assert_array_equal(standalone.S, ds.S)
            self.assertIsNone(standalone.metadata["source_file"])
            self.assertIsNone(standalone.u)

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


    def test_linearity_diagnostics(self):
        sources = np.random.default_rng(0).normal(size=(2, 3, 3))
        metrics = linearity_diagnostics(lambda S: np.repeat(S[..., None], 4, axis=-1), sources)
        self.assertEqual(metrics["zero_source_rmse"], 0.0)
        self.assertLess(metrics["amplitude_scaling_rel_l2"], 1e-14)
        self.assertLess(metrics["superposition_rel_l2"], 1e-14)


if __name__ == "__main__":
    unittest.main()
