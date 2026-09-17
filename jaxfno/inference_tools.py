"""Reusable FNO adapters, observation selection, derivative checks, and plots.

These helpers do not run inference or require NIFTy. The forward adapter reuses
checkpoint weights and preserves the training preprocessing with JAX operations.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from train import load_predictor
from jaxfno._plot_cache import temporary_plot_cache


def make_forward(checkpoint, grid_shape):
    """Reuse the trained network with a differentiable copy of its preprocessing."""
    predictor = load_predictor(checkpoint)
    if predictor.cfg.normalization != "source_mean":
        raise ValueError("Only the inspected source_mean normalization is supported")
    if (predictor.cfg.input_channels, predictor.cfg.output_channels) != (4, 1):
        raise ValueError("Expected four input channels and one output channel")
    if grid_shape is not None:
        if len(grid_shape) != 3 or any(n < 3 for n in grid_shape):
            raise ValueError("GRID_SHAPE needs three dimensions >= 3")
        predictor = predictor.on_grid(*[
            np.linspace(c[0], c[-1], n, dtype=np.float32)
            for c, n in zip(predictor.coordinates, grid_shape)
        ])
    coords = predictor.coordinates
    shape = tuple(len(c) for c in coords)
    # Exactly the coordinate encoding from encode_source_features, computed once.
    grid = jnp.asarray(np.stack(np.meshgrid(*[
        (c - c.mean()) / np.ptp(c) for c in coords
    ], indexing="ij")), dtype=jnp.float32)
    boundary = np.ones(shape[:2], dtype=np.float32)
    if predictor.cfg.dataset_layout.get("format") == "sol3d":
        boundary[[0, -1], :] = 0
        boundary[:, [0, -1]] = 0
    boundary = jnp.asarray(boundary)

    @jax.jit
    def forward(source):
        source = jnp.asarray(source, dtype=jnp.float32) * boundary
        mean = jnp.mean(source)  # Includes zero boundary nodes, just as training.
        encoded = jnp.broadcast_to((source / mean)[..., None], shape)
        features = jnp.concatenate((encoded[None], grid), axis=0)
        return predictor.model(features, padding=predictor.inference_padding)[0] * mean

    return predictor, forward, boundary


def make_observation(shape, mask_path):
    """Separate observation operator; replace this callable with a projection later."""
    mask = np.ones(shape, dtype=bool) if mask_path is None else np.load(mask_path, allow_pickle=False)
    if mask.shape != shape or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Observation mask must be a nonempty boolean array on the XYZ grid")
    indices = jnp.asarray(np.flatnonzero(mask.ravel()))

    def observe(volume):
        return volume.reshape(-1)[indices]

    return observe, mask


def check_forward(predictor, forward, boundary):
    """Check public API agreement, JVP/VJP, and a second derivative for Newton-CG."""
    shape = boundary.shape
    source = jnp.exp(0.15 * jax.random.normal(jax.random.PRNGKey(17), shape))
    result = forward(source)
    expected = predictor.predict(np.asarray(source * boundary))
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=3e-6)
    direction = jax.random.normal(jax.random.PRNGKey(18), shape)
    _, tangent = jax.jvp(forward, (source,), (direction,))
    cotangent = jax.random.normal(jax.random.PRNGKey(19), result.shape)
    _, pullback = jax.vjp(forward, source)
    adjoint = pullback(cotangent)[0]
    np.testing.assert_allclose(jnp.vdot(tangent, cotangent), jnp.vdot(direction, adjoint),
                               rtol=1e-3, atol=1e-4)
    eps = 1e-2
    finite_difference = (forward(source + eps * direction) - forward(source - eps * direction)) / (2 * eps)
    relative_error = float(jnp.linalg.norm(tangent - finite_difference) /
                           jnp.maximum(jnp.linalg.norm(tangent), 1e-10))
    if relative_error > 0.03:
        raise RuntimeError(f"Forward derivative check failed: relative error {relative_error}")
    grad = jax.grad(lambda s: jnp.mean(forward(s) ** 2))
    _, hessian_vector = jax.jvp(grad, (source,), (direction,))
    for value in (result, tangent, adjoint, hessian_vector):
        if not np.isfinite(np.asarray(value)).all():
            raise RuntimeError("Forward model has nonfinite values or derivatives")
    return {"public_predict_agreement": True, "jvp_vjp_agreement": True,
            "finite_difference_relative_error": relative_error, "finite_second_derivative": True}


@temporary_plot_cache()
def plot_results(odir, coords, truth, mean, std, observed, prediction, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y, z = coords
    iz = int(np.argmax(mask.sum(axis=(0, 1))))
    # Prefer the middle plane when it is observed (including the all-voxel case).
    if mask[:, :, len(z) // 2].any():
        iz = len(z) // 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    source_limits = dict(vmin=min(truth.min(), mean.min()), vmax=max(truth.max(), mean.max()))
    obs = observed[:, :, iz]
    pred = prediction[:, :, iz]
    volume_limits = dict(vmin=min(np.nanmin(obs), pred.min()), vmax=max(np.nanmax(obs), pred.max()))
    panels = [(truth, "True positive S", source_limits), (mean, "Posterior mean S", source_limits),
              (std, "Posterior std(S)", {}), (obs, f"Observed u, z={z[iz]:.3g}", volume_limits),
              (pred, "Posterior mean predicted u", volume_limits),
              (obs - pred, "Observed minus predicted u", {"cmap": "coolwarm"})]
    for ax, (values, title, kwargs) in zip(axes.flat, panels):
        im = ax.imshow(values.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], **kwargs)
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax)
    fig.savefig(odir / "reconstruction.png", dpi=160)
    plt.close(fig)
