from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxfno.data import encode_source_features, source_means


class PointwiseLayer(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_features: int, out_features: int, *, key):
        self.weight = jax.random.normal(key, (out_features, in_features)) / in_features**0.5
        self.bias = jnp.zeros(out_features)

    def __call__(self, x):
        return jnp.einsum("oc,cxyz->oxyz", self.weight, x) + self.bias[:, None, None, None]


class FourierBlock(eqx.Module):
    # Four independent quadrants: (+,+), (+,-), (-,+), (-,-).
    kernel_r: jax.Array
    kernel_i: jax.Array
    pointwise: PointwiseLayer

    def __init__(self, width: int, modes: tuple[int, int, int], *, key):
        kr, ki, kp = jax.random.split(key, 3)
        shape = (4, width, width, *modes)
        self.kernel_r = jax.random.normal(kr, shape) / width
        self.kernel_i = jax.random.normal(ki, shape) / width
        self.pointwise = PointwiseLayer(width, width, key=kp)

    def spectral(self, x):
        """(C,Nx,Ny,Nz) -> same shape; FFTs never transform channels."""
        spectrum = jnp.fft.rfftn(x, axes=(-3, -2, -1))
        nx, ny, nz = x.shape[-3:]
        mx, my, mz = self.kernel_r.shape[-3:]
        # Asymmetric counts include DC on odd/singleton grids without overlap.
        xp, xn = min(mx, (nx + 1) // 2), min(mx, nx // 2)
        yp, yn = min(my, (ny + 1) // 2), min(my, ny // 2)
        kz = min(mz, nz // 2 + 1)
        result = jnp.zeros_like(spectrum)
        for q, (xs, ys, cx, cy) in enumerate([
            (slice(0, xp), slice(0, yp), xp, yp),
            (slice(0, xp), slice(ny - yn, ny), xp, yn),
            (slice(nx - xn, nx), slice(0, yp), xn, yp),
            (slice(nx - xn, nx), slice(ny - yn, ny), xn, yn),
        ]):
            if cx and cy:
                weights = (self.kernel_r[q, :, :, :cx, :cy, :kz]
                           + 1j * self.kernel_i[q, :, :, :cx, :cy, :kz])
                values = jnp.einsum("cxyz,ocxyz->oxyz", spectrum[:, xs, ys, :kz], weights)
                result = result.at[:, xs, ys, :kz].set(values)
        return jnp.fft.irfftn(result, s=(nx, ny, nz), axes=(-3, -2, -1))

    def __call__(self, x):
        return jax.nn.gelu(self.spectral(x) + self.pointwise(x))


class FourierNet(eqx.Module):
    lift: PointwiseLayer
    blocks: list[FourierBlock]
    project: PointwiseLayer
    padding: int = eqx.field(static=True)

    def __init__(self, in_channels, out_channels, width, modes, *, key, padding=0):
        keys = jax.random.split(key, 6)
        self.lift = PointwiseLayer(in_channels, width, key=keys[0])
        self.blocks = [FourierBlock(width, modes, key=k) for k in keys[1:5]]
        self.project = PointwiseLayer(width, out_channels, key=keys[5])
        self.padding = padding

    def __call__(self, x, *, padding=None):
        """Single sample: (4,Nx,Ny,Nz) -> (1,Nx,Ny,Nz)."""
        nx, ny, nz = x.shape[-3:]
        x = self.lift(x)
        p = (self.padding,) * 3 if padding is None else padding
        if any(p):
            x = jnp.pad(x, ((0, 0), *((0, count) for count in p)))
        for block in self.blocks:
            x = block(x)
        return self.project(x[:, :nx, :ny, :nz])


def make_model(cfg, *, key=None):
    return FourierNet(cfg.input_channels, cfg.output_channels, cfg.width, cfg.modes,
                      key=jax.random.PRNGKey(cfg.seed) if key is None else key,
                      padding=cfg.padding)


@eqx.filter_jit
def infer_batch(model, features, padding=None):
    if padding is None:
        return jax.vmap(model)(features)[:, 0]
    return jax.vmap(lambda x: model(x, padding=padding))(features)[:, 0]


def predict(model, S, x, y, z, batch_size=1, *, padding=None):
    """Physical units; preserve a batch axis even for a batch of one sample."""
    S = np.asarray(S, dtype=np.float32)
    single = S.ndim == 2
    if single:
        S = S[None]
    if S.ndim != 3 or len(S) == 0 or batch_size < 1:
        raise ValueError("Expected a nonempty source batch and positive batch_size")
    outputs = []
    for start in range(0, len(S), batch_size):
        sources = S[start:start + batch_size]
        means = source_means(sources)
        sources = sources / means
        features = encode_source_features(sources, x, y, z)
        prediction = np.asarray(infer_batch(model, jnp.asarray(features), padding))
        prediction = prediction * means[..., None]
        outputs.append(prediction)
    result = np.concatenate(outputs)
    return result[0] if single else result
