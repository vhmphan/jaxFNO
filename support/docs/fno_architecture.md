# FNO architecture

This document describes the current implementation in [jaxfno/fno.py](../jaxfno/fno.py), with preprocessing in [jaxfno/data.py](../jaxfno/data.py) and training in [train.py](../train.py). Configuration defaults are defined in [jaxfno/config.py](../jaxfno/config.py). A particular trained model's configuration is stored in its checkpoint; it may override these defaults.

## Learned mapping and normalization

The network learns a mapping from a two-dimensional source `S(x, y)` to a three-dimensional solution `u(x, y, z)` for the shared physical problem represented by the dataset.

The loader arranges sources as `(N, Nx, Ny)` and targets as `(N, Nx, Ny, Nz)`. For the sol3d export, it trims coordinate ghost cells, transposes the target from `nzyx` to `nxyz`, and interpolates the source onto the interior solver XY grid, with a zero source perimeter.

Each realization uses its own source mean:

```python
S_mean = np.mean(S, axis=(1, 2), keepdims=True)
S_normalized = S / S_mean
u_normalized = u / S_mean[..., None]
```

The mean includes all loaded XY nodes, including the zero perimeter. It must be finite and nonzero. There are no global source or target scaling constants. During prediction, the network output is multiplied by the corresponding test source mean to restore physical units.

## Input features

Each sample has four input channels, shaped `(4, Nx, Ny, Nz)`:

1. The normalized source, broadcast along z.
2. The normalized x coordinate.
3. The normalized y coordinate.
4. The normalized z coordinate.

Each coordinate vector `c` is encoded as `(c - c.mean()) / np.ptp(c)`, giving approximately `[-0.5, 0.5]` on uniform grids. Features are float32. Broadcasting S along z is an input encoding; it does not change the physical source into a volume source or apply the solver's `-S/dz` factor.

## Network layers

The default hidden width is 16. Pointwise layers apply a learned affine transformation to the channels independently at every spatial location, equivalent to a 1 × 1 × 1 convolution.

| Stage | Operation | Output shape for one sample |
| --- | --- | --- |
| Features | Normalized source and coordinates | `(4, Nx, Ny, Nz)` |
| Lift | Pointwise affine map, 4 → 16 | `(16, Nx, Ny, Nz)` |
| Padding | Append 4 zero cells on the high-index end of each spatial axis | `(16, Nx+4, Ny+4, Nz+4)` |
| Fourier blocks | Four successive spectral-plus-pointwise blocks, each followed by GELU | Same padded shape |
| Crop | Remove the appended cells | `(16, Nx, Ny, Nz)` |
| Projection | Pointwise affine map, 16 → 1 | `(1, Nx, Ny, Nz)` |
| Physical rescaling | Multiply by the source mean | Solution in dataset units |

Padding is applied after lifting. There is no extra activation after the lift or final projection. The model has no dropout, batch/layer normalization, or explicit identity residual connection. Batch inference uses `jax.vmap`, removes the singleton output-channel axis, and returns `(N, Nx, Ny, Nz)`.

## Fourier block

For hidden features `v`, each block computes

```text
v_next = GELU(spectral(v) + W v + b)
```

The spectral branch performs:

1. A three-dimensional real FFT (`rfftn`) over spatial axes only.
2. Learned complex channel mixing at retained Fourier modes.
3. An inverse real FFT (`irfftn`) back to the same padded spatial shape.

The default mode counts are `(8, 8, 8)`. Four independent weight tensors cover the combinations of low positive/nonnegative and negative x/y frequencies: `(+,+)`, `(+,-)`, `(-,+)`, and `(-,-)`. The z spectrum is one-sided because the input is real. Thus, on sufficiently large grids, each quadrant retains 8 × 8 × 8 coefficients; the mode setting is not the full number of retained signed x/y frequencies.

Real and imaginary weights are stored separately. Each has shape `(4, width, width, mx, my, mz)`. Unretained spectral coefficients are zero in the spectral branch, while the parallel pointwise branch still acts on the full spatial field. Mode counts are clipped to available frequencies on small grids to avoid overlap between quadrants.

Pointwise weights are initialized from a normal distribution divided by the square root of the input-channel count, with zero biases. Spectral real and imaginary weights are initialized from normal distributions divided by the hidden width. The default random seed is 0.

For width 16 and modes `(8, 8, 8)`, there are **4,195,489 trainable real scalar parameters**:

- Lift: 80.
- Each Fourier block: 1,048,576 spectral parameters plus 272 pointwise parameters.
- Four blocks: 4,195,392.
- Projection: 17.

## Training procedure

The architecture is implemented with JAX and Equinox. Training uses Optax AdamW and the mean per-sample relative L2 loss on normalized targets:

```text
mean_i(||prediction_i - target_i||_2 / max(||target_i||_2, loss_floor))
```

Default settings are learning rate `0.001`, weight decay `0.0001`, batch size 1, loss floor `1e-8`, and an 80/10/10 training/validation/test split with seed 0. Related samples can be split by group. The default epoch limit is 200 with patience 20; JSON configuration and CLI overrides determine the actual run limit. The lowest-validation-loss weights are saved. Resume restores weights and splits but restarts AdamW state.

## Prediction on a different resolution

Fourier weights and pointwise maps do not depend on the number of grid nodes. Prediction supports another uniform grid with the same physical endpoints and orientation. The source is interpolated onto the prediction grid, its mean is recomputed there, and coordinate features are built on that grid.

Padding is adjusted independently for each axis to approximately preserve its physical extent:

```text
p_new = floor(p_train * (N_new - 1) / (N_train - 1) + 0.5)
```

For example, a training grid of 129 × 129 × 65 with padding 4 becomes a prediction grid of 257 × 257 × 129 with padding 8. The same learned mode counts and weights are used; predicted solutions are not spatially interpolated from coarse-grid outputs. Higher resolution does not guarantee improved physical accuracy.

## Physical limitations

This is a supervised surrogate. No PDE-residual loss, conservation constraint, positivity constraint, or hard zero-boundary enforcement is included. Padding reduces periodic wraparound interactions but does not enforce the physical boundary conditions. The nonlinear GELU blocks do not guarantee superposition, even when the underlying PDE is linear. Source-mean normalization handles overall amplitude scaling, but changes in spatial structure or contrast still require validation on representative data.
