# Codex task: FNO for Galactic cosmic-ray diffusion

Build a minimal Python project using **JAX, Equinox, and Optax** to learn
`S(x,y) -> u(x,y,z)` from N paired numerical solutions in `data_uxyz.npz`:

$$-\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}+\frac{\partial^2u}{\partial z^2}\right)=S(x,y)\delta(z).$$

This is steady, homogeneous, isotropic diffusion with the constant coefficient
absorbed into the normalization. No diffusion-coefficient input is needed.
N is the number of independent samples, not the number of optimizer minibatches.

## 1. Inspect and load data

- Use NumPy to inspect NPZ keys, shapes, dtypes, and coordinates before coding
  the loader. Do not assume the file's key names or axis order.
- Map to `S: (N,Nx,Ny)`, `u: (N,Nx,Ny,Nz)`, and coordinate vectors `x,y,z`.
  Validate pairing, finite values, axis order, and a shared uniform Cartesian
  grid. Do not blindly reshape flattened coordinates or silently interpolate.
- If source maps are absent, request their file or the exact source-generation
  parameters and sample correspondence. Do not fabricate sources from u.
- Determine domain, units, and boundary conditions from metadata or solver code;
  ask if unavailable. Use the same uniquely specified boundary-value problem
  for all samples. Do not assume periodic boundaries.

## 2. Implement a 3D FNO

- Broadcast each surface source S along z, then concatenate normalized x,y,z
  coordinates: four input channels. This is an encoding of the 2D source,
  not a physical replacement of the delta function by a volume source.
- Use pointwise lifting, four Fourier blocks (spectral convolution + pointwise
  linear path, followed by GELU), and pointwise projection to one u channel.
- Start with width 16 and up to 8 Fourier modes per axis; make these configurable.
  Use channel-first arrays internally and document all shape conversions.
- Implement spectral layers with `jnp.fft.rfftn/irfftn` over spatial axes only.
  Retain positive/negative modes on the first two axes and one-sided modes on
  the last; prevent overlapping slices on small grids. Specify inverse shape.
  Store real/imaginary trainable weights as separate real arrays.
- Support configurable padding and crop back to the original grid for
  nonperiodic domains. Padding does not enforce boundary conditions; enforce
  boundary values only when the actual geometry and prescribed values permit it.

## 3. Train and evaluate

- Seed all randomness. Split complete samples 80/10/10 into train/validation/test;
  keep related source variants together and ensure nonempty splits.
- Fit fixed global source/target normalization scales on training data only;
  preserve amplitude information and save the scales. Do not normalize each
  sample independently.
- Train using mean per-sample relative L2 loss with a denominator floor, AdamW
  (initial learning rate 1e-3), configurable batch size (start 1), up to 200
  epochs, and validation early stopping. Save the best validation checkpoint.
- Use `eqx.filter_jit`, `eqx.filter_value_and_grad`, and `jax.vmap` for training
  and batched inference. Keep NumPy/file I/O outside compiled functions and
  transfer minibatches rather than the full dataset to the accelerator.
- Evaluate the selected model on held-out test samples in physical units:
  mean/median/worst relative L2 and global RMSE. Report absolute errors for
  zero-source targets separately.
- Save loss curves and reference/prediction/error slices near z=0 and off-plane,
  plus vertical profiles. Share color limits between reference and prediction.
- Verify finite gradients, output shapes, tiny-subset overfitting, and checkpoint
  reload agreement. For homogeneous boundary conditions, also check zero-source,
  amplitude-scaling, and superposition errors: the true operator is linear,
  whereas an ordinary FNO does not enforce linearity.

## 4. Deliver

Provide `data.py`, `fno.py`, `train.py`, `evaluate.py`, configuration, requirements,
and a short README with exact commands. Save model configuration, normalization,
coordinates, and split indices with the checkpoint. Expose `predict(S)` returning
u in physical units. Keep training supervised; no new diffusion solver or physics
residual loss. Do not claim accuracy on unseen grids without testing it.

First summarize the observed data layout and any blocking missing information.
Physical validation requires the paired numerical dataset.
