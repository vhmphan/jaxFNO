# JAX FNO for cosmic-ray diffusion

Supervised JAX + Equinox + Optax model mapping surface sources `S(N,Nx,Ny)` to
`u(N,Nx,Ny,Nz)`. Prediction and comparison plotting are separate commands.
Coordinates x, y, z are in kpc; S and u retain the solver's saved amplitude units.

## Project layout

The top level contains the three command scripts, this README, and the two input
NPZ files. Supporting files are organized as follows:

- `jaxfno/`: model, data loading, configuration, plotting, prediction archive I/O,
  and optional accuracy helpers.
- `support/requirements.txt`: Python dependencies.
- `support/tests/`: regression tests.
- `support/docs/`: preserved original project brief, also merged below.
- `model/`: trained checkpoints, loss curves, and training history.
- `artifacts/`: previously generated outputs and archived checkpoints. New predictions
  default to the folder containing the scripts; comparison plots
  default to `model/`.

Run commands from the project directory. Hidden `.git/`, `.gitignore`, and
`.venv/` remain in place for version control and the existing environment.
Training automatically creates `model/` beside `train.py` and saves
`model/best_model.npz`; evaluation loads that file by default. The existing trained
checkpoint has been copied there from `artifacts/`. Explicit `checkpoint_dir`
settings remain supported; relative paths are resolved beside `train.py`.
Synthetic smoke training uses `model/smoke/` to preserve the physical model.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r support/requirements.txt
```

All `.npz` and `.png` files, local environments, and checkpoints are ignored by
Git. Supply datasets and a trained checkpoint locally.

## Train

```bash
python train.py
```

Training defaults to `uxyz_data.npz`, using the inspected `sol3D.py` export
format. No companion source file is needed: the uniform source coordinates are
reconstructed from `S.shape` and the physical domain endpoints in the NPZ. Best validation weights and the configuration,
normalization scales, coordinates, sample splits, and dataset fingerprint are
saved to `model/best_model.npz`. Loss curves and history are saved
beside it. No new PDE solver or residual loss is used.

To customize training, save a JSON configuration and pass `--config config.json`:

```json
{
  "data_path": "uxyz_data.npz",
  "dataset_layout": {"format": "sol3d", "source_grid": "uniform_domain"},
  "width": 16,
  "modes": [8, 8, 8],
  "padding": 4,
  "batch_size": 1,
  "epochs": 200,
  "patience": 20,
  "checkpoint_dir": "model"
}
```

## Predict all sources and measure runtime

```bash
python evaluate.py
```

This reads **every source** from `uxyz_test.npz` and writes `uxyz_pred.npz`.
It does **not read ground-truth u, compute accuracy metrics, or plot anything**.
The saved training normalization is reused. The input must use the checkpoint's
grid and physics, but may contain a different number of source realizations.

For sol3d exports, prediction uses only `--data`; no companion source file is
needed. The original source grid is assumed uniform over the same physical x/y
domain as the solver, as in the inspected source generator. The loader removes
the solver's ghost coordinates and reconstructs source coordinates using
`linspace(x[0], x[-1], S.shape[2])` and
`linspace(y[0], y[-1], S.shape[1])`, since stored S has order `(N,Ny,Nx)`.
It then applies the same linear interpolation onto the solver grid. Both the
coordinate values and grid dimensions must match the saved checkpoint.

```bash
python evaluate.py --data uxyz_test.npz --output uxyz_pred.npz
```

Existing checkpoints that name a training `source_path` remain compatible;
both prediction and training ignore that path for sol3d exports.
Prediction works without a `u` array. An optional one-based inclusive range is
still available:

```bash
python evaluate.py --data uxyz_test.npz --realization-range 1 10
```

Runtime measures the prediction pass that produces the saved arrays, after a
warm-up for every minibatch shape. It includes feature encoding, normalization,
CPU/device transfers, FNO computation, and output rescaling. Returning NumPy
arrays synchronizes JAX device work before the timer stops. Dataset loading,
source-grid interpolation, warm-up/JIT compilation, and NPZ writing are excluded.
The console reports total and per-realization seconds. Details including device,
batch size, throughput, and separate warm-up time are saved inside the prediction archive. No separate runtime JSON file is written.

The prediction NPZ contains:

- `u`: `(N,Nx,Ny,Nz)` predictions in saved solver units, `axis_order="xyz"`.
- `S`: `(N,Nx,Ny)` preprocessed model inputs, `source_axis_order="xy"`.
- `x,y,z`: coordinate vectors without ghost cells; `N`: selected sample count.
- `sample_indices`: zero-based indices in the original input file.
- Source hashes, original file layout, metadata, checkpoint path, and runtime.

## Plot ground truth versus saved predictions

```bash
python plot_results.py --realization 1
```

This reads `uxyz_test.npz` and `uxyz_pred.npz` and produces
`model/realization_1_comparison.png`. **It does not load the model or
run inference**, and it does not need the paired source file. Realization numbers
are **one-based**, including the figure title and filename.

```bash
python plot_results.py --data uxyz_test.npz --predictions uxyz_pred.npz \
  --realization 2 --slice-x 1 --slice-y -2 --slice-z 0.5
```

The figure contains the chosen solver-grid source and XY, XZ, YZ cross-sections
of ground truth, FNO prediction, and signed error. These are slices, not integrated
projections. Locations default to zero and select the nearest grid node. Reference
and prediction share color limits per plane. XZ/YZ panels have the physical 20:8
aspect ratio, with colorbars matching panel heights. Spatial axes are labeled kpc;
x/y ticks are −10, −5, 0, 5, 10 and z ticks are −4, −2, 0, 2, 4 on the supplied grid.
White source-panel guides mark the x/y slice locations. `--output DIR` changes
the plot directory. By default, plots are saved in `model/` beside
`plot_results.py` (created automatically if needed), and
`uxyz_pred.npz` is saved beside `evaluate.py`, regardless
of the working directory. Explicit relative `--output` paths use the working
directory. Source hashes and coordinates are checked before comparison.

## Data preprocessing

The supplied sol3d export stores `S` as `(N,Ny,Nx)` and `u` as `(N,Nz,Ny,Nx)`.
Coordinates include one ghost cell at each end, but the saved solution already
excludes those ghosts. The loader trims coordinates only, transposes u to nxyz,
and reproduces the generator's linear source interpolation onto interior x/y
nodes. The unused source perimeter is zero, as in the solver RHS. S retains its
surface amplitude; the RHS factor `-1/dz` is not part of the input encoding.

For the supplied training data this produces `S: (10,129,129)` and
`u: (10,129,129,65)`, on x,y ∈ [−10,10], z ∈ [−4,4]. The inspected solver uses
D=1, lambda=0, and homogeneous Dirichlet conditions on all six faces. Padding
reduces Fourier wraparound coupling; it does not enforce boundary conditions.

Inspect other formats with `python -m jaxfno.data FILE.npz`. Generic loaders require
explicit `dataset_layout` source/target keys, x/y/z keys, axis orders, and units,
boundary conditions, and `shared_bvp=true` in metadata. They reject incompatible
shapes, nonfinite values, nonuniform grids, and ambiguous axes. The sol3d adapter
is the only path that performs its specifically documented interpolation.

Training splits independent samples 80/10/10 with seeded randomness. Supply
`group_key` in the layout for related source variants; fractions then apply to
groups. Global max-absolute normalization is fitted on training data only. Four
GELU Fourier blocks retain signed x/y modes and one-sided z modes without overlap
on small grids. Training uses AdamW, mean per-sample relative L2, minibatch device
transfers, finite-gradient checks, and validation early stopping.

## Smoke test and programmatic prediction

```bash
python train.py --synthetic
python -m unittest discover -s support/tests -v
```

Synthetic targets are a source multiplied by a Gaussian vertical envelope, not
PDE solutions. These checks do not establish physical accuracy or unseen-grid
generalization. Optional metric helpers live in `jaxfno/evaluation_metrics.py`; the
prediction CLI does not call them.

```python
from train import load_predictor
operator = load_predictor("model/best_model.npz")
u = operator.predict(S)  # solver-grid S: (Nx,Ny) -> u: (Nx,Ny,Nz)
```

Batches preserve their batch axis, including N=1. For sol3d checkpoints this API
expects the preprocessed 129×129 source, not the raw 101×101 source. The prediction
CLI performs that preprocessing automatically.

## Original scientific brief

The original brief is preserved below for scientific context. The current usage
above reflects subsequent changes: source-grid reconstruction, separate prediction
and plotting commands, the renamed input files, and the reorganized directories.

### Codex task: FNO for Galactic cosmic-ray diffusion

Build a minimal Python project using **JAX, Equinox, and Optax** to learn
`S(x,y) -> u(x,y,z)` from N paired numerical solutions in `data_uxyz.npz`:

$$-\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}+\frac{\partial^2u}{\partial z^2}\right)=S(x,y)\delta(z).$$

This is steady, homogeneous, isotropic diffusion with the constant coefficient
absorbed into the normalization. No diffusion-coefficient input is needed.
N is the number of independent samples, not the number of optimizer minibatches.

#### 1. Inspect and load data

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

#### 2. Implement a 3D FNO

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

#### 3. Train and evaluate

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

#### 4. Deliver

Provide `data.py`, `fno.py`, `train.py`, `evaluate.py`, configuration, requirements,
and a short README with exact commands. Save model configuration, normalization,
coordinates, and split indices with the checkpoint. Expose `predict(S)` returning
u in physical units. Keep training supervised; no new diffusion solver or physics
residual loss. Do not claim accuracy on unseen grids without testing it.

First summarize the observed data layout and any blocking missing information,
then implement and run a smoke test. If the dataset is unavailable, provide a
clearly labeled synthetic smoke test without claiming physical validation.
