# JAX FNO for cosmic-ray diffusion

Supervised JAX + Equinox + Optax model mapping `S(N,Nx,Ny)` to
`u(N,Nx,Ny,Nz)`. Training defaults to `data_uxyz.npz` in the working directory.
The supplied `sol3D.py` export is adapted to the shared training grid as described below.
The included synthetic targets are source maps multiplied by a
Gaussian vertical envelope, **not solutions of the diffusion PDE**. Their
metrics only validate the software pipeline.

## Setup and smoke test

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --synthetic
python evaluate.py --checkpoint checkpoints/smoke/best_model.npz
python -m unittest discover -s tests -v
```

The explicit smoke run uses 32 samples on an 8×8×6 grid, width 8, modes (4,4,3),
padding 2, batch size 4, and eight epochs. It saves the best validation model to
`checkpoints/smoke/best_model.npz`, epoch-averaged losses to `history.json`, and
`loss_curves.png`. Evaluation loads that checkpoint without training and writes
`evaluation/metrics.json`, reference/prediction/error slices near z=0 and
at an interior off-plane location, and vertical profiles for the first and
worst test samples. Reference and prediction slices share color limits.
On systems with a read-only home directory, prefix plotting commands with
`MPLCONFIGDIR=/tmp/jaxfno-matplotlib`.

## Physical data

```bash
python train.py
# Or supply configuration and optionally override its dataset path:
python train.py --config physical.json --data data_uxyz.npz
python evaluate.py --checkpoint checkpoints/physical/best_model.npz
```

The default loader now follows the supplied `sol3D.py` generator:

- `S` is stored as `(N,Ny,Nx)` and `u` as `(N,Nz,Ny,Nx)`.
- Saved coordinates include one ghost cell at each end; `u` already excludes
  those ghosts. Only the coordinate vectors are trimmed with `[1:-1]`.
- `sources.npz`, beside the target file, supplies the original source coordinates.
  Its samples must match the exported `S` exactly after the generator's transpose.
- Sources are linearly interpolated onto the interior solver x/y nodes using
  the same SciPy interpolation as the generator. The unused perimeter is zero,
  matching the solver RHS. The input retains surface amplitude S, without the
  RHS factor `-1/dz`. Targets are transposed but never interpolated.

The resulting arrays are `S: (10,129,129)` and `u: (10,129,129,65)` on
`x,y ∈ [-10,10]`, `z ∈ [-4,4]`, with an 8/1/1 sample split. The inspected
boundary handler sets u=0 on all six faces; the generator uses D=1 and lambda=0.
The user confirmed that x, y, and z are in kpc. Source and target amplitudes
remain in their saved solver units. Training on this grid is substantially heavier than the synthetic smoke test.

To customize training, use a JSON configuration such as:

```json
{
  "data_path": "data_uxyz.npz",
  "dataset_layout": {"format": "sol3d", "source_path": "sources.npz"},
  "width": 16,
  "modes": [8, 8, 8],
  "batch_size": 1,
  "epochs": 200,
  "checkpoint_dir": "checkpoints/physical"
}
```

`source_path` is relative to the target NPZ directory, or can be absolute. This
adapter is specific to the inspected generator. The explicit generic layout
below is available for other NPZ formats.

Inspect before configuring the loader:

```bash
python data.py /path/to/data_uxyz.npz
```

For other datasets, verify axis order, units, boundary conditions, and
source-variant groups. Supply paired source maps if absent; the loader
never constructs them from targets. Coordinate arrays must be shared 1D,
strictly monotonic, uniformly spaced vectors. Flattened meshes, per-sample
grids, nonfinite values, and mismatched shapes are rejected. Named axis
permutations are transposed explicitly; arrays are never blindly reshaped or
interpolated by the generic loader. The sol3d adapter explicitly reproduces
the generator's source interpolation. A missing path raises an error; synthetic data must be selected
explicitly.

Create `physical.json` using the **actual inspected** keys, axes, units, and
solver boundary conditions. This is a schema example, with placeholders that
must be replaced:

```json
{
  "data_path": "/path/to/data_uxyz.npz",
  "dataset_layout": {
    "source_key": "ACTUAL_SOURCE_KEY",
    "target_key": "ACTUAL_TARGET_KEY",
    "x_key": "ACTUAL_X_KEY",
    "y_key": "ACTUAL_Y_KEY",
    "z_key": "ACTUAL_Z_KEY",
    "source_axes": "nxy",
    "target_axes": "nxyz",
    "metadata": {
      "units": {"x": "REQUIRED", "y": "REQUIRED", "z": "REQUIRED", "S": "REQUIRED", "u": "REQUIRED"},
      "boundary_conditions": "REQUIRED: geometry and prescribed values on each boundary",
      "shared_bvp": true,
      "homogeneous_boundary_conditions": false
    }
  },
  "width": 16,
  "modes": [8, 8, 8],
  "padding": 4,
  "batch_size": 1,
  "epochs": 200,
  "patience": 20,
  "checkpoint_dir": "checkpoints/physical"
}
```

For example, use `target_axes: "nzyx"` only if that is the actual storage order.
Set `shared_bvp` to true only after verifying identical geometry, units, and
boundary conditions across samples. Set `homogeneous_boundary_conditions` to
true only when confirmed; this enables zero-source, amplitude-scaling, and
superposition diagnostics. Padding reduces Fourier wraparound coupling but
does not enforce boundary values or establish periodicity. No boundary
constraints are imposed without a verified geometry and prescribed values.

If samples contain related source variants, add `group_key` to `dataset_layout`
pointing to one group ID per sample. Otherwise samples are treated as independent.
Splits are seeded 80/10/10, with rounding and at least one sample per subset;
with groups these fractions apply to groups, so sample proportions may differ.
At least three independent samples/groups are required. Confirm source/target
sample correspondence upstream: matching array shapes alone cannot establish it.

The NPZ may alternatively contain a non-object, scalar JSON string `metadata`
with `layout`, `units`, `boundary_conditions`, and `shared_bvp`; the configuration
can override that mapping. Pickled object arrays are never loaded. Domain
endpoints are recorded from the validated coordinates.

```bash
python train.py --config physical.json
python evaluate.py --checkpoint checkpoints/physical/best_model.npz
```

## Realization comparisons

`evaluate.py` defaults to `checkpoints/physical/best_model.npz` and reloads the
saved sol3d preprocessing, coordinates, normalization, and held-out split.
Choose any zero-based dataset index (0–9 for the supplied data):

```bash
python evaluate.py --realization 6
python evaluate.py --realization 2 --slice-x 1 --slice-y -2 --slice-z 0.5
```

Without `--realization`, the first held-out test sample is plotted. The output
`checkpoints/physical/evaluation/realization_6_comparison.png` contains the
chosen source S(x,y) on the model's solver grid and XY, XZ, YZ cross-sections of
u, with columns for ground truth, FNO, and signed error. These are slices,
not integrated projections. Slice coordinates default to zero and select the
nearest grid node; actual coordinates are printed on the figure. Ground truth
and prediction share color limits within each plane. Spatial panels use equal
physical scaling: XZ and YZ have a 20:8 width-to-height ratio, with colorbars
matching the panel height. Coordinate labels are in kpc; x/y ticks are
−10, −5, 0, 5, 10 and z ticks are −4, −2, 0, 2, 4 for this dataset.

The chosen realization may be from any split, which is identified on the plot;
reported evaluation metrics still use only the saved test samples. The source
panel shows the interpolated 129×129 surface input, including its zero
perimeter. Use `--output DIR` to change the destination. Existing loss curves,
off-plane slices, and vertical profiles remain available.

## Inference and validation

```python
from train import load_predictor

operator = load_predictor("checkpoints/smoke/best_model.npz")
u = operator.predict(S)  # S: (Nx,Ny) -> u: (Nx,Ny,Nz)
u_batch = operator.predict(S_batch)  # (N,Nx,Ny) -> (N,Nx,Ny,Nz), including N=1
```

For a sol3d checkpoint, `predict(S)` expects source maps already on the saved
129×129 solver grid, including the zero perimeter, as returned by
`load_dataset(..., layout={"format": "sol3d"})`. It does not accept raw 101×101
source maps.

Both source and target use fixed global max-absolute scales fitted on training
samples only; an all-zero training field uses scale 1. Evaluation and inference
reuse saved scales and return physical units. Checkpoints include model
configuration, real-valued parameter leaves, scales, coordinates, split indices,
groups when provided, metadata, and a fingerprint of paired sample values/order.
`evaluate.py --data /new/path.npz` allows moving the same dataset; modified or
reordered samples are rejected. Legacy checkpoints from the original model are
incompatible and require retraining; the old files are left in place.

The four-channel encoding broadcasts S along z and appends normalized x/y/z.
This broadcast is an input representation, not a replacement for the delta
function in the PDE. Four GELU Fourier blocks retain four signed x/y quadrants
and one-sided z modes, clip mode counts to avoid overlapping small-grid slices,
and use explicit inverse FFT sizes. Spectral real/imaginary parameters and
pointwise paths use independent random keys. Lifting, optional high-end zero
padding, and cropping operate on channel-first arrays.

Training transfers only encoded minibatches to JAX, uses AdamW at 1e-3 and
mean per-sample relative L2 with a configurable denominator floor, checks finite
gradients, and selects the lowest validation loss with patience-based stopping.
Evaluation reports mean/median/worst relative L2 for nonzero targets and global
RMSE for all targets; zero-source and zero-target absolute errors are separate.
Linearity diagnostics measure defects of the learned model, not PDE accuracy.
An ordinary FNO does not enforce linearity.

The regression suite verifies signed spectral reconstruction on small/odd grids,
padding/output shapes, finite gradients including exact fits, explicit dataset
mapping and validation, grouped splits, training-scale evaluation, checkpoint
reload agreement, preserved batch axes, linearity diagnostics, and one-sample
overfitting. A CPU smoke run achieved test mean relative L2 0.4193 and RMSE 0.0950;
one-sample relative L2 fell from 0.9976 to 0.0143. These are synthetic software
checks only. No physical accuracy or unseen-grid generalization is claimed.
