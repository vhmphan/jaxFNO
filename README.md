# JAX FNO for cosmic-ray diffusion

We aim to build an approximate solution for the following diffusion equation 

$$-\left(\frac{\partial^2u}{\partial x^2}+\frac{\partial^2u}{\partial y^2}+\frac{\partial^2u}{\partial z^2}\right)=S(x,y)\delta(z).$$

This might be later adapted for the physical case of infering the distribution of sources for Galactic cosmic rays, To this end, we present a supervised JAX + Equinox + Optax model mapping surface sources `S(x,y)` to `u(x,y,z)`. SInce we have  in mind application for cosmic rays, coordinates x, y, z are in kpc; S and u retain the solver's saved amplitude units.

## Project layout

The top level contains the three command scripts, this README, and the one input NPZ files (by default `uxyz_data.pz`). Supporting files are organized as follows:

- `jaxfno/`: model, data loading, configuration, plotting, prediction archive I/O,
  and optional accuracy helpers.
- `support/requirements.txt`: Python dependencies.
- `support/tests/`: regression tests.
- `support/docs/`: preserved original project brief.
- `model/`: trained checkpoints, loss curves, training history, and comparison plots.

Run commands from the project directory. Hidden `.git/`, `.gitignore`, and `.venv/` remain in place for version control and the existing environment. Training automatically creates `model/` beside `train.py` and saves `model/best_model.npz`; evaluation loads that file by default.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r support/requirements.txt
```

All `.npz` and `.png` files, local environments, and checkpoints are ignored by Git. Supply datasets and a trained checkpoint locally.

## Train

```bash
python train.py
```

Training defaults to `uxyz_data.npz`. The source coordinates are reconstructed from `S.shape` within `uxyz_data.npz` and the physical domain endpoints in the NPZ. Best validation weights and the configuration, coordinates, sample splits, and dataset fingerprint are saved to `model/best_model.npz`. Loss curves and history are saved beside it. 

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

## Continue training from the best checkpoint

```bash
python train.py --resume model/best_model.npz --epochs 50
# Equivalent for the default checkpoint:
python train.py --resume --epochs 50
```

`--epochs` is the maximum number of **additional** epochs in this invocation, not an absolute epoch number. The existing patience-based early stopping still applies. The saved weights, training/validation/test split and architecture are reused. The training dataset must match the checkpoint's values, grid, and sample order. Use `--data PATH` if the same dataset was moved.

The checkpoint does not contain AdamW state, so continuation starts a fresh optimizer and a fresh early-stopping counter. It is not an exact restart of the interrupted optimizer trajectory. The starting checkpoint is validated before training; `best_model.npz` is replaced only if a lower validation loss is reached. Checkpoint writes use a temporary file and atomic replacement to protect the previous best if writing is interrupted.

The epoch display starts at 1 for the new run. Best checkpoints now record their epoch within that run and validation loss. Continuation losses are saved after every completed epoch in `continuation_history.json`; the finished run writes `continuation_loss_curves.png` in the checkpoint directory. Original training history is kept. To change optimization settings, supply `--config config.json` as well; model architecture must remain compatible. `--epochs` also overrides the epoch limit for a fresh run without `--resume`.

## Predict all sources and measure runtime

```bash
python evaluate.py
```

This reads **every source** from `uxyz_test.npz` and writes `uxyz_pred.npz`. Each source is divided by its own spatial mean, and predictions are multiplied by that mean. The input must use the checkpoint's physical domain and physics, but may use a different uniform grid resolution and number of source realizations.

The original source grid is assumed uniform over the same physical x/y domain as the solver, as in the inspected source generator. The loader removes the solver's ghost coordinates and reconstructs source coordinates using `linspace(x[0], x[-1], S.shape[2])` and `linspace(y[0], y[-1], S.shape[1])`, since stored S has order `(N,Ny,Nx)`. It then applies the same linear interpolation onto the solver grid. Domain endpoints and axis orientation must match the checkpoint; the number of grid nodes may differ.

```bash
python evaluate.py --data uxyz_test.npz --output uxyz_pred.npz
```

Prediction works without a `u` array. An optional one inclusive range is
still available:

```bash
python evaluate.py --data uxyz_test.npz --realization-range 1 10
```

Runtime measures the prediction pass that produces the saved arrays, after a warm-up for every minibatch shape. It includes feature encoding, normalization, CPU/device transfers, FNO computation, and output rescaling. Returning NumPy arrays synchronizes JAX device work before the timer stops. Dataset loading, source-grid interpolation, warm-up/JIT compilation, and NPZ writing are excluded. The console reports total and per-realization seconds. Details including device, batch size, throughput, and separate warm-up time are saved inside the prediction archive. No separate runtime JSON file is written.

The prediction NPZ contains:

- `u`: `(N,Nx,Ny,Nz)` predictions in saved solver units, `axis_order="xyz"`.
- `S`: `(N,Nx,Ny)` preprocessed model inputs, `source_axis_order="xy"`.
- `x,y,z`: coordinate vectors without ghost cells; `N`: selected sample count.
- `sample_indices`: zero-based indices in the original input file.
- Source hashes, original file layout, metadata, checkpoint path, and runtime.

## Prediction on another grid resolution

The same command supports, for example, training on 129×129×65 and predicting on 257×257×129 over the same physical domain:

```bash
python evaluate.py --data uxyz_test.npz --realization-range 1 10
```

The loader uses the test file's x/y/z coordinates (after removing ghost nodes). It reconstructs and interpolates the source onto that test x/y grid. The FNO uses those coordinates with the original checkpoint weights and training normalization; no retraining or interpolation of predicted u is performed. Uniform spacing, domain endpoints, axis orientation, and physics metadata are validated. Changing the physical domain is rejected.

Padding is adjusted independently on each axis according to `round_half_up(training_padding * (N_test - 1) / (N_train - 1))`. Thus four training padding cells become eight when grid intervals are doubled. This preserves the physical padding extent to the nearest test-grid cell; Fourier mode counts remain unchanged. The training grid, prediction grid, and padding are printed and recorded with runtime metadata in the prediction NPZ. Same-grid inference retains its previous behavior.

Use `plot_results.py` with ground truth on the **same test grid** to inspect the result. Higher resolution increases memory/runtime and does not guarantee better accuracy. Resolution transfer must be validated against the finer-grid numerical solutions; it is not a claim of physical accuracy.

To additionally compare ground truth and FNO along z at a particular `(x, y)` in kpc:

```bash
python plot_results.py --realization 5 --z-profile 2 -3
```

The profile uses the nearest grid nodes and labels their actual coordinates. It is saved as `model/realization_5_z_profile_ix<index>_iy<index>.png`; the grid indices distinguish profiles at different positions.

## Plot ground truth versus saved predictions

```bash
python plot_results.py --realization 1
```

This reads `uxyz_test.npz` and `uxyz_pred.npz` and produces `model/realization_1_comparison.png`. **It does not load the model or run inference**, and it does not need the paired source file. Realization numbers are **one-based**, including the figure title and filename.

```bash
python plot_results.py --data uxyz_test.npz --predictions uxyz_pred.npz \
  --realization 2 --slice-x 1 --slice-y -2 --slice-z 0.5
```

The figure contains the chosen solver-grid source and XY, XZ, YZ cross-sections of ground truth, FNO prediction, and signed error. These are slices, not integrated projections. Locations default to zero and select the nearest grid node. Reference and prediction share color limits per plane. XZ/YZ panels have the physical 20:8 aspect ratio, with colorbars matching panel heights. Spatial axes are labeled kpc; x/y ticks are −10, −5, 0, 5, 10 and z ticks are −4, −2, 0, 2, 4 on the supplied grid. White source-panel guides mark the x/y slice locations. `--output DIR` changes the plot directory. By default, plots are saved in `model/` beside `plot_results.py` (created automatically if needed), and `uxyz_pred.npz` is saved beside `evaluate.py`, regardless of the working directory. Explicit relative `--output` paths use the working directory. Source hashes and coordinates are checked before comparison.

## Data preprocessing

The supplied data export stores `S` as `(N,Ny,Nx)` and `u` as `(N,Nz,Ny,Nx)`. Coordinates include one ghost cell at each end, but the saved solution already excludes those ghosts. The loader trims coordinates only, transposes u to nxyz, and reproduces the generator's linear source interpolation onto interior x/y nodes. The unused source perimeter is zero, as in the solver RHS. S retains its surface amplitude; the RHS factor `-1/dz` is not part of the input encoding.

For the supplied training data this produces `S: (10,129,129)` and `u: (10,129,129,65)`, on x,y ∈ [−10,10], z ∈ [−4,4]. The inspected solver uses D=1, lambda=0, and homogeneous Dirichlet conditions on all six faces. Padding reduces Fourier wraparound coupling; it does not enforce boundary conditions.

Inspect other formats with `python -m jaxfno.data FILE.npz`. Generic loaders require explicit `dataset_layout` source/target keys, x/y/z keys, axis orders, and units, boundary conditions, and `shared_bvp=true` in metadata. They reject incompatible shapes, nonfinite values, nonuniform grids, and ambiguous axes. The sol3d adapter is the only path that performs its specifically documented interpolation.

Only per-realization source-mean normalization is supported. After loading and source interpolation, each pair is normalized as follows:

```python
S_mean = np.mean(S, axis=(1, 2), keepdims=True)
S_normalized = S / S_mean
u_normalized = u / S_mean[..., None]
```

The mean includes all loaded XY nodes, including the zero source perimeter. Means must be finite and nonzero. Evaluation computes the mean of each test source, feeds its normalized source to the network, and multiplies the output by that mean to save predictions in physical units. Checkpoints use the new format without global scaling constants. Previous checkpoint formats are unsupported: start fresh with `python train.py` without `--resume`. Subsequent runs can resume new-format checkpoints.

Training splits independent samples 80/10/10 with seeded randomness. Supply `group_key` in the layout for related source variants; fractions then apply to groups. Four GELU Fourier blocks retain signed x/y modes and one-sided z modes without overlap on small grids. Training uses AdamW, mean per-sample relative L2, minibatch device transfers, finite-gradient checks, and validation early stopping.

## Unit checks and programmatic prediction

```bash
python -m unittest discover -s support/tests -v
```

These numerical unit checks do not establish physical accuracy or unseen-grid generalization. Optional metric helpers live in `jaxfno/evaluation_metrics.py`; the prediction CLI does not call them.

```python
from train import load_predictor
operator = load_predictor("model/best_model.npz")
u = operator.predict(S)  # training-grid S: (Nx,Ny) -> u: (Nx,Ny,Nz)
# For a different grid over the same domain:
# operator = operator.on_grid(x_test, y_test, z_test)
# u_test = operator.predict(S_on_test_grid)
```

Batches preserve their batch axis, including N=1. For sol3d checkpoints this API expects the preprocessed 129×129 source, not the raw 101×101 source. The prediction CLI performs that preprocessing automatically.
