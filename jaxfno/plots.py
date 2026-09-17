"""Headless diagnostics in physical units."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import StrMethodFormatter


def _coordinate_label(dataset, name):
    return f"{name} (kpc)"


def _format_coordinate_axis(axis, coord, name):
    ticks = [-4, -2, 0, 2, 4] if name == "z" else [-10, -5, 0, 5, 10]
    if np.allclose([np.min(coord), np.max(coord)], [ticks[0], ticks[-1]]):
        axis.set_ticks(ticks)
    axis.set_major_formatter(StrMethodFormatter("{x:g}"))


def _format_plane(ax, dataset, horizontal, vertical, xlabel, ylabel):
    ax.set(xlabel=_coordinate_label(dataset, xlabel), ylabel=_coordinate_label(dataset, ylabel),
           xlim=(horizontal[0], horizontal[-1]), ylim=(vertical[0], vertical[-1]))
    ax.set_aspect("equal", adjustable="box")
    _format_coordinate_axis(ax.xaxis, horizontal, xlabel)
    _format_coordinate_axis(ax.yaxis, vertical, ylabel)


def _panel_colorbar(fig, ax, image, label=""):
    # Axes-relative inset follows the displayed panel after equal-aspect sizing.
    cax = ax.inset_axes([1.04, 0, 0.045, 1])
    return fig.colorbar(image, cax=cax, label=label)


def plot_history(history, path):
    fig, ax = plt.subplots()
    epochs = np.arange(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="train (epoch average)")
    ax.plot(epochs, history["val"], label="validation")
    ax.set(xlabel="Epoch", ylabel="Mean relative L2", yscale="log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prediction(dataset, index, pred, output):
    output = Path(output)
    ref = dataset.u[index]
    near = int(np.argmin(np.abs(dataset.z)))
    off = int(np.argmax(np.abs(dataset.z)))
    # Prefer an interior off-plane slice if the far edge is a boundary.
    if len(dataset.z) > 3:
        off = 1 if off == 0 else len(dataset.z) - 2
        if off == near:
            off = int(np.argmax(np.abs(dataset.z)))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row, iz in enumerate((near, off)):
        low = min(ref[:, :, iz].min(), pred[:, :, iz].min())
        high = max(ref[:, :, iz].max(), pred[:, :, iz].max())
        error = pred[:, :, iz] - ref[:, :, iz]
        limit = max(float(np.abs(error).max()), 1e-12)
        for col, (values, title) in enumerate(((ref[:, :, iz], "Reference"),
                                             (pred[:, :, iz], "Prediction"), (error, "Prediction − reference"))):
            ax = axes[row, col]
            im = ax.pcolormesh(dataset.x, dataset.y, values.T, shading="auto",
                               cmap="RdBu_r" if col == 2 else "viridis",
                               vmin=-limit if col == 2 else low, vmax=limit if col == 2 else high)
            ax.set_title(f"{title}; {_coordinate_label(dataset, 'z')}={dataset.z[iz]:.4g}")
            _format_plane(ax, dataset, dataset.x, dataset.y, "x", "y")
            _panel_colorbar(fig, ax, im)
    fig.suptitle(f"Sample {index}: u in dataset units")
    fig.tight_layout()
    fig.savefig(output / f"sample_{index}_slices.png", dpi=150)
    plt.close(fig)
    points = [(len(dataset.x) // 2, len(dataset.y) // 2),
              np.unravel_index(np.argmax(np.abs(dataset.S[index])), dataset.S[index].shape)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (ix, iy) in zip(axes, points):
        ax.plot(dataset.z, ref[ix, iy], label="reference")
        ax.plot(dataset.z, pred[ix, iy], "--", label="prediction")
        ax.set(xlabel=_coordinate_label(dataset, "z"), ylabel="u (dataset units)", title=f"x={dataset.x[ix]:.3g}, y={dataset.y[iy]:.3g}")
        _format_coordinate_axis(ax.xaxis, dataset.z, "z")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"sample_{index}_profiles.png", dpi=150)
    plt.close(fig)


def plot_z_profile(dataset, index, pred, output, *, x=0.0, y=0.0, display_realization=None):
    """Plot truth and prediction along z at the nearest requested XY grid node."""
    ref = dataset.u[index]
    pred = np.asarray(pred)
    if pred.shape != ref.shape or not np.isfinite(pred).all():
        raise ValueError("Prediction must be finite and match the selected target shape")
    indices = []
    for name, requested in (("x", x), ("y", y)):
        coord = getattr(dataset, name)
        if not np.isfinite(requested) or not coord.min() <= requested <= coord.max():
            raise ValueError(f"Requested {name} profile position is outside the dataset domain")
        indices.append(int(np.argmin(np.abs(coord - requested))))
    ix, iy = indices
    number = index if display_realization is None else display_realization
    numbering = "zero-based" if display_realization is None else "one-based"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"realization_{number}_z_profile_ix{ix}_iy{iy}.png"
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(dataset.z, ref[ix, iy, :], label="Ground truth")
    ax.plot(dataset.z, pred[ix, iy, :], "--", label="FNO")
    ax.set(xlabel="z (kpc)", ylabel="u (dataset units)",
           xlim=(dataset.z.min(), dataset.z.max()),
           title=f"Realization {number} ({numbering})\n"
                 f"Z profile at x={dataset.x[ix]:.4g}, y={dataset.y[iy]:.4g} kpc")
    _format_coordinate_axis(ax.xaxis, dataset.z, "z")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_realization(dataset, index, pred, output, *, slice_coordinates=(0.0, 0.0, 0.0), subset=None, display_realization=None, relative_error=False):
    """Source plus orthogonal XY/XZ/YZ slices (not line-of-sight integrals).

    Coordinates select nearest grid nodes. All fields are in dataset units,
    with shared reference/FNO color limits within each plane.
    relative_error uses abs(pred - truth) / truth, masks zero truth, and
    saves the comparison with a _rel filename suffix.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ref = dataset.u[index]
    pred = np.asarray(pred)
    if pred.shape != ref.shape or not np.isfinite(pred).all():
        raise ValueError("Prediction must be finite and match the selected target shape")
    # mask = ref > 1.0e-4 * np.nanmean(ref)
    # mean_relative_error = np.nanmean(np.abs(ref[mask]-pred[mask])/ref[mask])
    indices = []
    for name, requested in zip("xyz", slice_coordinates):
        coord = getattr(dataset, name)
        if not np.isfinite(requested) or not coord.min() <= requested <= coord.max():
            raise ValueError(f"Requested {name} slice is outside the dataset domain")
        indices.append(int(np.argmin(np.abs(coord - requested))))
    ix, iy, iz = indices
    planes = [
        (dataset.x, dataset.y, ref[:, :, iz], pred[:, :, iz], "x", "y", f"XY at z={dataset.z[iz]:.4g}"),
        (dataset.x, dataset.z, ref[:, iy, :], pred[:, iy, :], "x", "z", f"XZ at y={dataset.y[iy]:.4g}"),
        (dataset.y, dataset.z, ref[ix, :, :], pred[ix, :, :], "y", "z", f"YZ at x={dataset.x[ix]:.4g}"),
    ]
    fig = plt.figure(figsize=(14, 12), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.08, wspace=0.18, hspace=0.04)
    xy_ratio = np.ptp(dataset.y) / np.ptp(dataset.x)
    xz_ratio = np.ptp(dataset.z) / np.ptp(dataset.x)
    yz_ratio = np.ptp(dataset.z) / np.ptp(dataset.y)
    grid = fig.add_gridspec(4, 3, height_ratios=[xy_ratio, xy_ratio, xz_ratio, yz_ratio])
    source_ax = fig.add_subplot(grid[0, 0])
    source_image = source_ax.pcolormesh(dataset.x, dataset.y, dataset.S[index].T,
                                      shading="auto", cmap="viridis")
    source_ax.set_title("Chosen source S(x,y) — model input")
    _format_plane(source_ax, dataset, dataset.x, dataset.y, "x", "y")
    _panel_colorbar(fig, source_ax, source_image, label="S (dataset units)")
    source_ax.axvline(dataset.x[ix], color="white", ls="--", lw=0.8)
    source_ax.axhline(dataset.y[iy], color="white", ls="--", lw=0.8)
    info_ax = fig.add_subplot(grid[0, 1:])
    info_ax.set_axis_off()
    kind = dataset.metadata.get("kind", "physical")
    realization_label = (f"Realization {index} (zero-based)" if display_realization is None
                         else f"Realization {display_realization} (one-based)")
    info = (realization_label + (f" · {subset} subset" if subset else "")
            + f"\nDataset: {kind}\n"
            + f"Prediction resolution (Nx × Ny × Nz): {pred.shape[0]} × {pred.shape[1]} × {pred.shape[2]}\n"
            + "XY, XZ, YZ cross-sections at the labeled grid coordinates.\n"
            + "Ground truth and FNO share a color scale in each row.\n"
            + ("Relative error = |FNO − ground truth| / ground truth (dimensionless).\n"            
               if relative_error else "Errors are FNO − ground truth, in dataset units."))
    if relative_error:
        valid = ref > 1.0e-4 * np.nanmean(ref)
        if np.any(valid):
            truth_values = np.asarray(ref[valid], dtype=np.float64)
            relative_values = np.abs(np.asarray(pred[valid], dtype=np.float64) - truth_values) / truth_values
            info += f"Mean relative error (full volume, nonzero truth): {np.mean(relative_values):.4g}"
        else:
            info += "Mean relative error: undefined (ground truth is zero everywhere)."
    # if dataset.metadata.get("layout", {}).get("format") == "sol3d":
    #     info += "\nS is the interpolated solver-grid source; x, y, z are in kpc."
    info_ax.text(0.05, 0.5, info, va="center", fontsize=12, linespacing=1.7)
    for row, (horizontal, vertical, truth, estimate, xlabel, ylabel, title) in enumerate(planes, 1):
        low, high = min(truth.min(), estimate.min()), max(truth.max(), estimate.max())
        error = estimate - truth
        if relative_error:
            error = np.ma.masked_invalid(np.divide(
                np.abs(error), truth, out=np.full(truth.shape, np.nan, dtype=float),
                where=truth != 0))
            error = np.ma.masked_less_equal(error, 0)
        error_label = "Relative error" if relative_error else "Error"
        valid_error = np.ma.asarray(error).compressed()
        limit = max(float(np.abs(valid_error).max()), 1e-12) if valid_error.size else 1.0
        if relative_error:
            error_norm = LogNorm(vmin=1e-3, vmax=10, clip=True)
        for col, (values, label) in enumerate(((truth, "Ground truth"), (estimate, "FNO"), (error, error_label))):
            ax = fig.add_subplot(grid[row, col])
            color_limits = ({"norm": error_norm} if relative_error else {"vmin": -limit, "vmax": limit}) if col == 2 else {"vmin": low, "vmax": high}
            im = ax.pcolormesh(horizontal, vertical, values.T, shading="auto",
                               cmap=("magma" if relative_error else "RdBu_r") if col == 2 else "viridis",
                               **color_limits)
            ax.set_title(f"{label}: {title} kpc")
            _format_plane(ax, dataset, horizontal, vertical, xlabel, ylabel)
            _panel_colorbar(fig, ax, im, label="Relative error"
                           if col == 2 and relative_error else "u (dataset units)")
    number = index if display_realization is None else display_realization
    suffix = "_rel" if relative_error else ""
    path = output / f"realization_{number}_comparison{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
