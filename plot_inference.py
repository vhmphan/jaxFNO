"""Plot saved ground truth and inference results; requires NumPy and Matplotlib.

Run: python -B plot_inference.py results_fno/seed42_o656l61k
Choose a zero-based z slice: python -B plot_inference.py RUN_DIRECTORY --z-index 2
Logarithmic colors: python -B plot_inference.py RUN_DIRECTORY --log-scale
Observed voxels only: python -B plot_inference.py RUN_DIRECTORY --mask
Custom mask: python -B plot_inference.py RUN_DIRECTORY --mask observation_mask.npy
Outline the observation region without hiding data: python -B plot_inference.py RUN_DIRECTORY --dash
Reads truth.npz and posterior.npz from nifty_example.py. No model is loaded and
no inference is run. The figure is saved directly in RUN_DIRECTORY as
ground_truth_vs_inference.png (or ground_truth_vs_inference_log.png in log mode).
Plotting again replaces that image. All u error panels show relative error.
Log mode sets nonpositive/nonfinite plotted values to NaN, without clipping or a floor.
Masking applies only to u; all three S panels remain unmasked. Masked plots
have a _masked filename suffix. True mask entries are observed voxels.
Both u panels span mean(truth u)/100 to 10*mean(truth u), using the full
selected slice before observation masking. Log relative error spans 0.01 to 10.
The four rows show S, an XY u slice, the XZ projection (integrated over y),
and the YZ projection (integrated over x). Projection units are u units times kpc.
With --mask, only observed nodes contribute to the trapezoidal integral;
lines of sight with no observations are NaN. Error is computed after integration.
--dash uses the saved observation mask (or the custom --mask file). Boxes bound
observed cells, including their half-cell edges. S uses the XY footprint over z;
the XY u row uses the selected slice; projection rows use projected footprints.
Irregular masks get a bounding rectangle; empty footprints get no box.
"""
from pathlib import Path
import argparse

import numpy as np
from jaxfno._plot_cache import temporary_plot_cache


# Text sizes in points, shared by map axes and color bars.
TICK_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
TITLE_FONT_SIZE = 18
FIGURE_TITLE_FONT_SIZE = 20


def load_results(run_directory):
    with np.load(run_directory / "truth.npz", allow_pickle=False) as truth:
        coords = tuple(truth[a].copy() for a in "xyz")
        source, u = truth["S"].copy(), truth["u"].copy()
        if str(truth["source_axis_order"].item()) != "xy" or str(truth["axis_order"].item()) != "xyz":
            raise ValueError("Expected truth source axes xy and output axes xyz")
    with np.load(run_directory / "posterior.npz", allow_pickle=False) as posterior:
        if str(posterior["source_axis_order"].item()) != "xy" or str(posterior["axis_order"].item()) != "xyz":
            raise ValueError("Expected posterior source axes xy and output axes xyz")
        for axis, coordinate in zip("xyz", coords):
            if not np.array_equal(coordinate, posterior[axis]):
                raise ValueError(f"Truth and inference {axis} coordinates differ")
        mean, std, inferred_u = (posterior[k].copy() for k in ("S_mean", "S_std", "u_mean"))
    for coordinate in coords:
        if coordinate.ndim != 1 or len(coordinate) < 2 or not np.isfinite(coordinate).all():
            raise ValueError("Coordinates must be finite one-dimensional arrays of length >= 2")
    shape = tuple(len(c) for c in coords)
    for name, values, expected in (("S", source, shape[:2]), ("S_mean", mean, shape[:2]),
                                   ("S_std", std, shape[:2]), ("u", u, shape),
                                   ("u_mean", inferred_u, shape)):
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name}: expected finite values with shape {expected}")
    if np.any(std < 0):
        raise ValueError("Source standard deviation cannot be negative")
    return coords, source, mean, std, u, inferred_u


def load_mask(run_directory, selection, coords):
    if selection is True:
        with np.load(run_directory / "observations.npz", allow_pickle=False) as data:
            mask = data["mask"].copy()
            for axis, coordinate in zip("xyz", coords):
                if not np.array_equal(data[axis], coordinate):
                    raise ValueError(f"Observation mask {axis} coordinates differ from results")
    else:
        mask = np.load(selection, allow_pickle=False)
    shape = tuple(len(c) for c in coords)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(f"Mask must be a boolean XYZ array with shape {shape}")
    return mask


def project_u(volume, coordinate, axis, mask=None):
    """Trapezoidal line integral; an optional mask selects contributing nodes."""
    steps = np.diff(coordinate)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Projection coordinates must be strictly monotonic")
    steps = np.abs(steps)
    weights = np.concatenate(([steps[0] / 2], (steps[:-1] + steps[1:]) / 2, [steps[-1] / 2]))
    shape = [1] * volume.ndim
    shape[axis] = len(coordinate)
    values = volume if mask is None else np.where(mask, volume, 0.0)
    projection = np.sum(values * weights.reshape(shape), axis=axis)
    if mask is not None:
        projection = np.where(mask.any(axis=axis), projection, np.nan)
    return projection


def draw_mask_box(ax, horizontal, vertical, footprint):
    """Outline the bounding rectangle of True cells without changing plot data."""
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects

    if not footprint.any():
        return

    def cell_bounds(coordinate, occupied):
        edges = np.concatenate(([coordinate[0] - (coordinate[1] - coordinate[0]) / 2],
                                (coordinate[:-1] + coordinate[1:]) / 2,
                                [coordinate[-1] + (coordinate[-1] - coordinate[-2]) / 2]))
        indices = np.flatnonzero(occupied)
        return sorted((edges[indices[0]], edges[indices[-1] + 1]))

    left, right = cell_bounds(horizontal, footprint.any(axis=1))
    bottom, top = cell_bounds(vertical, footprint.any(axis=0))
    box = Rectangle((left, bottom), right - left, top - bottom, fill=False,
                    edgecolor="white", linewidth=2, linestyle="--", zorder=10)
    box.set_path_effects([patheffects.Stroke(linewidth=3.5, foreground="black"), patheffects.Normal()])
    ax.add_patch(box)


def draw_projection_rows(fig, axes, coords, truth, inferred, log_scale=False, selected=None):
    """Draw XZ and YZ comparisons into the last two rows of the shared figure."""
    from matplotlib.colors import LogNorm, Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for row, (plane, integrated_axis, horizontal_axis) in enumerate((("XZ", 1, 0), ("YZ", 0, 1))):
        coordinate = coords[integrated_axis]
        reference = project_u(truth, coordinate, integrated_axis)
        actual = project_u(truth, coordinate, integrated_axis, selected)
        prediction = project_u(inferred, coordinate, integrated_axis, selected)
        reference_mean = float(np.mean(reference))
        if not np.isfinite(reference_mean) or reference_mean <= 0:
            raise ValueError(f"{plane} color limits require a positive mean ground-truth projection")
        norm_type = LogNorm if log_scale else Normalize
        u_norm = norm_type(vmin=reference_mean / 100, vmax=reference_mean * 10)
        error_norm = norm_type(vmin=1e-2, vmax=10)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            error = np.abs(prediction - actual) / actual
        error = np.where(np.isfinite(error) & (actual > 0), error, np.nan)
        panels = ((actual, "Ground truth u", u_norm), (prediction, "Inferred u", u_norm),
                  (error, "Relative error", error_norm))
        for ax, (values, title, norm) in zip(axes[row], panels):
            if log_scale:
                values = np.where(np.isfinite(values) & (values > 0), values, np.nan)
            mesh = ax.pcolormesh(coords[horizontal_axis], coords[2], values.T,
                                 shading="nearest", cmap="magma", norm=norm)
            ax.set(title=f"{plane}: {title}", xlabel=f"{'xyz'[horizontal_axis]} [kpc]",
                   ylabel="z [kpc]", aspect="equal", xticks=(-10, -5, 0, 5, 10),
                   yticks=(-4, -2, 0, 2, 4))
            if not np.isfinite(values).any():
                ax.text(0.5, 0.5, "No plottable values", ha="center", va="center", transform=ax.transAxes)
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.08)
            colorbar = fig.colorbar(mesh, cax=cax)
            colorbar.set_label("dimensionless" if title == "Relative error" else "u units × kpc")


@temporary_plot_cache()
def plot_comparison(run_directory, z_index=None, log_scale=False, mask=False, dash=False):
    coords, source, mean, std, truth_u, inferred_u = load_results(run_directory)
    x, y, z = coords
    iz = len(z) // 2 if z_index is None else z_index
    if not 0 <= iz < len(z):
        raise ValueError(f"z-index must be between 0 and {len(z) - 1}")
    volume_mask = None if mask is False else load_mask(run_directory, mask, coords)
    outline_mask = None
    if dash:
        outline_mask = volume_mask if volume_mask is not None else load_mask(run_directory, True, coords)
    selected = None if volume_mask is None else volume_mask[:, :, iz]
    output = run_directory
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Saved S_mean/std were computed from transformed posterior source samples.
    # Saved u_mean is E[FNO(S)], not FNO(E[S]). Observations are not used here.
    actual, inferred = truth_u[:, :, iz], inferred_u[:, :, iz]
    # Keep the same scale when toggling the observation mask.
    u_reference_mean = float(np.mean(actual, dtype=np.float64))
    if not np.isfinite(u_reference_mean) or u_reference_mean <= 0:
        raise ValueError("The requested u color limits require a positive ground-truth slice mean; choose another z slice")
    u_limits = dict(vmin=u_reference_mean / 100, vmax=u_reference_mean * 10)
    if selected is not None:
        actual, inferred = (np.where(selected, a, np.nan) for a in (actual, inferred))
    error = inferred - actual
    source_scale = dict(vmin=min(source.min(), mean.min(), std.min()),
                        vmax=max(source.max(), mean.max(), std.max()))
    u_scale = u_limits
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        relative_error = np.abs(error) / actual
    relative_error = np.where(np.isfinite(relative_error) & (actual > 0), relative_error, np.nan)
    panels = [
        (source, "Ground truth S", source_scale),
        (mean, "Inferred S (posterior mean)", source_scale),
        (std, "Uncertainty in S (posterior std)", source_scale),
        (actual, "XY: Ground truth u", u_scale),
        (inferred, "XY: Inferred u", u_scale),
        (relative_error, "XY: Relative error", dict(vmin=1e-2, vmax=10)),
    ]
    if log_scale:
        def positive_values(values):
            return np.where(np.isfinite(values) & (values > 0), values, np.nan)

        def log_options(*arrays):
            valid = np.concatenate([a[np.isfinite(a)] for a in arrays])
            # Empty panels stay NaN. These bounds only provide a valid color bar.
            if not valid.size:
                return dict(norm=LogNorm(vmin=1, vmax=10))
            lo, hi = float(valid.min()), float(valid.max())
            if lo == hi:
                lo, hi = lo / 2, hi * 2
            return dict(norm=LogNorm(vmin=lo, vmax=hi))

        source, mean, std, actual, inferred = map(
            positive_values, (source, mean, std, actual, inferred)
        )
        # Use the original output values in the requested formula, not log(u).
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative_error = np.abs(error) / truth_u[:, :, iz]
        relative_error = positive_values(relative_error)
        source_scale = log_options(source, mean, std)
        u_scale = dict(norm=LogNorm(**u_limits))
        panels = [
            (source, "Ground truth S", source_scale),
            (mean, "Inferred S (posterior mean)", source_scale),
            (std, "Uncertainty in S (posterior std)", source_scale),
            (actual, "XY: Ground truth u", u_scale),
            (inferred, "XY: Inferred u", u_scale),
            (relative_error, "XY: Relative error", dict(norm=LogNorm(vmin=1e-2, vmax=10))),
        ]
    fig, axes = plt.subplots(4, 3, figsize=(22, 20),
                             gridspec_kw={"height_ratios": [1, 1, 0.55, 0.55]})
    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.94, wspace=0.48, hspace=0.35)
    for panel_index, (ax, (values, title, options)) in enumerate(zip(axes[:2].flat, panels)):
        mesh = ax.pcolormesh(x, y, values.T, shading="nearest", **{"cmap": "magma", **options})
        ax.set(title=title, xlabel="x [kpc]", ylabel="y [kpc]", aspect="equal",
               xticks=(-10, -5, 0, 5, 10), yticks=(-10, -5, 0, 5, 10))
        if not np.isfinite(values).any():
            panel_mask = None if panel_index < 3 else selected
            message = "No observed voxels" if panel_mask is not None and not panel_mask.any() else "No positive finite values"
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(mesh, cax=colorbar_ax)
    draw_projection_rows(fig, axes[2:], coords, truth_u, inferred_u, log_scale, volume_mask)
    if outline_mask is not None:
        footprints = ((x, y, outline_mask.any(axis=2)),
                      (x, y, outline_mask[:, :, iz]),
                      (x, z, outline_mask.any(axis=1)),
                      (y, z, outline_mask.any(axis=0)))
        for row, (horizontal, vertical, footprint) in zip(axes, footprints):
            for ax in row:
                draw_mask_box(ax, horizontal, vertical, footprint)
    fig.suptitle(f"{run_directory.name} — u slice at z={z[iz]:.5g} kpc (index {iz})"
                 + (" — logarithmic colors" if log_scale else "")
                 + (" — observation mask applied" if selected is not None else ""),
                 fontsize=FIGURE_TITLE_FONT_SIZE)
    # Color bars are axes too: include their ticks, labels and exponent offsets.
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=TICK_FONT_SIZE)
        ax.title.set_fontsize(TITLE_FONT_SIZE)
        ax.xaxis.label.set_fontsize(LABEL_FONT_SIZE)
        ax.yaxis.label.set_fontsize(LABEL_FONT_SIZE)
        ax.xaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
        ax.yaxis.get_offset_text().set_fontsize(TICK_FONT_SIZE)
        for annotation in ax.texts:
            annotation.set_fontsize(TICK_FONT_SIZE)
    suffix = ("_log" if log_scale else "") + ("_masked" if selected is not None else "")
    suffix += "_dash" if dash else ""
    destination = output / f"ground_truth_vs_inference{suffix}.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directory", type=Path, help="Directory containing truth.npz and posterior.npz")
    parser.add_argument("--z-index", type=int, help="Zero-based z index; default: middle slice")
    parser.add_argument("--log-scale", action="store_true",
                        help="Logarithmic colors on every panel; show relative u error and use NaN for nonpositive values")
    parser.add_argument("--mask", nargs="?", const=True, default=False, metavar="MASK.npy",
                        help="Mask u only using the saved observations.npz mask or an explicit .npy path")
    parser.add_argument("--dash", action="store_true",
                        help="Outline observed-region bounds; uses the saved mask or the custom --mask file")
    args = parser.parse_args()
    try:
        destination = plot_comparison(args.run_directory.resolve(), args.z_index,
                                      log_scale=args.log_scale, mask=args.mask, dash=args.dash)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
