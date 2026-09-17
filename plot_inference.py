"""Plot saved ground truth and inference results; requires NumPy and Matplotlib.

Run: python -B plot_inference.py results_fno/seed42_o656l61k
Choose a zero-based z slice: python -B plot_inference.py RUN_DIRECTORY --z-index 2
Logarithmic colors: python -B plot_inference.py RUN_DIRECTORY --log-scale
Reads truth.npz and posterior.npz from nifty_example.py. No model is loaded and
no inference is run. The figure is saved directly in RUN_DIRECTORY as
ground_truth_vs_inference.png (or ground_truth_vs_inference_log.png in log mode).
Plotting again replaces that image. Log mode uses relative error for u and
sets nonpositive/nonfinite plotted values to NaN, without clipping or a floor.
"""
from pathlib import Path
import argparse

import numpy as np
from jaxfno._plot_cache import temporary_plot_cache


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


@temporary_plot_cache()
def plot_comparison(run_directory, z_index=None, log_scale=False):
    coords, source, mean, std, truth_u, inferred_u = load_results(run_directory)
    x, y, z = coords
    iz = len(z) // 2 if z_index is None else z_index
    if not 0 <= iz < len(z):
        raise ValueError(f"z-index must be between 0 and {len(z) - 1}")
    output = run_directory
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Saved S_mean/std were computed from transformed posterior source samples.
    # Saved u_mean is E[FNO(S)], not FNO(E[S]). Observations are not used here.
    actual, inferred = truth_u[:, :, iz], inferred_u[:, :, iz]
    error = inferred - actual
    source_scale = dict(vmin=min(source.min(), mean.min()), vmax=max(source.max(), mean.max()))
    u_scale = dict(vmin=min(actual.min(), inferred.min()), vmax=max(actual.max(), inferred.max()))
    error_limit = max(float(np.max(np.abs(error))), np.finfo(float).eps)
    panels = [
        (source, "Ground truth S", source_scale),
        (mean, "Inferred S (posterior mean)", source_scale),
        (std, "Uncertainty in S (posterior std)", dict(vmin=0)),
        (actual, "Ground truth u", u_scale),
        (inferred, "Inferred u (posterior mean)", u_scale),
        (error, "Inferred u − ground truth u", dict(cmap="coolwarm", vmin=-error_limit, vmax=error_limit)),
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
        source_scale, u_scale = log_options(source, mean), log_options(actual, inferred)
        panels = [
            (source, "Ground truth S", source_scale),
            (mean, "Inferred S (posterior mean)", source_scale),
            (std, "Uncertainty in S (posterior std)", log_options(std)),
            (actual, "Ground truth u", u_scale),
            (inferred, "Inferred u (posterior mean)", u_scale),
            (relative_error, "Relative error |inferred u − truth u| / truth u", log_options(relative_error)),
        ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (values, title, options) in zip(axes.flat, panels):
        mesh = ax.pcolormesh(x, y, values.T, shading="nearest", **{"cmap": "magma", **options})
        ax.set(title=title, xlabel="x [kpc]", ylabel="y [kpc]", aspect="equal",
               xticks=(-10, -5, 0, 5, 10), yticks=(-10, -5, 0, 5, 10))
        if log_scale and not np.isfinite(values).any():
            ax.text(0.5, 0.5, "No positive finite values", ha="center", va="center", transform=ax.transAxes)
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(mesh, cax=colorbar_ax)
    fig.suptitle(f"{run_directory.name} — u slice at z={z[iz]:.5g} kpc (index {iz})"
                 + (" — logarithmic colors" if log_scale else ""))
    destination = output / ("ground_truth_vs_inference_log.png" if log_scale else "ground_truth_vs_inference.png")
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_directory", type=Path, help="Directory containing truth.npz and posterior.npz")
    parser.add_argument("--z-index", type=int, help="Zero-based z index; default: middle slice")
    parser.add_argument("--log-scale", action="store_true",
                        help="Logarithmic colors on every panel; show relative u error and use NaN for nonpositive values")
    args = parser.parse_args()
    try:
        destination = plot_comparison(args.run_directory.resolve(), args.z_index, log_scale=args.log_scale)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
