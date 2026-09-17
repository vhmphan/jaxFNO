"""Compare a chosen realization using saved ground truth and FNO predictions."""
import argparse
from pathlib import Path

from jaxfno.prediction_io import load_comparison
from jaxfno.plots import plot_realization, plot_z_profile


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="uxyz_test.npz")
    parser.add_argument("--predictions", default=str(script_dir / "uxyz_pred.npz"))
    parser.add_argument("--realization", type=int, default=1, help="One-based realization number (default 1)")
    parser.add_argument("--relative-error", action="store_true",
                        help="Plot abs(FNO - truth) / truth with zero truth masked; save with _rel suffix")
    parser.add_argument("--z-profile", nargs=2, type=float, metavar=("X", "Y"),
                        help="Also plot u versus z at (X, Y) in kpc, using the nearest grid nodes")
    parser.add_argument("--output", default=str(script_dir / "model"),
                        help="Plot directory (default: model/ beside plot_results.py; created if needed)")
    for axis in "xyz":
        parser.add_argument(f"--slice-{axis}", type=float, default=0.0, help="Slice coordinate in kpc (nearest grid node)")
    args = parser.parse_args()
    try:
        dataset, prediction = load_comparison(args.data, args.predictions, args.realization)
        path = plot_realization(dataset, 0, prediction, Path(args.output),
                                slice_coordinates=(args.slice_x, args.slice_y, args.slice_z),
                                display_realization=args.realization, relative_error=args.relative_error)
        profile_path = None
        if args.z_profile is not None:
            profile_path = plot_z_profile(dataset, 0, prediction, Path(args.output),
                                          x=args.z_profile[0], y=args.z_profile[1],
                                          display_realization=args.realization)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {path}")
    if profile_path is not None:
        print(f"Saved {profile_path}")


if __name__ == "__main__":
    main()
