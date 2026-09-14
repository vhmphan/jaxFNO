"""Compare a chosen realization using saved ground truth and FNO predictions."""
import argparse
from pathlib import Path

from jaxfno.prediction_io import load_comparison
from jaxfno.plots import plot_realization


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="uxyz_test.npz")
    parser.add_argument("--predictions", default=str(script_dir / "uxyz_pred.npz"))
    parser.add_argument("--realization", type=int, default=1, help="One-based realization number (default 1)")
    parser.add_argument("--output", default=str(script_dir / "model"),
                        help="Plot directory (default: model/ beside plot_results.py; created if needed)")
    for axis in "xyz":
        parser.add_argument(f"--slice-{axis}", type=float, default=0.0, help="Slice coordinate in kpc (nearest grid node)")
    args = parser.parse_args()
    try:
        dataset, prediction = load_comparison(args.data, args.predictions, args.realization)
        path = plot_realization(dataset, 0, prediction, Path(args.output),
                                slice_coordinates=(args.slice_x, args.slice_y, args.slice_z),
                                display_realization=args.realization)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
