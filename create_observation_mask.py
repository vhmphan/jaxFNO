"""Create a boolean XYZ observation mask for inference.py.

Edit the six bounds below, then run:
    python -B create_observation_mask.py

In inference.py, set OBSERVATION_MASK to the printed .npy path before running
inference. True means observed; False means excluded from the likelihood.
The observed region is the intersection of all three inclusive intervals.
For inference.py --smoke-test, generate the mask with --smoke-test too.
Regenerate the mask whenever the inference grid or checkpoint changes.
"""
from pathlib import Path
import argparse

import numpy as np


# Observed region bounds in kpc. Use None for a domain edge (no restriction).
x_min = -6.0
x_max = 6.0
y_min = -6.0
y_max = 6.0
z_min = -2.0
z_max = 2.0

OUTPUT_PATH = Path(__file__).resolve().parent / "observation_mask.npy"


def make_mask(coords):
    """Include a voxel only when x, y AND z fall inside the configured bounds."""
    selected = []
    for axis, coordinate, lower, upper in zip(
        "xyz", coords, (x_min, y_min, z_min), (x_max, y_max, z_max)
    ):
        lower = float(np.min(coordinate)) if lower is None else lower
        upper = float(np.max(coordinate)) if upper is None else upper
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"Invalid {axis} bounds: min and max must be finite and min <= max")
        selected.append((coordinate >= lower) & (coordinate <= upper))
    mask = selected[0][:, None, None] & selected[1][None, :, None] & selected[2][None, None, :]
    if not mask.any():
        raise ValueError("The observed region contains no grid nodes; adjust the bounds")
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true", help="Match inference's 5x5x5 smoke-test grid")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination .npy file (replaced if present)")
    args = parser.parse_args()
    # Import settings only; inference.main() is not executed and no FNO is loaded.
    from inference import CHECKPOINT, GRID_SHAPE

    try:
        if args.output.suffix.lower() != ".npy":
            raise ValueError("--output must end in .npy")
        with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
            coords = tuple(np.asarray(checkpoint[a], dtype=np.float32) for a in "xyz")
        shape = (5, 5, 5) if args.smoke_test else GRID_SHAPE
        if shape is not None:
            coords = tuple(np.linspace(c[0], c[-1], n, dtype=np.float32) for c, n in zip(coords, shape))
        mask = make_mask(coords)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, mask, allow_pickle=False)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.output.resolve()}")
    print(f"XYZ shape: {mask.shape}; observed: {mask.sum()}/{mask.size} voxels ({mask.mean():.1%})")
    print(f"Set OBSERVATION_MASK = {str(args.output.resolve())!r} in inference.py")


if __name__ == "__main__":
    main()
