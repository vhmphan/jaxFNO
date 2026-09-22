"""Convert raw source maps into a source-only sol3d archive for evaluate.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxfno.data import validate_coordinate


def create_src_data(source_path, checkpoint_path, output_path, *, source_axes="xy",
                    grid_shape=(257, 257, 129)):
    """Preserve source amplitudes; sample the checkpoint's physical domain.

    Raw generator maps default to (N, Nx, Ny), including square grids, matching
    sol3D's normalization convention. The exported S uses (N, Ny, Nx).
    """
    source_path, checkpoint_path, output_path = map(
        Path, (source_path, checkpoint_path, output_path)
    )
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output must end in .npz")
    if output_path.resolve() in (source_path.resolve(), checkpoint_path.resolve()):
        raise ValueError("Output must not overwrite the sources or checkpoint")
    if source_axes not in ("xy", "yx"):
        raise ValueError("source_axes must be xy or yx")
    if len(grid_shape) != 3 or any(
        not isinstance(n, (int, np.integer)) or n < 3 for n in grid_shape
    ):
        raise ValueError("grid_shape must contain three integers >= 3")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        config = json.loads(str(checkpoint["config"].item()))
        layout = config.get("dataset_layout", {})
        if layout.get("format") != "sol3d" or layout.get("group_key"):
            raise ValueError("Expected a sol3d checkpoint without a group_key")
        coordinates = {a: np.asarray(checkpoint[a], dtype=np.float64) for a in "xyz"}
    with np.load(source_path, allow_pickle=False) as data:
        key = next((k for k in ("realizations", "source", "S") if k in data), None)
        if key is None:
            raise ValueError("Sources must contain realizations, source, or S")
        source = np.asarray(data[key])
        sx, sy = (np.asarray(data[a]) for a in "xy")
    for name, coord in [("source x", sx), ("source y", sy), *coordinates.items()]:
        validate_coordinate(coord, name)
        if np.any(np.diff(coord) <= 0):
            raise ValueError(f"{name} must increase")
    for axis, coord in (("x", sx), ("y", sy)):
        if not np.allclose(coord[[0, -1]], coordinates[axis][[0, -1]], rtol=0, atol=1e-10):
            raise ValueError(f"Source {axis} domain differs from the checkpoint")
    if not np.allclose(coordinates["z"][[0, -1]], [-4, 4], rtol=0, atol=1e-10):
        raise ValueError("sol3d requires z in [-4, 4]")
    if any(len(c) < 3 for c in coordinates.values()):
        raise ValueError("sol3d prediction grids require at least three nodes per axis")
    if source.ndim == 2:
        source = source[None]
    expected = (len(sx), len(sy)) if source_axes == "xy" else (len(sy), len(sx))
    if source.ndim != 3 or not len(source) or source.shape[1:] != expected:
        raise ValueError(f"Expected sources with shape (N, {expected[0]}, {expected[1]})")
    if source.dtype.kind not in "fiu" or not np.isfinite(source).all():
        raise ValueError("Sources must contain finite real numbers")
    if source_axes == "xy":
        source = source.transpose(0, 2, 1)
    coordinates = {
        a: np.linspace(c[0], c[-1], n, dtype=np.float64)
        for (a, c), n in zip(coordinates.items(), grid_shape)
    }
    # evaluate.py trims one ghost coordinate at each end before interpolation.
    ghost_coordinates = {
        a: np.concatenate(([c[0] - (c[1] - c[0])], c, [c[-1] + (c[-1] - c[-2])]))
        for a, c in coordinates.items()
    }
    np.savez_compressed(output_path, S=source, N=len(source), axis_order="zyx",
                        **ghost_coordinates)
    return len(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="sources.npz")
    parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent / "model" / "best_model.npz"),
                        help="Checkpoint providing the physical domain")
    parser.add_argument("--output", default="src_data.npz")
    parser.add_argument("--grid-shape", nargs=3, type=int, default=(257, 257, 129),
                        metavar=("NX", "NY", "NZ"),
                        help="Prediction grid without ghosts (default: 257 257 129)")
    parser.add_argument("--source-axes", choices=("xy", "yx"), default="xy",
                        help="Spatial axis order of raw maps (default: xy, as in the source generator)")
    args = parser.parse_args()
    try:
        count = create_src_data(args.sources, args.checkpoint, args.output,
                                source_axes=args.source_axes, grid_shape=args.grid_shape)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Saved {count} sources to {args.output}")
    print(f"Prediction grid (x, y, z): {tuple(args.grid_shape)}")


if __name__ == "__main__":
    main()
