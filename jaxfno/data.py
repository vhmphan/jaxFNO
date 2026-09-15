from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Dataset:
    S: np.ndarray
    u: np.ndarray | None  # None when loading only sources for prediction
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    metadata: dict[str, Any]
    groups: np.ndarray | None = None

    def __post_init__(self):
        for name in ("S", "u", "x", "y", "z"):
            if name == "u" and self.u is None:
                continue
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain finite real numbers")
            setattr(self, name, value.astype(np.float32))
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} exceeds float32 range")
        for name in ("x", "y", "z"):
            validate_coordinate(getattr(self, name), name)
        expected = (len(self.S), len(self.x), len(self.y))
        if self.S.shape != expected or (self.u is not None and self.u.shape != (*expected, len(self.z))) or not len(self.S):
            raise ValueError(f"Expected paired S{expected}, u{(*expected, len(self.z))}; "
                             f"got {self.S.shape}, {None if self.u is None else self.u.shape}")
        if self.groups is not None:
            self.groups = np.asarray(self.groups)
            if self.groups.shape != (len(self.S),):
                raise ValueError("groups must have one identifier per sample")
            if self.groups.dtype.kind in "fc" and not np.isfinite(self.groups).all():
                raise ValueError("groups must contain finite identifiers")


def dataset_fingerprint(dataset):
    """Identify paired sample values and order without copying the full dataset."""
    digest = hashlib.sha256()
    for name in ("S", "u", "x", "y", "z"):
        arr = getattr(dataset, name)
        digest.update(f"{name}:{arr.shape}".encode())
        for chunk in arr:
            digest.update(np.asarray(chunk, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def validate_coordinate(coord, name):
    coord = np.asarray(coord)
    if coord.ndim != 1 or len(coord) < 2 or not np.isfinite(coord).all():
        raise ValueError(f"{name} must be a finite 1D coordinate vector with >= 2 points")
    steps = np.diff(coord.astype(np.float64))
    if not ((steps > 0).all() or (steps < 0).all()):
        raise ValueError(f"{name} must be strictly monotonic")
    if not np.allclose(steps, steps[0], rtol=1e-4, atol=abs(steps[0]) * 1e-5):
        raise ValueError(f"{name} must be uniformly spaced; interpolation is not implicit")


def inspect_npz(path: str | Path) -> dict[str, Any]:
    """Inspect without enabling pickle, including nonnumeric metadata safely."""
    summary = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            try:
                arr = data[key]
            except ValueError as exc:
                summary[key] = {"unreadable": str(exc)}
                continue
            item = {"shape": arr.shape, "dtype": str(arr.dtype)}
            if arr.dtype.kind in "fiu" and arr.size:
                item.update(finite=bool(np.isfinite(arr).all()),
                            min=float(np.min(arr)), max=float(np.max(arr)))
                if arr.ndim == 1:
                    item.update(first=float(arr[0]), last=float(arr[-1]))
            elif arr.dtype.kind in "US" and arr.size <= 1:
                item["value"] = str(arr.item())
            summary[key] = item
    return summary


def source_means(S):
    """Per-realization mean on the loaded XY grid, including boundary nodes."""
    S = np.asarray(S, dtype=np.float32)
    if S.ndim != 3 or not np.isfinite(S).all():
        raise ValueError("Expected finite sources with shape (N, Nx, Ny)")
    means = np.mean(S, axis=(1, 2), keepdims=True)
    if not np.isfinite(means).all() or np.any(means == 0):
        raise ValueError("Source-mean normalization requires finite, nonzero source means")
    return means


def encode_source_features(S, x, y, z):
    """(N,Nx,Ny) -> (N,4,Nx,Ny,Nz); z broadcast is encoding, not a volume source."""
    S = np.asarray(S, dtype=np.float32)
    if S.ndim == 2:
        S = S[None]
    coords = [np.asarray(c, dtype=np.float32) for c in (x, y, z)]
    for name, coord in zip(("x", "y", "z"), coords):
        validate_coordinate(coord, name)
    if S.ndim != 3 or S.shape[1:] != (len(x), len(y)) or not np.isfinite(S).all():
        raise ValueError("Source shape or values do not match the coordinate grid")
    shape = (len(S), len(x), len(y), len(z))
    source = np.broadcast_to(S[..., None], shape)
    grid = np.meshgrid(*[(c - c.mean()) / np.ptp(c) for c in coords], indexing="ij")
    return np.stack([source, *[np.broadcast_to(c, shape) for c in grid]], axis=1).astype(np.float32)


def _load_npz_dataset(path, layout=None, *, sources_only=False):
    """Require an explicit key/axis mapping; equal grid sizes cannot identify axes."""
    if layout and layout.get("format") == "sol3d":
        return _load_sol3d_dataset(path, layout, sources_only=sources_only)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item())) if "metadata" in data else {}
        mapping = dict(metadata.get("layout", {}))
        mapping.update(layout or {})
        required = ("source_key", "target_key", "x_key", "y_key", "z_key", "source_axes", "target_axes")
        if any(k not in mapping for k in required):
            raise ValueError("Inspect the NPZ first and specify dataset_layout: " + ", ".join(required)
                             + ". Missing sources require their file or exact generation parameters and pairing.")
        metadata.update(mapping.get("metadata", {}))
        if not metadata.get("units") or not metadata.get("boundary_conditions") or metadata.get("shared_bvp") is not True:
            raise ValueError("Specify metadata units, boundary_conditions, and shared_bvp=true after verifying "
                             "all samples use the same boundary-value problem")
        def ordered(key, axes, desired):
            arr = np.asarray(data[key])
            if len(axes) != len(desired) or set(axes) != set(desired) or arr.ndim != len(desired):
                raise ValueError(f"Invalid axis order {axes!r} for {key}; expected a permutation of {desired}")
            return arr.transpose(tuple(axes.index(a) for a in desired))
        S = ordered(mapping["source_key"], mapping["source_axes"], "nxy")
        u = None if sources_only else ordered(mapping["target_key"], mapping["target_axes"], "nxyz")
        coords = [np.asarray(data[mapping[f"{axis}_key"]]) for axis in "xyz"]
        groups = np.asarray(data[mapping["group_key"]]) if mapping.get("group_key") else None
        metadata.update(file=str(path), layout=mapping,
                        domain={a: [float(c[0]), float(c[-1])] for a, c in zip("xyz", coords) if c.ndim == 1 and c.size})
        return Dataset(S, u, *coords, metadata, groups)


def _load_sol3d_dataset(path, layout, *, sources_only=False):
    """Adapt the inspected sol3D.py export, reproducing its source interpolation.

    S is already normalized to nyx by sol3D.load_source; do not transpose it
    a second time before RegularGridInterpolator((source_y, source_x), S).
    Only coordinates contain ghost cells. The saved u has already been cropped.
    """
    from scipy.interpolate import RegularGridInterpolator

    path = Path(path)
    source_path = Path(layout.get("source_path", "sources.npz"))
    if not source_path.is_absolute():
        source_path = path.parent / source_path
    with np.load(path, allow_pickle=False) as data:
        if str(data["axis_order"].item()) != "zyx":
            raise ValueError("sol3d expects axis_order='zyx'")
        coords = [np.asarray(data[a], dtype=np.float64) for a in "xyz"]
        source = np.asarray(data["S"], dtype=np.float64)
        target = None if sources_only else np.asarray(data["u"], dtype=np.float64)
        if source.ndim != 3 or (target is not None and target.ndim != 4) or int(data["N"]) != len(source):
            raise ValueError("Invalid sol3d sample count or array ranks")
        if not np.isfinite(source).all() or (target is not None and not np.isfinite(target).all()):
            raise ValueError("sol3d source and target must be finite")
    for a, c in zip("xyz", coords):
        validate_coordinate(c, a)
        if len(c) < 5 or np.any(np.diff(c) <= 0):
            raise ValueError("sol3d requires increasing coordinate vectors with ghost cells")
    x, y, z = [c[1:-1] for c in coords]
    if target is not None and target.shape != (len(source), len(z), len(y), len(x)):
        raise ValueError("sol3d target shape does not match coordinates with one ghost cell per end")

    if layout.get("source_grid") == "uniform_domain":
        # The inspected source generator samples a uniform grid over the same
        # physical domain. sol3D.py stores S in (N, Ny_source, Nx_source) order.
        # Solver coordinates include one ghost node at each end, removed above.
        sx = np.linspace(x[0], x[-1], source.shape[2], dtype=np.float64)
        sy = np.linspace(y[0], y[-1], source.shape[1], dtype=np.float64)
        validate_coordinate(sx, "source x")
        validate_coordinate(sy, "source y")
    else:
        # sol3D.py omits source coordinates, so recover them from its paired input.
        with np.load(source_path, allow_pickle=False) as data:
            sx, sy = [np.asarray(data[a], dtype=np.float64) for a in "xy"]
            key = next((k for k in ("realizations", "source", "S") if k in data), None)
            if key is None:
                raise ValueError("Source file must contain realizations, source, or S")
            original = np.asarray(data[key], dtype=np.float64)
        for a, c in (("source x", sx), ("source y", sy)):
            validate_coordinate(c, a)
            if np.any(np.diff(c) <= 0):
                raise ValueError("sol3d source coordinates must increase")
        if original.ndim == 2:
            original = original[None]
        # Mirror normalize_realizations exactly, including its square-grid branch.
        if original.ndim == 3 and original.shape[-2:] == (len(sx), len(sy)):
            original = original.transpose(0, 2, 1)
        if original.shape != (len(source), len(sy), len(sx)) or not np.array_equal(original, source):
            raise ValueError("Source file does not match sol3d S values and sample order")
    if not (np.allclose(x[[0, -1]], sx[[0, -1]], rtol=0, atol=1e-10)
            and np.allclose(y[[0, -1]], sy[[0, -1]], rtol=0, atol=1e-10)):
        raise ValueError("Source and solver domain endpoints differ")
    if not np.allclose(z[[0, -1]], [-4, 4], rtol=0, atol=1e-10):
        raise ValueError("sol3D.py uses z in [-4,4]; this export has a different domain")
    for axis in (1, 2, 3):
        if target is not None and np.any(np.take(target, [0, -1], axis=axis) != 0):
            raise ValueError("sol3d targets must satisfy zero Dirichlet values on all faces")

    # Saved x[1:-1] is full grid_x.centers[2:-2], exactly as in the RHS.
    # Keep the unused boundary source values zero, matching rhs initialization.
    xp, yp = np.meshgrid(x[1:-1], y[1:-1], indexing="xy")
    points = np.stack((yp, xp), axis=-1)
    S = np.zeros((len(source), len(x), len(y)), dtype=np.float32)
    for i, surface_yx in enumerate(source):
        interpolator = RegularGridInterpolator((sy, sx), surface_yx, method="linear", bounds_error=True)
        S[i, 1:-1, 1:-1] = interpolator(points).T
    metadata = {
        "kind": "physical", "file": str(path),
        "source_file": None if layout.get("source_grid") == "uniform_domain" else str(source_path),
        "source_coordinate_origin": ("uniform source grid reconstructed from S.shape and physical domain"
                                     if layout.get("source_grid") == "uniform_domain" else "paired source file"),
        "layout": dict(layout), "generator": "sol3D.py", "shared_bvp": True,
        "units": "solver units; physical units not declared by generator",
        "boundary_conditions": "u=0 on all six Cartesian domain faces",
        "homogeneous_boundary_conditions": True,
        "domain": {a: [float(c[0]), float(c[-1])] for a, c in zip("xyz", (x, y, z))},
        "diffusion_coefficients": [1.0, 1.0, 1.0], "lambda": 0.0,
        "preprocessing": "Trim coordinate ghosts; transpose u nzyx->nxyz; linear source interpolation "
                         "on interior xy nodes with zero perimeter, as in sol3D.py RHS. "
                         "S is the surface amplitude, not -S/dz.",
    }
    groups = None
    if layout.get("group_key"):
        with np.load(path, allow_pickle=False) as data:
            groups = np.asarray(data[layout["group_key"]])
    return Dataset(S, None if target is None else target.transpose(0, 3, 2, 1), x, y, z, metadata, groups)

def load_dataset(path: str | Path, *, layout=None, sources_only=False) -> Dataset:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}.")
    if dataset_path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported data format: {dataset_path.suffix}")
    return _load_npz_dataset(dataset_path, layout, sources_only=sources_only)


def dataset_from_config(cfg):
    return load_dataset(cfg.data_path, layout=cfg.dataset_layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect NPZ keys and grid candidates before configuring the loader")
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(inspect_npz(args.path), indent=2))
