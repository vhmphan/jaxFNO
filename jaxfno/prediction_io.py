"""Portable prediction archives and source-pairing checks (NumPy only)."""
import hashlib
import json
from pathlib import Path

import numpy as np

from jaxfno.data import Dataset


def source_hashes(sources):
    """Hash original source samples, including shape and storage dtype."""
    return np.asarray([hashlib.sha256(
        f"{sample.shape}:{sample.dtype}".encode() + np.ascontiguousarray(sample).tobytes()
    ).hexdigest() for sample in sources])


def input_layout(dataset):
    mapping = dataset.metadata.get("layout", {})
    if mapping.get("format") == "sol3d":
        return dict(source_key="S", source_axes="nyx", target_key="u", target_axes="nzyx",
                    x_key="x", y_key="y", z_key="z", coordinate_ghosts=1)
    return {**mapping, "coordinate_ghosts": 0}


def original_sources(data, mapping):
    axes = mapping["source_axes"]
    sources = data[mapping["source_key"]]
    if sources.ndim != 3 or len(axes) != 3 or set(axes) != set("nxy"):
        raise ValueError("Invalid original source axes")
    return np.moveaxis(sources, axes.index("n"), 0)


def save_predictions(path, predictions, dataset, indices, hashes, runtime, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, format_version=1, S=dataset.S[indices], u=predictions,
        x=dataset.x, y=dataset.y, z=dataset.z, N=len(indices), axis_order="xyz",
        source_axis_order="xy", sample_indices=indices, source_hashes=hashes,
        input_layout=json.dumps(input_layout(dataset)),
        dataset_metadata=json.dumps(dataset.metadata), runtime=json.dumps(runtime),
        checkpoint=str(checkpoint),
    )


def load_comparison(truth_path, prediction_path, realization=1):
    """Load one selected pair without running JAX or requiring a checkpoint.

    Realization numbers are one-based indices in the original input file.
    """
    with np.load(prediction_path, allow_pickle=False) as data:
        if int(data["format_version"]) != 1 or str(data["axis_order"].item()) != "xyz":
            raise ValueError("Unsupported prediction archive format")
        indices = data["sample_indices"]
        matches = np.flatnonzero(indices == realization - 1)
        if realization < 1 or len(matches) != 1:
            raise ValueError(f"Realization {realization} is not present in the prediction file")
        local = int(matches[0])
        coords = [data[a] for a in "xyz"]
        source = data["S"][local:local + 1]
        prediction = data["u"][local]
        expected_hash = data["source_hashes"][local]
        mapping = json.loads(str(data["input_layout"].item()))
        metadata = json.loads(str(data["dataset_metadata"].item()))
    with np.load(truth_path, allow_pickle=False) as data:
        sources = original_sources(data, mapping)
        if realization > len(sources):
            raise ValueError("Realization is outside the ground-truth file")
        if source_hashes(sources[realization - 1:realization])[0] != expected_hash:
            raise ValueError("Source values or sample order do not match the prediction file")
        rim = mapping["coordinate_ghosts"]
        for name, saved in zip("xyz", coords):
            coord = data[mapping[f"{name}_key"]]
            if rim:
                coord = coord[rim:-rim]
            if not np.array_equal(np.asarray(coord, dtype=np.float32), saved):
                raise ValueError(f"Ground-truth {name} grid differs from predictions")
        axes = mapping["target_axes"]
        targets = data[mapping["target_key"]]
        if targets.ndim != 4 or len(axes) != 4 or set(axes) != set("nxyz"):
            raise ValueError("Invalid ground-truth target axes")
        if targets.shape[axes.index("n")] != len(sources):
            raise ValueError("Ground-truth source/target sample counts differ")
        target = np.take(targets, realization - 1, axis=axes.index("n"))
        spatial_axes = axes.replace("n", "")
        target = target.transpose(tuple(spatial_axes.index(a) for a in "xyz"))
    dataset = Dataset(source, target[None], *coords, metadata)
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Prediction shape or values are invalid")
    return dataset, prediction
