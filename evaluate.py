"""Predict source realizations and save an NPZ archive; report runtime only."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import jax
import numpy as np

from jaxfno.data import load_dataset
from jaxfno.prediction_io import input_layout, original_sources, save_predictions, source_hashes
from train import load_predictor


def timed_prediction(predictor, sources):
    """Return the timed predictions themselves, after warming each batch shape.

    Predictor.predict returns host NumPy arrays, synchronizing JAX device work.
    Loading, source-grid interpolation, warm-up, and file writing are excluded.
    Feature encoding, normalization, transfers, inference and rescaling are timed.
    """
    count = len(sources)
    if count == 0:
        raise ValueError("No source realizations selected")
    batch_size = predictor.cfg.batch_size
    sizes = {min(batch_size, count)}
    if count % batch_size:
        sizes.add(count % batch_size)
    start = perf_counter()
    for size in sorted(sizes):
        predictor.predict(sources[:size])
    warmup_seconds = perf_counter() - start
    start = perf_counter()
    predictions = predictor.predict(sources)
    seconds = perf_counter() - start
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite predictions")
    runtime = dict(realizations=count, batch_size=batch_size,
                   devices=[str(device) for device in jax.devices()],
                   total_seconds=seconds, seconds_per_realization=seconds / count,
                   realizations_per_second=count / seconds,
                   warmup_seconds_excluded=warmup_seconds,
                   scope="Feature encoding, normalization, device transfers, FNO prediction, and output "
                         "rescaling. Excludes loading, source-grid interpolation, warm-up/JIT, and saving.")
    return predictions, runtime


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(script_dir / "model" / "best_model.npz"))
    parser.add_argument("--data", default="uxyz_test.npz", help="Input source NPZ (u is not read)")
    parser.add_argument("--output", default=str(script_dir / "uxyz_pred.npz"),
                        help="Prediction NPZ destination (default: beside evaluate.py)")
    parser.add_argument("--realization-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Optional one-based inclusive range; default predicts ALL sources")
    args = parser.parse_args()
    try:
        destination = Path(args.output)
        if destination.suffix.lower() != ".npz":
            raise ValueError("--output must end in .npz")
        if destination.resolve() in (Path(args.data).resolve(), Path(args.checkpoint).resolve()):
            raise ValueError("Prediction output must not overwrite the input or checkpoint")
        predictor = load_predictor(args.checkpoint)
        layout = dict(predictor.cfg.dataset_layout)
        if layout.get("format") == "sol3d":
            layout.pop("source_path", None)
            layout["source_grid"] = "uniform_domain"
        dataset = load_dataset(args.data, layout=layout, sources_only=True)
        for name, saved in zip("xyz", predictor.coordinates):
            if not np.array_equal(getattr(dataset, name), saved):
                raise ValueError(f"Input {name} grid differs from the checkpoint")
        for key in ("units", "boundary_conditions", "shared_bvp", "homogeneous_boundary_conditions",
                    "kind", "diffusion_coefficients", "lambda"):
            if dataset.metadata.get(key) != predictor.metadata.get(key):
                raise ValueError(f"Input metadata {key} differs from the checkpoint")
        start, end = args.realization_range or (1, len(dataset.S))
        if not 1 <= start <= end <= len(dataset.S):
            raise ValueError(f"Range must satisfy 1 <= START <= END <= {len(dataset.S)}")
        indices = np.arange(start - 1, end)
        with np.load(args.data, allow_pickle=False) as data:
            hashes = source_hashes(original_sources(data, input_layout(dataset))[indices])
        print(f"Predicting {len(indices)} realizations ({start}–{end}); warming up...", flush=True)
        predictions, runtime = timed_prediction(predictor, dataset.S[indices])
        runtime.update(data_file=args.data, realization_numbers=(indices + 1).tolist())
        save_predictions(destination, predictions, dataset, indices, hashes, runtime, args.checkpoint)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Runtime: {runtime['total_seconds']:.4f} s total; "
          f"{runtime['seconds_per_realization']:.4f} s/realization (warm-up excluded).")
    print(f"Saved {len(indices)} predictions to {destination}")


if __name__ == "__main__":
    main()
