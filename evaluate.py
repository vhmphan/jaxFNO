from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import dataset_fingerprint, dataset_from_config
from fno import predict
from train import load_predictor


def error_metrics(pred, target, sources, floor=1e-8):
    """Relative metrics exclude zero targets, whose absolute errors are separate."""
    diff = np.asarray(pred, dtype=np.float64) - target
    flat = diff.reshape(len(diff), -1)
    target_norm = np.linalg.norm(target.reshape(len(target), -1), axis=1)
    nonzero = target_norm > 0
    rel = np.linalg.norm(flat[nonzero], axis=1) / np.maximum(target_norm[nonzero], floor)
    result = {"sample_count": len(diff), "relative_sample_count": int(nonzero.sum()),
              "mean_rel_l2": float(np.mean(rel)) if rel.size else None,
              "median_rel_l2": float(np.median(rel)) if rel.size else None,
              "worst_rel_l2": float(np.max(rel)) if rel.size else None,
              "global_rmse": float(np.sqrt(np.mean(diff**2)))}
    zero_source = np.all(sources == 0, axis=tuple(range(1, sources.ndim)))
    for name, mask in (("zero_source", zero_source), ("zero_target", ~nonzero)):
        result[name] = {"count": int(mask.sum()),
                        "rmse": float(np.sqrt(np.mean(diff[mask]**2))) if mask.any() else None,
                        "max_abs_error": float(np.abs(diff[mask]).max()) if mask.any() else None}
    return result


def linearity_diagnostics(predict_fn, sources, floor=1e-8):
    a, b = sources[0], sources[-1]
    pa, pb = predict_fn(np.stack([a, b]))
    zero, scaled, summed = predict_fn(np.stack([np.zeros_like(a), 2 * a, a + b]))
    return {"zero_source_rmse": float(np.sqrt(np.mean(zero.astype(np.float64)**2))),
            "amplitude_scaling_rel_l2": float(np.linalg.norm(scaled - 2 * pa) / max(np.linalg.norm(2 * pa), floor)),
            "superposition_rel_l2": float(np.linalg.norm(summed - pa - pb) / max(np.linalg.norm(pa + pb), floor))}


def evaluate_model(model, dataset, cfg, split, *, scales, output_dir=None,
                   realization=None, slice_coordinates=(0.0, 0.0, 0.0), realization_subset=None):
    """Require the fixed training scales; never fit scales to the test set."""
    split = np.asarray(split, dtype=int)
    if split.ndim != 1 or len(split) == 0 or np.any(split < 0) or np.any(split >= len(dataset.S)):
        raise ValueError("Invalid or empty test split")
    realization = int(split[0]) if realization is None else realization
    if not 0 <= realization < len(dataset.S):
        raise ValueError(f"realization must be between 0 and {len(dataset.S) - 1}")
    S_test, u_test = dataset.S[split], dataset.u[split]
    predict_fn = lambda S: predict(model, S, dataset.x, dataset.y, dataset.z,
                                    scales["S_scale"], scales["u_scale"], cfg.batch_size)
    pred = predict_fn(S_test)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite predictions")
    metrics = error_metrics(pred, u_test, S_test, cfg.loss_floor * scales["u_scale"])
    metrics["dataset_kind"] = dataset.metadata.get("kind", "physical")
    if dataset.metadata.get("homogeneous_boundary_conditions") is True:
        metrics["linearity"] = linearity_diagnostics(predict_fn, S_test, cfg.loss_floor * scales["u_scale"])
    else:
        metrics["linearity"] = "Skipped: homogeneous boundary conditions have not been confirmed."
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
        from plots import plot_prediction
        errors = np.linalg.norm((pred - u_test).reshape(len(split), -1), axis=1)
        errors /= np.maximum(np.linalg.norm(u_test.reshape(len(split), -1), axis=1), cfg.loss_floor * scales["u_scale"])
        for local in sorted({0, int(np.argmax(errors))}):
            plot_prediction(dataset, int(split[local]), pred[local], output)
        from plots import plot_realization
        matches = np.flatnonzero(split == realization)
        selected_pred = pred[matches[0]] if len(matches) else predict_fn(dataset.S[realization])
        plot_realization(dataset, realization, selected_pred, output,
                         slice_coordinates=slice_coordinates, subset=realization_subset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved best checkpoint without retraining")
    parser.add_argument("--checkpoint", default="checkpoints/physical/best_model.npz")
    parser.add_argument("--data", help="Optional replacement path to the same dataset with identical sample order")
    parser.add_argument("--output", default=None)
    parser.add_argument("--realization", type=int, help="Zero-based dataset index to plot; defaults to first test sample")
    for axis in "xyz":
        parser.add_argument(f"--slice-{axis}", type=float, default=0.0,
                            help=f"Requested {axis} coordinate for the perpendicular slice (nearest grid node; default 0)")
    args = parser.parse_args()
    predictor = load_predictor(args.checkpoint)
    cfg = predictor.cfg
    if args.data:
        cfg.data_path = args.data
    dataset = dataset_from_config(cfg)
    if args.realization is not None and not 0 <= args.realization < len(dataset.S):
        parser.error(f"--realization must be between 0 and {len(dataset.S) - 1}")
    slices = (args.slice_x, args.slice_y, args.slice_z)
    for axis, value in zip("xyz", slices):
        coord = getattr(dataset, axis)
        if not np.isfinite(value) or not coord.min() <= value <= coord.max():
            parser.error(f"--slice-{axis} must lie in [{coord.min()}, {coord.max()}]")
    if dataset_fingerprint(dataset) != predictor.fingerprint:
        raise ValueError("Dataset values or sample order differ from the checkpoint")
    for name, saved in zip("xyz", predictor.coordinates):
        if not np.array_equal(getattr(dataset, name), saved):
            raise ValueError(f"Dataset {name} grid differs from the checkpoint")
    all_indices = np.concatenate(list(predictor.splits.values()))
    if not np.array_equal(np.sort(all_indices), np.arange(len(dataset.S))):
        raise ValueError("Dataset sample count does not match saved splits")
    if predictor.groups is not None and not np.array_equal(dataset.groups, predictor.groups):
        raise ValueError("Dataset groups/order differ from the checkpoint")
    for key in ("units", "boundary_conditions", "shared_bvp", "homogeneous_boundary_conditions", "kind"):
        if dataset.metadata.get(key) != predictor.metadata.get(key):
            raise ValueError(f"Dataset metadata {key} differs from the checkpoint")
    if cfg.data_path == "synthetic":
        print("SYNTHETIC SMOKE TEST ONLY: these metrics do not measure PDE accuracy.")
    realization = int(predictor.splits["test_idx"][0]) if args.realization is None else args.realization
    subset = next(name.removesuffix("_idx") for name, indices in predictor.splits.items() if realization in indices)
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "evaluation"
    metrics = evaluate_model(predictor.model, dataset, cfg, predictor.splits["test_idx"],
                             scales=predictor.scales,
                             output_dir=output, realization=realization,
                             slice_coordinates=slices, realization_subset=subset)
    print(json.dumps(metrics, indent=2))
    print(f"Saved realization {realization} ({subset}) comparison to {output / f'realization_{realization}_comparison.png'}")


if __name__ == "__main__":
    main()
