"""Optional accuracy diagnostics; not run by evaluate.py."""
import numpy as np
from jaxfno.data import dataset_fingerprint
from jaxfno.fno import predict


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


def evaluate_model(model, dataset, cfg, split, *, output_dir=None,
                   realization=None, slice_coordinates=(0.0, 0.0, 0.0), realization_subset=None):
    """Evaluate predictions restored using each source mean."""
    split = np.asarray(split, dtype=int)
    if split.ndim != 1 or len(split) == 0 or np.any(split < 0) or np.any(split >= len(dataset.S)):
        raise ValueError("Invalid or empty test split")
    realization = int(split[0]) if realization is None else realization
    if not 0 <= realization < len(dataset.S):
        raise ValueError(f"realization must be between 0 and {len(dataset.S) - 1}")
    S_test, u_test = dataset.S[split], dataset.u[split]
    predict_fn = lambda S: predict(model, S, dataset.x, dataset.y, dataset.z,
                                    cfg.batch_size)
    pred = predict_fn(S_test)
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite predictions")
    metrics = error_metrics(pred, u_test, S_test, cfg.loss_floor)
    metrics["dataset_kind"] = dataset.metadata.get("kind", "physical")
    metrics["linearity"] = "Skipped: zero-source diagnostic has undefined source-mean normalization."
    return metrics


def select_evaluation_indices(predictor, dataset, *, external=False, realization_range=None):
    """Validate compatibility; enforce original sample identity only for saved splits."""
    for name, saved in zip("xyz", predictor.coordinates):
        if not np.array_equal(getattr(dataset, name), saved):
            raise ValueError(f"Dataset {name} grid differs from the checkpoint")
    for key in ("units", "boundary_conditions", "shared_bvp", "homogeneous_boundary_conditions",
                "kind", "diffusion_coefficients", "lambda"):
        if dataset.metadata.get(key) != predictor.metadata.get(key):
            raise ValueError(f"Dataset metadata {key} differs from the checkpoint")
    if not external:
        if dataset_fingerprint(dataset) != predictor.fingerprint:
            raise ValueError("Dataset values or sample order differ from the checkpoint; use --data for external evaluation")
        all_indices = np.concatenate(list(predictor.splits.values()))
        if not np.array_equal(np.sort(all_indices), np.arange(len(dataset.S))):
            raise ValueError("Dataset sample count does not match saved splits")
        if predictor.groups is not None and not np.array_equal(dataset.groups, predictor.groups):
            raise ValueError("Dataset groups/order differ from the checkpoint")
    if realization_range is not None:
        start, end = realization_range
        if not 1 <= start <= end <= len(dataset.S):
            raise ValueError(f"--realization-range must satisfy 1 <= START <= END <= {len(dataset.S)}")
        return np.arange(start - 1, end)
    return np.arange(len(dataset.S)) if external else predictor.splits["test_idx"]

