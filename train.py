from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import FNOConfig
from data import Dataset, dataset_fingerprint, dataset_from_config, encode_source_features, inspect_npz
from fno import infer_batch, make_model, predict


def split_dataset(n, val_frac=0.1, test_frac=0.1, seed=0, groups=None):
    """Seeded sample split, or whole-group split for related source variants.

    Fractions apply to groups when supplied; sample fractions can then differ.
    """
    if n < 3 or min(val_frac, test_frac) <= 0 or val_frac + test_frac >= 1:
        raise ValueError("Need >= 3 samples and positive validation/test fractions summing to < 1")
    if groups is None:
        groups = np.arange(n)
    groups = np.asarray(groups)
    if groups.shape != (n,):
        raise ValueError("Expected one group identifier per sample")
    labels = np.unique(groups)
    if len(labels) < 3:
        raise ValueError("Need at least three independent groups for nonempty splits")
    labels = np.random.default_rng(seed).permutation(labels)
    nv = max(1, round(len(labels) * val_frac))
    nt = max(1, round(len(labels) * test_frac))
    if nv + nt >= len(labels):
        raise ValueError("Fractions leave no training groups; adjust fractions or add groups")
    return tuple(np.flatnonzero(np.isin(groups, part)) for part in
                 (labels[nv + nt:], labels[:nv], labels[nv:nv + nt]))


def relative_l2_loss(pred, target, eps=1e-8):
    diff = (pred - target).reshape(pred.shape[0], -1)
    squared = jnp.sum(diff**2, axis=1)
    # This branch gives finite (zero) gradients at an exact fit.
    num = jnp.where(squared > 0, jnp.sqrt(jnp.maximum(squared, 1e-30)), 0.0)
    den = jnp.sqrt(jnp.sum(target.reshape(target.shape[0], -1)**2, axis=1))
    return jnp.mean(num / jnp.maximum(den, eps))


def save_checkpoint(model, path, cfg, scales, dataset, split, *, fingerprint=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree_util.tree_leaves(model)
    flat = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    flat.update(format_version=np.array(2), config=np.array(json.dumps(asdict(cfg))),
                dataset_fingerprint=np.array(fingerprint or dataset_fingerprint(dataset)),
                dataset_metadata=np.array(json.dumps(dataset.metadata)),
                **{k: np.asarray(v) for k, v in scales.items()},
                **{a: getattr(dataset, a) for a in "xyz"},
                **{k: np.asarray(v, dtype=np.int64) for k, v in split.items()})
    if dataset.groups is not None:
        flat["groups"] = dataset.groups
    np.savez(path, **flat)


def load_checkpoint(model_template, path):
    """Restore leaves by template count and validate shapes; never count metadata keys."""
    template_leaves, treedef = jax.tree_util.tree_flatten(model_template)
    with np.load(path, allow_pickle=False) as data:
        names = {k for k in data.files if k.startswith("leaf_")}
        if names != {f"leaf_{i}" for i in range(len(template_leaves))}:
            raise ValueError("Checkpoint leaves do not match the model; retrain legacy checkpoints")
        leaves = []
        for i, template in enumerate(template_leaves):
            arr = data[f"leaf_{i}"]
            if arr.shape != template.shape or not np.isfinite(arr).all():
                raise ValueError(f"Invalid checkpoint leaf_{i}: shape or finite-value mismatch")
            leaves.append(jnp.asarray(arr, dtype=template.dtype))
    return jax.tree_util.tree_unflatten(treedef, leaves)


@dataclass
class Predictor:
    model: object
    cfg: FNOConfig
    scales: dict
    coordinates: tuple
    splits: dict
    metadata: dict
    fingerprint: str
    groups: np.ndarray | None = None

    def predict(self, S):
        """S: (Nx,Ny) or (N,Nx,Ny); return u in saved physical units."""
        return predict(self.model, S, *self.coordinates, self.scales["S_scale"],
                       self.scales["u_scale"], self.cfg.batch_size)


def load_predictor(path):
    with np.load(path, allow_pickle=False) as data:
        if "format_version" not in data or int(data["format_version"]) != 2:
            raise ValueError("Unsupported legacy checkpoint; retrain with the current code")
        cfg = FNOConfig(**json.loads(str(data["config"].item())))
        coords = tuple(data[a].copy() for a in "xyz")
        scales = {k: float(data[k]) for k in ("S_scale", "u_scale")}
        splits = {k: data[k].copy() for k in ("train_idx", "val_idx", "test_idx")}
        metadata = json.loads(str(data["dataset_metadata"].item()))
        fingerprint = str(data["dataset_fingerprint"].item())
        groups = data["groups"].copy() if "groups" in data else None
    return Predictor(load_checkpoint(make_model(cfg), path), cfg, scales, coords, splits, metadata, fingerprint, groups)


def train_model(cfg: FNOConfig, dataset: Dataset):
    train_idx, val_idx, test_idx = split_dataset(len(dataset.S), cfg.validation_fraction,
                                                cfg.test_fraction, cfg.seed, dataset.groups)
    split = dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    fingerprint = dataset_fingerprint(dataset)
    # Global training-only max-absolute scales; retain small physical amplitudes.
    scales = {"S_scale": float(np.max(np.abs(dataset.S[train_idx]))),
              "u_scale": float(np.max(np.abs(dataset.u[train_idx])))}
    scales = {k: v if v > 0 else 1.0 for k, v in scales.items()}
    model = make_model(cfg)
    optimizer = optax.adamw(cfg.learning_rate, weight_decay=cfg.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def loss_fn(model, features, targets):
        return relative_l2_loss(jax.vmap(model)(features)[:, 0], targets, cfg.loss_floor)

    @eqx.filter_jit
    def step(model, features, targets, state):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, features, targets)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)]))
        updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), state, loss, finite

    def batch(indices):
        features = encode_source_features(dataset.S[indices], dataset.x, dataset.y, dataset.z, scales["S_scale"])
        return jnp.asarray(features), jnp.asarray(dataset.u[indices] / scales["u_scale"])

    best_model, best_val, stale = model, float("inf"), 0
    history = {"train": [], "val": []}
    output = Path(cfg.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        total = 0.0
        order = rng.permutation(train_idx)
        for start in range(0, len(order), cfg.batch_size):
            indices = order[start:start + cfg.batch_size]
            features, targets = batch(indices)
            model, opt_state, loss, finite = step(model, features, targets, opt_state)
            if not bool(finite) or not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite loss or gradients at epoch {epoch + 1}")
            total += float(loss) * len(indices)
        val_total = 0.0
        for start in range(0, len(val_idx), cfg.batch_size):
            indices = val_idx[start:start + cfg.batch_size]
            features, targets = batch(indices)
            val_total += float(relative_l2_loss(infer_batch(model, features), targets, cfg.loss_floor)) * len(indices)
        val_loss = val_total / len(val_idx)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        history["train"].append(total / len(train_idx))
        history["val"].append(val_loss)
        improved_for_stopping = val_loss < best_val - cfg.min_delta
        if val_loss < best_val:
            best_val, best_model = val_loss, model
            save_checkpoint(model, output / "best_model.npz", cfg, scales, dataset, split, fingerprint=fingerprint)
        stale = 0 if improved_for_stopping else stale + 1
        print(f"epoch {epoch + 1}: train={history['train'][-1]:.6g} val={val_loss:.6g}", flush=True)
        if stale >= cfg.patience:
            break
    (output / "history.json").write_text(json.dumps(history, indent=2))
    from plots import plot_history
    plot_history(history, output / "loss_curves.png")
    return best_model, {**scales, **split}


def main():
    parser = argparse.ArgumentParser(description="Supervised 3D FNO training")
    parser.add_argument("--config", help="JSON FNOConfig with model settings and dataset layout")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", help="Override the training NPZ path (default: data_uxyz.npz)")
    inputs.add_argument("--synthetic", action="store_true", help="Run the explicit synthetic smoke test")
    args = parser.parse_args()
    if args.config:
        cfg = FNOConfig.load(args.config)
    elif args.synthetic:
        cfg = FNOConfig(data_path="synthetic", width=8, modes=(4, 4, 3), padding=2,
                        epochs=8, batch_size=4, synthetic_grid=(8, 8, 6),
                        checkpoint_dir="checkpoints/smoke")
    else:
        cfg = FNOConfig(data_path="data_uxyz.npz", checkpoint_dir="checkpoints/physical",
                        dataset_layout={"format": "sol3d", "source_path": "sources.npz"})
    if args.data:
        cfg.data_path = args.data
    elif args.synthetic:
        cfg.data_path = "synthetic"
    try:
        if cfg.data_path != "synthetic":
            print(json.dumps(inspect_npz(cfg.data_path), indent=2))
        dataset = dataset_from_config(cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(f"Cannot load training dataset {cfg.data_path}: {exc}")
    print(f"{dataset.metadata.get('kind', 'physical')}: S{dataset.S.shape}, u{dataset.u.shape}")
    if cfg.data_path == "synthetic":
        print("SYNTHETIC SMOKE TEST ONLY: no PDE or physical accuracy validation.")
    elif cfg.dataset_layout.get("format") == "sol3d":
        print(dataset.metadata["preprocessing"])
        print(dataset.metadata["units"])
    train_model(cfg, dataset)
    print(f"Saved best checkpoint to {cfg.checkpoint_dir}/best_model.npz")


if __name__ == "__main__":
    main()
