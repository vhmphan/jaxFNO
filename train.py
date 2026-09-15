from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxfno.config import FNOConfig
from jaxfno.data import Dataset, dataset_fingerprint, dataset_from_config, encode_source_features, inspect_npz, validate_coordinate
from jaxfno.fno import infer_batch, make_model, predict
from jaxfno.data import source_means


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


def save_checkpoint(model, path, cfg, dataset, split, *, fingerprint=None, training_progress=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    leaves = jax.tree_util.tree_leaves(model)
    flat = {f"leaf_{i}": np.asarray(leaf) for i, leaf in enumerate(leaves)}
    flat.update(format_version=np.array(3), config=np.array(json.dumps(asdict(cfg))),
                dataset_fingerprint=np.array(fingerprint or dataset_fingerprint(dataset)),
                dataset_metadata=np.array(json.dumps(dataset.metadata)),
                **{a: getattr(dataset, a) for a in "xyz"},
                **{k: np.asarray(v, dtype=np.int64) for k, v in split.items()})
    if dataset.groups is not None:
        flat["groups"] = dataset.groups
    if training_progress is not None:
        flat["training_progress"] = np.array(json.dumps(training_progress))
    # Publish a complete archive atomically so interruption cannot corrupt best.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **flat)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


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
    coordinates: tuple
    splits: dict
    metadata: dict
    fingerprint: str
    groups: np.ndarray | None = None
    inference_padding: tuple[int, int, int] | None = None

    def on_grid(self, x, y, z):
        """Reuse weights on another uniform grid over the same domain.

        Padding is scaled per axis to preserve its physical width, rounded to
        the nearest whole cell (half up). The checkpoint/model is not modified.
        """
        coordinates = tuple(np.asarray(c, dtype=np.float32) for c in (x, y, z))
        padding = []
        old_padding = self.inference_padding or (self.model.padding,) * 3
        for name, old, new, cells in zip("xyz", self.coordinates, coordinates, old_padding):
            validate_coordinate(old, f"training {name}")
            validate_coordinate(new, f"prediction {name}")
            span = abs(float(old[-1]) - float(old[0]))
            # Tolerate only coordinate representation roundoff, not domain shifts.
            tolerance = 8 * np.finfo(np.float32).eps * max(span, abs(float(old[0])), abs(float(old[-1])))
            if not np.allclose(new[[0, -1]], np.asarray(old)[[0, -1]], rtol=0, atol=tolerance):
                raise ValueError(f"Input {name} domain or axis orientation differs from the checkpoint")
            ratio = (len(new) - 1) / (len(old) - 1)
            padding.append(int(np.floor(cells * ratio + 0.5)))
        if all(np.array_equal(a, b) for a, b in zip(self.coordinates, coordinates)):
            return self
        return replace(self, coordinates=coordinates, inference_padding=tuple(padding))

    def predict(self, S):
        """S: (Nx,Ny) or (N,Nx,Ny); return u in saved physical units."""
        return predict(self.model, S, *self.coordinates, self.cfg.batch_size, padding=self.inference_padding)


def load_predictor(path):
    with np.load(path, allow_pickle=False) as data:
        if "format_version" not in data or int(data["format_version"]) != 3:
            raise ValueError("Unsupported legacy checkpoint; retrain with the current code")
        config_values = json.loads(str(data["config"].item()))
        cfg = FNOConfig.from_dict(config_values)
        coords = tuple(data[a].copy() for a in "xyz")
        splits = {k: data[k].copy() for k in ("train_idx", "val_idx", "test_idx")}
        metadata = json.loads(str(data["dataset_metadata"].item()))
        fingerprint = str(data["dataset_fingerprint"].item())
        groups = data["groups"].copy() if "groups" in data else None
    return Predictor(load_checkpoint(make_model(cfg), path), cfg, coords, splits, metadata, fingerprint, groups)


def train_model(cfg: FNOConfig, dataset: Dataset, *, resume: Predictor | None = None):
    fingerprint = dataset_fingerprint(dataset)
    if resume is None:
        train_idx, val_idx, test_idx = split_dataset(len(dataset.S), cfg.validation_fraction,
                                                    cfg.test_fraction, cfg.seed, dataset.groups)
        split = dict(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        model = make_model(cfg)
    else:
        if fingerprint != resume.fingerprint:
            raise ValueError("Continuation requires the same training dataset values, grid, and sample order")
        for field in ("width", "modes", "input_channels", "output_channels", "padding"):
            if getattr(cfg, field) != getattr(resume.cfg, field):
                raise ValueError(f"Cannot change model {field} when continuing a checkpoint")
        if resume.inference_padding is not None:
            raise ValueError("Continue the original checkpoint, not a predictor adapted to another grid")
        if not np.array_equal(dataset.groups, resume.groups):
            raise ValueError("Training groups differ from the checkpoint")
        for key in ("units", "boundary_conditions", "shared_bvp", "homogeneous_boundary_conditions",
                    "kind", "diffusion_coefficients", "lambda"):
            if dataset.metadata.get(key) != resume.metadata.get(key):
                raise ValueError(f"Training metadata {key} differs from the checkpoint")
        split = resume.splits
        indices = np.concatenate(list(split.values()))
        if (any(len(part) == 0 for part in split.values())
                or not np.array_equal(np.sort(indices), np.arange(len(dataset.S)))):
            raise ValueError("Invalid saved train/validation/test split")
        train_idx, val_idx, test_idx = (split[k] for k in ("train_idx", "val_idx", "test_idx"))
        model = resume.model
        print("Continuing saved weights and splits with a fresh AdamW optimizer.", flush=True)
    means = source_means(dataset.S)
    print(f"Normalization: {cfg.normalization}", flush=True)
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
        sources, targets = dataset.S[indices], dataset.u[indices]
        sources = sources / means[indices]
        targets = targets / means[indices][..., None]
        features = encode_source_features(sources, dataset.x, dataset.y, dataset.z)
        return jnp.asarray(features), jnp.asarray(targets)

    def validation_loss(model):
        total = 0.0
        for start in range(0, len(val_idx), cfg.batch_size):
            indices = val_idx[start:start + cfg.batch_size]
            features, targets = batch(indices)
            total += float(relative_l2_loss(infer_batch(model, features), targets, cfg.loss_floor)) * len(indices)
        loss = total / len(val_idx)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        return loss

    best_model, best_val, stale = model, float("inf"), 0
    history = {"train": [], "val": []}
    output = Path(cfg.checkpoint_dir)
    if not output.is_absolute():
        output = Path(__file__).resolve().parent / output
    cfg.checkpoint_dir = str(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is not None:
        best_val = validation_loss(model)
        history["initial_val"] = best_val
        history["optimizer_restarted"] = True
        print(f"Starting checkpoint validation loss: {best_val:.6g}", flush=True)
        # Keep a recoverable starting model even when writing to a new directory.
        if not (output / "best_model.npz").exists():
            save_checkpoint(model, output / "best_model.npz", cfg, dataset, split,
                            fingerprint=fingerprint)
    history_path = output / ("continuation_history.json" if resume is not None else "history.json")
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
        val_loss = validation_loss(model)
        history["train"].append(total / len(train_idx))
        history["val"].append(val_loss)
        improved_for_stopping = val_loss < best_val - cfg.min_delta
        if val_loss < best_val:
            best_val, best_model = val_loss, model
            save_checkpoint(model, output / "best_model.npz", cfg, dataset, split, fingerprint=fingerprint,
                            training_progress={"epoch_in_run": epoch + 1, "validation_loss": val_loss,
                                               "continued_from_checkpoint": resume is not None})
        stale = 0 if improved_for_stopping else stale + 1
        history_path.write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch + 1}/{cfg.epochs}: train={history['train'][-1]:.6g} val={val_loss:.6g}", flush=True)
        if stale >= cfg.patience:
            break
    from jaxfno.plots import plot_history
    plot_history(history, output / ("continuation_loss_curves.png" if resume is not None else "loss_curves.png"))
    return best_model, split


def main():
    parser = argparse.ArgumentParser(description="Supervised 3D FNO training")
    parser.add_argument("--resume", nargs="?", const=str(Path(__file__).resolve().parent / "model" / "best_model.npz"),
                        help="Continue weights from a checkpoint (default model/best_model.npz); restarts AdamW")
    parser.add_argument("--epochs", type=int, help="Epoch limit for this run; additional epochs with --resume")
    parser.add_argument("--config", help="JSON FNOConfig with model settings and dataset layout")
    parser.add_argument("--data", help="Override the training NPZ path (default: uxyz_data.npz)")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.resume and args.epochs is None:
        parser.error("Specify --epochs N for the number of additional epochs")
    try:
        resumed = load_predictor(args.resume) if args.resume else None
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(f"Cannot load continuation checkpoint: {exc}")
    if args.config:
        cfg = FNOConfig.load(args.config)
    elif resumed is not None:
        cfg = replace(resumed.cfg, dataset_layout=dict(resumed.cfg.dataset_layout),
                      checkpoint_dir=str(Path(args.resume).resolve().parent))
    else:
        cfg = FNOConfig(data_path="uxyz_data.npz",
                        dataset_layout={"format": "sol3d", "source_grid": "uniform_domain"})
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.data:
        cfg.data_path = args.data
    if cfg.dataset_layout.get("format") == "sol3d":
        # Reconstruct the uniform source grid from this export's S and domain,
        # including when an older configuration still names a companion file.
        cfg.dataset_layout = {**cfg.dataset_layout, "source_grid": "uniform_domain"}
        cfg.dataset_layout.pop("source_path", None)
    try:
        print(json.dumps(inspect_npz(cfg.data_path), indent=2))
        dataset = dataset_from_config(cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(f"Cannot load training dataset {cfg.data_path}: {exc}")
    print(f"{dataset.metadata.get('kind', 'physical')}: S{dataset.S.shape}, u{dataset.u.shape}")
    if cfg.dataset_layout.get("format") == "sol3d":
        print(dataset.metadata["preprocessing"])
        print(dataset.metadata["units"])
    try:
        train_model(cfg, dataset, resume=resumed)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Saved best checkpoint to {cfg.checkpoint_dir}/best_model.npz")


if __name__ == "__main__":
    main()
