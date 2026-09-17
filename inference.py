#!/usr/bin/env python3
# Copyright(C) 2013-2021 Max-Planck-Society
# SPDX-License-Identifier: GPL-2.0+ OR BSD-2-Clause
"""Infer S(x,y)=exp(s) from noisy FNO predictions u(x,y,z) with NIFTy.re.

Run: python -B inference.py
Small integration test: python -B inference.py --smoke-test
Results are saved in results_fno/seed{SEED}_it{iterations}; existing runs are not overwritten.
Requires nifty, jax, equinox, optax, numpy, scipy, matplotlib and the checkpoint.
Reuses jaxfno.inference_tools' differentiable FNO adapter and plotting helpers.
The FNO uses saved solver units, source-mean normalization, XY/XYZ axes,
zero source perimeter for sol3d, and checkpoint-scaled inference padding.
No output boundary clamp is added. Positive source boundary values are constrained
only through the correlated prior. The periodic Gaussian log-field prior and
exact-surrogate assumption are demonstration choices, not calibrated physics.
"""
from pathlib import Path
from time import perf_counter
import argparse
import importlib.metadata
import json

import jax
import nifty.re as jft
import numpy as np
from jax import numpy as jnp
from jax import random

from jaxfno.inference_tools import make_forward, make_observation, check_forward, plot_results

# Settings: keep the existing 33x33 source prior, with 17 output z nodes.
SEED = 42
GRID_SHAPE = (33, 33, 17)  # None uses the checkpoint's full training grid.
CHECKPOINT = Path(__file__).resolve().parent / "model" / "best_model.npz"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_fno"
OBSERVATION_MASK = None  # Optional boolean XYZ .npy mask; None observes all voxels.
NOISE_STD = 0.15  # Absolute standard deviation in saved u units.
VI_ITERATIONS = 6
SAMPLE_PAIRS = 4  # NIFTy returns twice this many (antithetic) posterior samples.

# These describe s=log(S), NOT the mean and std of S. Hyperparameters are inferred.
LOG_OFFSET_MEAN = 2.0
LOG_OFFSET_STD = (0.1, 0.03)
LOG_FLUCTUATIONS = (1.0, 0.5)
LOG_SPECTRAL_SLOPE = (-5.0, 0.2)
LOG_FLEXIBILITY = (1.0, 0.2)
LOG_ASPERITY = (0.5, 0.05)


class Forward(jft.Model):
    """Latent prior variables -> positive XY source -> FNO u -> observations."""

    def __init__(self, log_S, fno, observe):
        self.log_S = log_S
        self.fno = fno
        self.observe = observe
        super().__init__(init=log_S.init)

    def S(self, position):
        return jnp.exp(self.log_S(position))

    def u(self, position):
        return self.fno(self.S(position))

    def __call__(self, position):
        return self.observe(self.u(position))


def make_prior(coords):
    dims = tuple(len(c) for c in coords[:2])
    distances = tuple(float(abs(c[1] - c[0])) for c in coords[:2])
    cfm = jft.CorrelatedFieldMaker("cf")
    cfm.set_amplitude_total_offset(offset_mean=LOG_OFFSET_MEAN, offset_std=LOG_OFFSET_STD)
    cfm.add_fluctuations(
        dims, distances=distances, fluctuations=LOG_FLUCTUATIONS,
        loglogavgslope=LOG_SPECTRAL_SLOPE, flexibility=LOG_FLEXIBILITY,
        asperity=LOG_ASPERITY, prefix="ax1", non_parametric_kind="power",
    )
    return cfm.finalize()


def infer(forward, data, key, iterations, pairs, output, smoke=False):
    """Only observations, noise model and response enter inference, never truth."""
    likelihood = jft.Gaussian(data, noise_cov_inv=lambda v: v / NOISE_STD**2).amend(forward)
    k_i, k_o = random.split(key)
    return jft.optimize_kl(
        likelihood, jft.Vector(likelihood.init(k_i)),
        n_total_iterations=iterations, n_samples=pairs, key=k_o,
        draw_linear_kwargs=dict(cg_name="SL", cg_kwargs=dict(
            absdelta=1e-5 * jft.size(likelihood.domain), maxiter=100)),
        kl_kwargs=dict(minimize_kwargs=dict(
            name="M", xtol=1e-4, cg_kwargs=dict(name="MCG", maxiter=100),
            maxiter=2 if smoke else 35)),
        sample_mode="linear_resample", odir=str(output / "nifty"), resume=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="5x5x5 grid, one VI iteration and one sample pair; not a converged fit")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if not np.isfinite(NOISE_STD) or NOISE_STD <= 0:
        parser.error("NOISE_STD must be finite and positive")
    iterations = 1 if args.smoke_test else VI_ITERATIONS
    pairs = 1 if args.smoke_test else SAMPLE_PAIRS
    output = OUTPUT_DIR / f"seed{SEED}_it{iterations}"
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}. Move the previous run before rerunning.")
    print(f"NIFTy {importlib.metadata.version('nifty')}; output: {output}", flush=True)
    predictor, fno, boundary = make_forward(CHECKPOINT, (5, 5, 5) if args.smoke_test else GRID_SHAPE)
    checks = check_forward(predictor, fno, boundary)
    print("FNO agreement and derivative checks:", checks, flush=True)
    shape = tuple(len(c) for c in predictor.coordinates)
    observe, mask = make_observation(shape, OBSERVATION_MASK)
    forward = Forward(make_prior(predictor.coordinates), fno, observe)
    k_truth, k_noise, k_infer = random.split(random.PRNGKey(SEED), 3)
    truth_position = forward.init(k_truth)
    true_source = np.asarray(forward.S(truth_position))
    true_u = forward.u(truth_position)

    # Fixed known noise std, independent of the actual random noise realization.
    clean_data = observe(true_u)
    data = clean_data + NOISE_STD * random.normal(k_noise, clean_data.shape, dtype=clean_data.dtype)
    coords = dict(zip("xyz", predictor.coordinates))
    np.savez_compressed(output / "truth.npz", S=true_source, u=np.asarray(true_u),
                        S_model_input=true_source * np.asarray(boundary),
                        source_axis_order="xy", axis_order="xyz", **coords)
    observed = np.full(shape, np.nan)
    observed[mask] = np.asarray(data)
    np.savez_compressed(output / "observations.npz", data=np.asarray(data), u=observed,
                        mask=mask, noise_std=NOISE_STD, axis_order="xyz", **coords)

    settings = dict(seed=SEED, grid_shape=shape, checkpoint=str(CHECKPOINT),
                    noise_std=NOISE_STD, iterations=iterations, sample_pairs=pairs,
                    nifty_version=importlib.metadata.version("nifty"), checks=checks,
                    log_offset_mean=LOG_OFFSET_MEAN, log_offset_std=LOG_OFFSET_STD,
                    log_fluctuations=LOG_FLUCTUATIONS, log_spectral_slope=LOG_SPECTRAL_SLOPE,
                    log_flexibility=LOG_FLEXIBILITY, log_asperity=LOG_ASPERITY,
                    checkpoint_metadata=predictor.metadata,
                    inference_padding=predictor.inference_padding)
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    # Finish synthetic-data computation before timing; JAX dispatch is asynchronous.
    jax.block_until_ready(data)
    inference_start = perf_counter()
    samples, state = infer(forward, data, k_infer, iterations, pairs, output, args.smoke_test)
    jax.block_until_ready((samples, state))
    inference_seconds = perf_counter() - inference_start
    runtime = dict(
        inference_seconds=inference_seconds,
        iterations=iterations,
        seconds_per_iteration=inference_seconds / iterations,
        scope="Wall time for infer(), including likelihood setup, initialization, JIT compilation, "
              "MGVI sampling/optimization, and NIFTy's internal logging/checkpoint writes. "
              "Excludes data generation, forward checks, posterior summaries, NPZ saving, and plotting.",
    )
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    print(f"Inference runtime: {inference_seconds:.2f} s ({inference_seconds / 60:.2f} min); "
          f"{runtime['seconds_per_iteration']:.2f} s/iteration on average.", flush=True)

    # Iterate full posterior positions, then transform EACH sample to source units.
    source_samples = np.stack([np.asarray(forward.S(s)) for s in samples])
    predicted = np.mean([np.asarray(forward.u(s)) for s in samples], axis=0)
    if len(source_samples) < 2 or not all(np.isfinite(v).all() for v in (source_samples, predicted)):
        raise RuntimeError("Nonfinite or missing posterior samples")
    mean, std = source_samples.mean(axis=0), source_samples.std(axis=0)
    np.savez_compressed(output / "posterior.npz", S_samples=source_samples, S_mean=mean,
                        S_std=std, u_mean=predicted, source_axis_order="xy", axis_order="xyz", **coords)
    (output / "inference_state.txt").write_text(str(state))
    plot_results(output, predictor.coordinates, true_source, mean, std, observed, predicted, mask)
    print(f"Saved {len(source_samples)} posterior source samples and reconstruction.png to {output}")
    print("Check NIFTy convergence diagnostics before interpreting the reconstruction.")


if __name__ == "__main__":
    main()
