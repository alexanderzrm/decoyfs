"""Decoy (control) features: Random probes and Shadow copies.

* **Random probes** are pure-noise columns drawn from seven distribution families.
* **Shadow features** are copies of every real column whose rows are permuted independently: they keep the marginal
  distribution (scale, discreteness, cardinality) and destroy any relation with ``y`` and with the other features.
"""
from __future__ import annotations

import numpy as np


def robust_like(R, clip=10.0):
    """Apply to the decoys the same robust scaling and clipping as to the real features."""
    med = np.median(R, axis=0)
    q1, q3 = np.percentile(R, [25, 75], axis=0)
    iqr = q3 - q1
    sc = np.where(iqr > 0, iqr, np.where(R.std(0) > 0, R.std(0), 1.0))
    return np.clip((R - med) / sc, -clip, clip).astype("float32")


def random_probes(n, n_probes, rng):
    """Seven families: uniform, normal, lognormal, exponential, sparse beta, Bernoulli, Poisson."""
    makers = [
        lambda: rng.random(n),
        lambda: rng.normal(size=n),
        lambda: rng.lognormal(0, 1, n),
        lambda: rng.exponential(1, n),
        lambda: rng.beta(0.3, 3.0, n) * (rng.random(n) > 0.4),
        lambda: (rng.random(n) < rng.uniform(0.05, 0.5)).astype(float),
        lambda: rng.poisson(rng.uniform(0.3, 4.0), n).astype(float),
    ]
    cols = [makers[j % len(makers)]() for j in range(n_probes)]
    return robust_like(np.column_stack(cols))


def shadow_features(Z, reps, rng):
    """``reps`` copies of every column with rows permuted independently per column."""
    n, p = Z.shape
    out = []
    for _ in range(reps):
        order = np.argsort(rng.random((n, p)), axis=0)
        out.append(np.take_along_axis(Z, order, axis=0))
    return np.concatenate(out, axis=1)


def default_shadow_reps(p):
    return int(min(10, max(1, np.ceil(40 / p))))


def augment(Z, rng, n_random=None, shadow_reps=None):
    """Return ``[real | random | shadow]`` and the column indices of each block."""
    n, p = Z.shape
    n_random = n_random or max(14, int(np.ceil(0.05 * p)))
    n_random = int(np.ceil(n_random / 7) * 7)
    shadow_reps = shadow_reps or default_shadow_reps(p)
    R = random_probes(n, n_random, rng)
    S = shadow_features(Z, shadow_reps, rng)
    Xa = np.concatenate([Z, R, S], axis=1)
    idx = dict(real=np.arange(p), rand=np.arange(p, p + n_random),
               shad=np.arange(p + n_random, p + n_random + S.shape[1]))
    return np.ascontiguousarray(Xa), idx


VALID_CONTROLS = ("random", "shadow", "block")


def build_controls(Z, rng, controls=("random", "shadow"), n_random=None, shadow_reps=None, block_thr=0.5, seed=0):
    """Augment ``Z`` with the requested decoy blocks. Returns ``(matrix, idx)``.

    ``controls`` is a subset of ``("random", "shadow", "block")``; ``shadow`` and ``block`` are mutually exclusive
    (Block-Shadow replaces the independent Shadow). Keys of ``idx``: ``real``, ``rand`` and ``shad`` (the Shadow or the
    Block-Shadow columns), present only for the requested blocks. The default reproduces :func:`augment` exactly.
    """
    controls = tuple(controls)
    bad = [c for c in controls if c not in VALID_CONTROLS]
    if bad or not controls:
        raise ValueError(f"controls must be a non-empty subset of {VALID_CONTROLS}; got {controls}.")
    if "shadow" in controls and "block" in controls:
        raise ValueError("'shadow' and 'block' are mutually exclusive.")
    if set(controls) == {"random", "shadow"}:
        return augment(Z, rng, n_random, shadow_reps)
    n, p = Z.shape
    parts, idx, off = [Z], dict(real=np.arange(p)), p
    if "random" in controls:
        n_random = n_random or max(14, int(np.ceil(0.05 * p)))
        n_random = int(np.ceil(n_random / 7) * 7)
        R = random_probes(n, n_random, rng)
        parts.append(R)
        idx["rand"] = np.arange(off, off + n_random)
        off += n_random
    reps = shadow_reps or default_shadow_reps(p)
    if "shadow" in controls:
        S = shadow_features(Z, reps, rng)
    elif "block" in controls:
        from .dependence import block_shadow, leader_clusters
        S = block_shadow(Z, leader_clusters(Z, block_thr, seed=seed), reps, rng)
    else:
        S = None
    if S is not None:
        parts.append(S)
        idx["shad"] = np.arange(off, off + S.shape[1])
    return np.ascontiguousarray(np.concatenate(parts, axis=1)), idx
