"""scikit-learn compatible wrappers of the comparators (Fixed-k, CV-k, Boruta-style, NABFS).

They share the interface and the preprocessing of :class:`decoyfs.DecoySelector`, so every method can be swapped in
:func:`decoyfs.compare_selectors` or in a scikit-learn ``Pipeline``.

Note: ``BorutaSelector`` is a Boruta-style re-implementation (XGBoost gain importance, shadow features, binomial test
with Bonferroni correction) as used in the DecoyFS benchmark. It is NOT the ``BorutaPy`` package.
"""
from __future__ import annotations

import numpy as np

from . import comparators
from ._base import BaseSelector
from .policies import consensus, cv_k
from .rankers import DEFAULT_RANKERS, RANKERS, run_rankers


def _order(Z, y, k, seed, rankers):
    scores, _, errors = run_rankers(Z, y, k, seed, tuple(rankers))
    if not scores:
        raise RuntimeError(f"All rankers failed: {errors}")
    return np.argsort(-consensus(scores), kind="stable")


class _RankedSelector(BaseSelector):
    def _check_rankers(self):
        unknown = [r for r in self.rankers if r not in RANKERS]
        if unknown:
            raise ValueError(f"Unknown rankers {unknown}. Available: {sorted(RANKERS)}")


class FixedKSelector(_RankedSelector):
    """Top-k by the Borda consensus of the rankers. ``k`` is an int, or a float in (0, 1] meaning a fraction of p."""

    def __init__(self, k=0.25, rankers=DEFAULT_RANKERS, random_state=0):
        self.k = k
        self.rankers = rankers
        self.random_state = random_state

    def _select(self, Z, y, n_classes):
        self._check_rankers()
        p = Z.shape[1]
        if isinstance(self.k, float):
            if not 0 < self.k <= 1:
                raise ValueError("A float k must be in (0, 1].")
            kk = int(np.ceil(self.k * p))
        else:
            kk = int(self.k)
            if kk < 1:
                raise ValueError("k must be >= 1.")
        order = _order(Z, y, n_classes, self.random_state, self.rankers)
        return order[:min(kk, p)]


class CVKSelector(_RankedSelector):
    """Number of features chosen by inner cross-validation with the one-standard-error rule, on the consensus order."""

    def __init__(self, rankers=DEFAULT_RANKERS, folds=3, random_state=0):
        self.rankers = rankers
        self.folds = folds
        self.random_state = random_state

    def _select(self, Z, y, n_classes):
        self._check_rankers()
        order = _order(Z, y, n_classes, self.random_state, self.rankers)
        return np.asarray(cv_k(Z, y, order, self.random_state, folds=self.folds))


class BorutaSelector(BaseSelector):
    """Boruta-style all-relevant selection (see module note)."""

    def __init__(self, n_iter=25, alpha=0.05, max_rows=20000, budget_s=420.0, random_state=0):
        self.n_iter = n_iter
        self.alpha = alpha
        self.max_rows = max_rows
        self.budget_s = budget_s
        self.random_state = random_state

    def _select(self, Z, y, n_classes):
        return comparators.boruta(Z, y, self.random_state, self.n_iter, self.alpha, self.max_rows, self.budget_s)


class NABFSSelector(BaseSelector):
    """Noise-augmented bootstrap feature selection with Holm correction."""

    def __init__(self, n_noise=7, n_boot=20, alpha=0.05, random_state=0):
        self.n_noise = n_noise
        self.n_boot = n_boot
        self.alpha = alpha
        self.random_state = random_state

    def _select(self, Z, y, n_classes):
        return comparators.nabfs(Z, y, self.random_state, self.n_noise, self.n_boot, self.alpha)
