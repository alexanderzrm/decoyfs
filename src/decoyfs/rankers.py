"""Rankers (7 built-in methods) and the ranker registry.

Every ranker maps the *augmented* matrix ``[real | decoys]`` to one relevance score per column (higher = more
relevant). ReliefF, chi-square and mutual information work on uniform quantiles (no advantage from scale) and run on
torch (GPU when available, CPU otherwise).

Custom rankers: ``register_ranker("name", fn)`` where ``fn(X, y) -> 1-D array`` with one score per column of ``X``
(``X`` is a float32 matrix, ``y`` the integer class codes). Pass ``takes_ctx=True`` to receive the internal
:class:`Ctx` instead (advanced).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from .device import device, xgb_device
from .metrics import balanced_weights as _bal_weights


class Ctx:
    """Shared state for one ranking pass: data, number of classes, seed, device and cached transforms."""

    def __init__(self, X, y, k, seed):
        self.X, self.y, self.k, self.seed = X, y, k, seed
        self.dev = device()
        self._U = self._Xt = self._Ut = None
        self.yt = torch.as_tensor(y, dtype=torch.long, device=self.dev)

    @property
    def U(self):
        if self._U is None:
            n = self.X.shape[0]
            self._U = ((rankdata(self.X, axis=0, method="average") - 0.5) / n).astype("float32")
        return self._U

    @property
    def Xt(self):
        if self._Xt is None:
            self._Xt = torch.as_tensor(self.X, dtype=torch.float32, device=self.dev)
        return self._Xt

    @property
    def Ut(self):
        if self._Ut is None:
            self._Ut = torch.as_tensor(self.U, dtype=torch.float32, device=self.dev)
        return self._Ut


def anova(c: Ctx):
    """ANOVA F statistic."""
    X, y, k = c.Xt, c.yt, c.k
    n = X.shape[0]
    Y = F.one_hot(y, k).float()
    cnt = Y.sum(0).clamp(min=1)
    mu_c = (Y.T @ X) / cnt[:, None]
    mu = X.mean(0)
    ssb = (cnt[:, None] * (mu_c - mu) ** 2).sum(0)
    ssw = ((X - Y @ mu_c) ** 2).sum(0)
    return ((ssb / (k - 1)) / (ssw / (n - k) + 1e-12)).cpu().numpy().astype("float64")


def chi2q(c: Ctx):
    """Chi-square statistic on uniform quantiles."""
    U, y, k = c.Ut, c.yt, c.k
    Y = F.one_hot(y, k).float()
    obs = Y.T @ U
    exp = (Y.sum(0) / U.shape[0])[:, None] * U.sum(0)[None, :]
    return (((obs - exp) ** 2) / (exp + 1e-12)).sum(0).cpu().numpy().astype("float64")


def mi_bin(c: Ctx, bins=10):
    """Mutual information with ``y`` on 10 quantile bins."""
    U, y, k = c.Ut, c.yt, c.k
    n, p = U.shape
    Y = F.one_hot(y, k).float()
    b = torch.clamp((U * bins).long(), max=bins - 1)
    py = Y.sum(0) / n
    joint = torch.stack([Y.T @ (b == v).float() for v in range(bins)])      # bins x k x p
    joint = joint / n
    pb = joint.sum(1, keepdim=True)
    mi = (joint * (torch.log(joint + 1e-12) - torch.log(pb + 1e-12) - torch.log(py[None, :, None] + 1e-12))).sum((0, 1))
    return mi.cpu().numpy().astype("float64")


def relieff(c: Ctx, n_samp=400, k_nn=10):
    """ReliefF (L1 distance on quantiles, up to 400 sampled rows, 10 neighbours per class)."""
    U, y, kc = c.Ut, c.yt, c.k
    n, p = U.shape
    g = torch.Generator().manual_seed(c.seed)
    m = min(n, n_samp)
    idx = torch.randperm(n, generator=g)[:m].to(U.device)
    prior = torch.bincount(y, minlength=kc).float() / n
    W = torch.zeros(p, device=U.device)
    chunk = int(max(4, min(128, 3e7 // max(1, k_nn * p), 3e7 // max(1, n))))
    inf = float("inf")
    for s in range(0, m, chunk):
        ii = idx[s:s + chunk]
        S, yi = U[ii], y[ii]
        D = torch.cdist(S, U, p=1)
        D[torch.arange(len(ii), device=U.device), ii] = inf
        for cl in range(kc):
            Dm = D.masked_fill((y != cl)[None, :], inf)
            kk = int(min(k_nn, (y == cl).sum().item() - 1))
            if kk < 1:
                continue
            vals, ind = torch.topk(Dm, kk, dim=1, largest=False)
            valid = torch.isfinite(vals).float()
            diff = (S[:, None, :] - U[ind]).abs() * valid[..., None]
            msum = diff.sum(1) / valid.sum(1).clamp(min=1)[:, None]
            w = torch.where(yi == cl, torch.full_like(prior[yi], -1.0), prior[cl] / (1 - prior[yi] + 1e-9))
            W += (w[:, None] * msum).sum(0) / m
        del D
    return W.cpu().numpy().astype("float64")


def rf_mdi(c: Ctx):
    """Random-forest mean decrease in impurity."""
    from sklearn.ensemble import RandomForestClassifier
    n = c.X.shape[0]
    m = RandomForestClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=3, n_jobs=-1,
                               random_state=c.seed, class_weight="balanced_subsample",
                               max_samples=(8000 / n if n > 8000 else None))
    m.fit(c.X, c.y)
    return m.feature_importances_.astype("float64")


def xgb_gain(c: Ctx):
    """XGBoost total gain."""
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                          tree_method="hist", device=xgb_device(), random_state=c.seed, n_jobs=4, verbosity=0)
    m.fit(c.X, c.y, sample_weight=_bal_weights(c.y))
    sc = m.get_booster().get_score(importance_type="total_gain")
    out = np.zeros(c.X.shape[1])
    for kf, v in sc.items():
        out[int(kf[1:])] = v
    return out


def l1_logreg(c: Ctx):
    """Largest absolute coefficient of an L1-penalised logistic regression."""
    from sklearn.linear_model import LogisticRegression
    X = (c.X - c.X.mean(0)) / (c.X.std(0) + 1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, class_weight="balanced", max_iter=200)
        if len(np.unique(c.y)) > 2:
            # explicit one-vs-rest: sklearn >= 1.8 no longer supports multiclass liblinear implicitly
            from sklearn.multiclass import OneVsRestClassifier
            m = OneVsRestClassifier(base).fit(X, c.y)
            coef = np.vstack([e.coef_ for e in m.estimators_])
        else:
            coef = base.fit(X, c.y).coef_
    return np.abs(coef).max(0).astype("float64")


RANKERS = {"anova": anova, "chi2q": chi2q, "mi": mi_bin, "relieff": relieff,
           "rf": rf_mdi, "xgb_gain": xgb_gain, "l1lr": l1_logreg}
DEFAULT_RANKERS = tuple(RANKERS)
_BUILTIN = frozenset(RANKERS)


def register_ranker(name, fn, takes_ctx=False, overwrite=False):
    """Add a ranker. ``fn(X, y) -> scores`` (one per column of ``X``); with ``takes_ctx=True``, ``fn(ctx) -> scores``."""
    if not isinstance(name, str) or not name:
        raise ValueError("The ranker name must be a non-empty string.")
    if name in RANKERS and not overwrite:
        raise ValueError(f"A ranker called {name!r} already exists (use overwrite=True to replace it).")
    if not callable(fn):
        raise TypeError("fn must be callable.")
    RANKERS[name] = fn if takes_ctx else (lambda c, _f=fn: np.asarray(_f(c.X, c.y), dtype="float64"))


def unregister_ranker(name):
    """Remove a custom ranker (built-in rankers cannot be removed)."""
    if name in _BUILTIN:
        raise ValueError(f"{name!r} is a built-in ranker.")
    RANKERS.pop(name, None)


def list_rankers():
    return sorted(RANKERS)


def run_rankers(X, y, k, seed, methods):
    """Run the rankers on the augmented matrix. Returns ``(scores, times, errors)``."""
    ctx = Ctx(X, y, k, seed)
    scores, times, errors = {}, {}, {}
    for m in methods:
        t0 = time.perf_counter()
        try:
            s = np.asarray(RANKERS[m](ctx), dtype="float64")
            if s.shape != (X.shape[1],):
                raise ValueError(f"ranker {m!r} returned shape {s.shape}, expected ({X.shape[1]},)")
            if not np.all(np.isfinite(s)):
                s = np.nan_to_num(s)
            scores[m] = s
        except Exception as e:  # noqa: BLE001
            errors[m] = repr(e)[:200]
        if ctx.dev == "cuda":
            torch.cuda.synchronize()
        times[m] = time.perf_counter() - t0
    del ctx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, times, errors
