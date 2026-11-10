"""Comparator selectors that are *not* decoy-referenced protocols: Boruta-style and NABFS-style (re-implementations).

All functions receive the fold's ALREADY preprocessed matrix (training data only) and return the selected column
indices. They are own re-implementations:

* ``boruta`` uses XGBoost-RF (gain, GPU-capable) instead of RandomForest-Gini, because BorutaPy is not compatible with
  numpy 2. It is therefore a *Boruta-style* selector, not the BorutaPy package.
* ``nabfs`` follows the description of Sinha et al. (2025): Gaussian noise features, bootstrap resamples, one-sided
  Wilcoxon test on D = importance - max noise importance, Holm correction.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy import stats

from .controls import shadow_features
from .device import xgb_device
from .metrics import balanced_weights as _bal_weights


def _gain(m, p_total):
    sc = m.get_booster().get_score(importance_type="total_gain")
    out = np.zeros(p_total)
    for kf, v in sc.items():
        out[int(kf[1:])] = v
    return out


def boruta(Z, y, seed=0, n_iter=25, alpha=0.05, max_rows=20000, budget_s=420.0):
    """Boruta-style: hit = real importance > max Shadow importance; binomial test with Bonferroni correction.

    ``budget_s`` caps wall-clock time; if exhausted the completed iterations are used (conservative: fewer trials mean
    fewer rejections of H0)."""
    import xgboost as xgb
    n, p = Z.shape
    rng = np.random.default_rng(seed)
    hits = np.zeros(p)
    trials = np.zeros(p)
    alive = np.ones(p, bool)
    t_start = time.perf_counter()
    for it in range(n_iter):
        if it >= 5 and time.perf_counter() - t_start > budget_s:
            break
        rows = rng.choice(n, size=max_rows, replace=False) if n > max_rows else np.arange(n)
        cols = np.where(alive)[0]
        sh = shadow_features(Z[rows][:, cols], max(1, int(np.ceil(5 / len(cols)))), rng)
        X = np.concatenate([Z[rows][:, cols], sh], axis=1)
        m = xgb.XGBRFClassifier(n_estimators=60, max_depth=5, subsample=0.632, colsample_bynode=0.3,
                                learning_rate=1.0, reg_lambda=1e-5, tree_method="hist", device=xgb_device(),
                                random_state=seed + it, n_jobs=4, verbosity=0)
        m.fit(X, y[rows], sample_weight=_bal_weights(y[rows]))
        imp = _gain(m, X.shape[1])
        thr = imp[len(cols):].max()
        hits[cols] += imp[:len(cols)] > thr
        trials[cols] += 1
        if it >= 9:                                           # early rejection of clearly irrelevant features
            lo = np.array([stats.binomtest(int(h), int(t), 0.5, alternative="less").pvalue
                           for h, t in zip(hits[cols], trials[cols])])
            alive[cols[lo < alpha / p]] = False
        if not alive.any():
            break
    pv = np.array([stats.binomtest(int(h), int(t), 0.5, alternative="greater").pvalue if t > 0 else 1.0
                   for h, t in zip(hits, trials)])
    return np.where(pv < alpha / p)[0]


def nabfs(Z, y, seed=0, n_noise=7, n_boot=20, alpha=0.05):
    """NABFS-style (Sinha et al.): Gaussian noise + bootstrap + one-sided Wilcoxon + Holm on D."""
    import xgboost as xgb
    n, p = Z.shape
    rng = np.random.default_rng(seed)
    D = np.zeros((n_boot, p))
    for b in range(n_boot):
        rows = rng.integers(0, n, n)
        N = rng.normal(size=(n, n_noise)).astype("float32")
        X = np.concatenate([Z[rows], N[rows]], axis=1)
        m = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", device=xgb_device(), random_state=seed + b, n_jobs=4, verbosity=0)
        m.fit(X, y[rows], sample_weight=_bal_weights(y[rows]))
        imp = _gain(m, X.shape[1])
        D[b] = imp[:p] - imp[p:].max()
    pv = np.ones(p)
    for j in range(p):
        d = D[:, j]
        if np.all(d <= 0):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pv[j] = stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue
            except ValueError:
                pv[j] = 1.0
    return np.where(_holm(pv) < alpha)[0]


def _holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def run_baselines(Z, y, seed, cfg):
    """Return ``{policy: (indices, time_s)}`` for the baselines enabled in ``cfg['baselines']`` (benchmark helper)."""
    out = {}
    for name in cfg.get("baselines", []):
        t0 = time.perf_counter()
        try:
            if name == "BORUTA":
                sel = boruta(Z, y, seed, n_iter=cfg.get("boruta_iter", 25), alpha=cfg["alpha"],
                             budget_s=cfg.get("boruta_budget", 420.0))
            elif name == "NABFS":
                sel = nabfs(Z, y, seed, n_noise=cfg.get("nabfs_noise", 7), n_boot=cfg.get("nabfs_boot", 20),
                            alpha=cfg["alpha"])
            else:
                continue
            out[name] = (np.asarray(sel, int), time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            out[name + "_ERR"] = (np.array([], int), repr(e)[:150])
    return out
