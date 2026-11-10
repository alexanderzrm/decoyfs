"""Dependence-aware decoys: Block-Shadow and Gaussian knockoffs (second-order, model-X).

The standard Shadow permutes every column separately, which also breaks the dependence *between* features, so the
decoys are independent of each other even when the real features are correlated. If that mismatch inflates the false
positive rate, these decoys reveal it:

* **Block-Shadow**: one row permutation per cluster of correlated features (keeps intra-block correlation, breaks the
  relation with ``y``).
* **Gaussian knockoffs** (Candes et al., 2018): copies with the same covariance structure as the real features
  (Ledoit-Wolf estimate on normal scores), exactly exchangeable under the Gaussian model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def _normal_scores(Z):
    n = Z.shape[0]
    U = np.apply_along_axis(rankdata, 0, Z) / (n + 1.0)
    return norm.ppf(U)


def leader_clusters(Z, thr=0.5, max_rows=2000, seed=0):
    """Greedy 'leader' clustering on |Spearman| > thr. Returns integer labels of shape (p,)."""
    n, p = Z.shape
    rng = np.random.default_rng(seed)
    rows = rng.choice(n, size=min(n, max_rows), replace=False) if n > max_rows else np.arange(n)
    G = _normal_scores(Z[rows].astype("float64"))
    G = (G - G.mean(0)) / (G.std(0) + 1e-12)
    C = np.abs(G.T @ G) / len(rows)
    lab = -np.ones(p, int)
    nxt = 0
    for j in range(p):
        if lab[j] >= 0:
            continue
        members = np.where((C[j] > thr) & (lab < 0))[0]
        lab[members] = nxt
        lab[j] = nxt
        nxt += 1
    return lab


def block_shadow(Z, labels, reps, rng):
    """``reps`` Shadow copies where all columns of a cluster share the same row permutation."""
    n, p = Z.shape
    out = np.empty((n, p * reps), dtype=Z.dtype)
    groups = [np.where(labels == c)[0] for c in np.unique(labels)]
    for r in range(reps):
        for cols in groups:
            perm = rng.permutation(n)
            out[:, r * p + cols] = Z[perm[:, None], cols[None, :]]
    return out


def gaussian_knockoffs(Z, rng, max_rows=5000):
    """Gaussian knockoffs with equicorrelated s. Empirical marginals are preserved (quantile mapping)."""
    from sklearn.covariance import LedoitWolf
    n, p = Z.shape
    Z = Z.astype("float64")
    G = _normal_scores(Z)
    rows = rng.choice(n, size=min(n, max_rows), replace=False) if n > max_rows else np.arange(n)
    cov = LedoitWolf().fit(G[rows]).covariance_
    d = np.sqrt(np.diag(cov))
    S = cov / np.outer(d, d)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-6, None)
    s = float(min(1.0, 2.0 * w.min()) * 0.999)
    Sinv = (V / w) @ V.T
    Gs = (G - G.mean(0)) / d
    mu = Gs @ (np.eye(p) - s * Sinv)
    V2 = 2.0 * s * np.eye(p) - s * s * Sinv
    w2, U2 = np.linalg.eigh((V2 + V2.T) / 2)
    L = U2 * np.sqrt(np.clip(w2, 0.0, None))
    Gk = mu + rng.normal(size=(n, p)) @ L.T
    Uk = norm.cdf(Gk)
    out = np.empty_like(Z)
    for j in range(p):
        srt = np.sort(Z[:, j])
        out[:, j] = np.interp(Uk[:, j] * (n - 1), np.arange(n), srt)
    return out.astype("float32"), s


def knockoff_threshold(W, q=0.10, plus=True):
    """Barber-Candes knockoff(+) threshold. Returns the selected indices (W >= tau)."""
    W = np.asarray(W, float)
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    off = 1.0 if plus else 0.0
    for t in ts:
        fdp_hat = (off + np.sum(W <= -t)) / max(1, np.sum(W >= t))
        if fdp_hat <= q:
            return np.where(W >= t)[0]
    return np.array([], dtype=int)
