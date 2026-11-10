"""DecoySelector: decoy-referenced feature selection (Random + Shadow controls, heterogeneous rankers, vote policy)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import policies as P
from ._base import BaseSelector
from .controls import VALID_CONTROLS, build_controls
from .rankers import DEFAULT_RANKERS, RANKERS, run_rankers


class DecoySelector(BaseSelector):
    """Keep the features that beat the decoys according to several heterogeneous rankers.

    Protocol (per ``fit``): (1) preprocess the training data (median imputation, constant-column removal, Yeo-Johnson on
    skewed columns, robust scaling); (2) append decoys: Random probes (pure noise) and Shadow copies (independently
    permuted real columns); (3) score real features and decoys with every ranker; (4) a ranker *votes* for a feature
    when its score exceeds the best Random probe and the ``1 - alpha`` quantile of the Shadow scores; (5) the
    ``policy`` turns the votes into the final subset; (6) optionally, a stability filter keeps the features selected in
    at least ``stability_pi`` of ``stability_B`` subsamples.

    Parameters
    ----------
    rankers : sequence of str
        Names registered in :data:`decoyfs.rankers.RANKERS` (default: the 7 built-in rankers; see
        :func:`decoyfs.list_rankers`).
    controls : sequence of {"random", "shadow", "block"}
        Decoy blocks. Default ``("random", "shadow")``. ``"block"`` replaces ``"shadow"`` with Block-Shadow (one
        permutation per cluster of correlated features), a diagnostic for strongly dependent features.
    policy : {"all", "majority", "any"} or int or callable
        Votes needed: all rankers (precision-oriented), at least half (balanced, the reference policy), at least one
        (recall-oriented), an integer count, or a function ``(votes, n_rankers, consensus) -> indices``.
    alpha : float
        Tail level of the Shadow quantile (smaller = stricter).
    n_random, shadow_reps : int or None
        Number of Random probes (rounded up to a multiple of 7) and of Shadow copies per feature; ``None`` uses the
        protocol defaults (max(14, 5% of p) probes; min(10, ceil(40/p)) copies).
    stability_B, stability_pi, stability_frac : int, float, float
        Stability filter: ``B`` subsamples of fraction ``frac``; ``B = 0`` disables it.
    random_state : int
        Seed (controls, rankers and subsamples).

    Attributes
    ----------
    support_ : ndarray of bool, shape (n_features_in_,)
    report_ : DataFrame
        One row per input feature: consensus rank score, votes, share of rankers beating each control, per-ranker
        scores and the final decision. This is the audit trail of the selection.
    decoy_thresholds_ : DataFrame
        Per ranker, the Random maximum and the Shadow quantile that a feature has to exceed.
    null_pass_rate_ : dict
        Fraction of Random / Shadow decoys that would be inside the selected set (empirical proxy of the false-selection
        rate; NOT a formal FDR guarantee).
    fallback_ : bool
        True if the policy returned an empty set and the 3 top-consensus features were returned instead.
    ranker_errors_ : dict
        Rankers that failed (they are skipped and the vote is taken over the remaining ones).
    """

    def __init__(self, rankers=DEFAULT_RANKERS, controls=("random", "shadow"), policy="majority", alpha=0.05,
                 n_random=None, shadow_reps=None, stability_B=0, stability_pi=0.6, stability_frac=0.8,
                 random_state=0):
        self.rankers = rankers
        self.controls = controls
        self.policy = policy
        self.alpha = alpha
        self.n_random = n_random
        self.shadow_reps = shadow_reps
        self.stability_B = stability_B
        self.stability_pi = stability_pi
        self.stability_frac = stability_frac
        self.random_state = random_state

    # -------------------------------------------------------------- parameter checks
    def _check_params(self):
        rk = tuple(self.rankers)
        if not rk:
            raise ValueError("At least one ranker is required.")
        unknown = [r for r in rk if r not in RANKERS]
        if unknown:
            raise ValueError(f"Unknown rankers {unknown}. Available: {sorted(RANKERS)}")
        if len(set(rk)) != len(rk):
            raise ValueError("Duplicate rankers.")
        ctr = tuple(self.controls)
        if any(c not in VALID_CONTROLS for c in ctr) or not ctr:
            raise ValueError(f"controls must be a non-empty subset of {VALID_CONTROLS}.")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        if self.stability_B < 0 or int(self.stability_B) != self.stability_B:
            raise ValueError("stability_B must be a non-negative integer.")
        if not 0 < self.stability_pi <= 1 or not 0 < self.stability_frac < 1:
            raise ValueError("stability_pi must be in (0, 1] and stability_frac in (0, 1).")
        P.resolve_policy(self.policy)
        return rk, ctr

    # -------------------------------------------------------------- one pass
    def _pass(self, Z, y, k, seed, rk, ctr):
        rng = np.random.default_rng(seed)
        Xa, idx = build_controls(Z, rng, ctr, self.n_random, self.shadow_reps, seed=seed)
        scores, _, errors = run_rankers(Xa, y, k, seed, rk)
        if not scores:
            raise RuntimeError(f"All rankers failed: {errors}")
        fl = P.flags(scores, idx, self.alpha, ctr)
        votes = P.vote_counts(fl)
        c = P.consensus(scores)
        sel = np.asarray(P.resolve_policy(self.policy)(votes, len(scores), c[idx["real"]]))
        if sel.dtype == bool:
            sel = np.where(sel)[0]
        return np.asarray(sel, dtype=int), dict(scores=scores, idx=idx, fl=fl, votes=votes, c=c, errors=errors)

    # -------------------------------------------------------------- fit
    def _select(self, Z, y, k):
        rk, ctr = self._check_params()
        seed = 0 if self.random_state is None else int(self.random_state)
        p = Z.shape[1]
        sel, info = self._pass(Z, y, k, seed, rk, ctr)
        scores, idx, fl, votes, c = info["scores"], info["idx"], info["fl"], info["votes"], info["c"]
        M = len(scores)
        self.ranker_errors_ = info["errors"]
        self.rankers_used_ = list(scores)
        if info["errors"]:
            warnings.warn(f"Rankers skipped because they failed: {list(info['errors'])}", stacklevel=3)

        freq = None
        if self.stability_B > 0:
            cnt, valid = np.zeros(p), 0
            n = len(y)
            for b in range(int(self.stability_B)):
                sb = seed + 31 * (b + 1)
                sub = np.random.default_rng(sb).choice(n, size=int(self.stability_frac * n), replace=False)
                try:
                    s_b, _ = self._pass(Z[sub], y[sub], k, sb, rk, ctr)
                except Exception as e:  # noqa: BLE001
                    warnings.warn(f"Stability subsample {b} skipped: {e!r}", stacklevel=3)
                    continue
                cnt[s_b] += 1
                valid += 1
            if valid == 0:
                raise RuntimeError("All stability subsamples failed.")
            freq = cnt / valid
            sel = np.where(freq >= self.stability_pi)[0]

        order = np.argsort(-c[idx["real"]], kind="stable")
        self.fallback_ = len(sel) == 0
        if self.fallback_:
            warnings.warn("The policy selected no feature; returning the 3 features with the highest consensus. "
                          "Consider policy='any' or a larger alpha.", stacklevel=3)
            sel = order[:min(3, p)]
        sel = np.sort(sel)

        keep = self.preprocessor_.keep_
        n_in = self.n_features_in_
        cr = c[idx["real"]]
        sel_mask_local = np.zeros(p, bool)
        sel_mask_local[sel] = True
        rep = pd.DataFrame(index=np.arange(n_in))
        names = getattr(self, "feature_names_in_", None)
        rep["feature"] = names if names is not None else np.arange(n_in)
        rep["used"] = False
        rep.loc[keep, "used"] = True
        for col, vals in (("consensus", cr), ("votes", votes),
                          ("frac_beat_random", np.mean([a for a, _ in fl.values()], axis=0)),
                          ("frac_beat_shadow", np.mean([b for _, b in fl.values()], axis=0))):
            s = np.full(n_in, np.nan)
            s[keep] = vals
            rep[col] = s
        if "random" not in ctr:
            rep["frac_beat_random"] = np.nan
        if not any(c_ in ctr for c_ in ("shadow", "block")):
            rep["frac_beat_shadow"] = np.nan
        for m, s_m in scores.items():
            col = np.full(n_in, np.nan)
            col[keep] = s_m[idx["real"]]
            rep[f"score_{m}"] = col
        if freq is not None:
            s = np.full(n_in, np.nan)
            s[keep] = freq
            rep["stability_freq"] = s
        sel_full = np.zeros(n_in, bool)
        sel_full[keep[sel]] = True
        rep["selected"] = sel_full
        rep["n_rankers"] = M
        self.report_ = rep

        thr = {}
        for m, s_m in scores.items():
            thr[m] = dict(
                random_max=float(s_m[idx["rand"]].max()) if "rand" in idx else np.nan,
                shadow_quantile=float(np.quantile(s_m[idx["shad"]], 1 - self.alpha)) if "shad" in idx else np.nan)
        self.decoy_thresholds_ = pd.DataFrame(thr).T
        nr, ns = (P.null_pass_rates(sel, c, idx) if "rand" in idx and "shad" in idx else (np.nan, np.nan))
        if "rand" in idx and "shad" not in idx:
            nr = P.null_rate(sel, c, idx, "rand")
        if "shad" in idx and "rand" not in idx:
            ns = P.null_rate(sel, c, idx, "shad")
        self.null_pass_rate_ = dict(random=nr, shadow=ns)
        return sel
