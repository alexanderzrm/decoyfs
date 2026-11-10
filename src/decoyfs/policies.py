"""Selection policies: conventional baselines versus decoy-referenced policies (Random + Shadow).

A *policy* decides which features are kept once every ranker has scored real features and decoys. Decoy-referenced
policies are registered by name (``"all"``, ``"majority"``, ``"any"``) and can be extended with
:func:`register_policy`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

FIXK_FRACS = (0.10, 0.25, 0.50)


def consensus(scores: dict):
    """Normalised mean rank (Borda) over the augmented matrix."""
    r = [rankdata(s) / len(s) for s in scores.values()]
    return np.mean(r, axis=0)


def flags(scores, idx, alpha=0.05, controls=("random", "shadow"), shadow_key="shad"):
    """Per ranker: does a feature beat the maximum Random score and/or the (1-alpha) Shadow quantile?

    Returns ``{ranker: (beats_random, beats_shadow)}``; a block that is not requested is returned as all-True so that
    it does not restrict the vote.
    """
    out = {}
    for m, s in scores.items():
        sr = s[idx["real"]]
        a = sr > s[idx["rand"]].max() if "random" in controls else np.ones(len(sr), bool)
        b = sr > np.quantile(s[idx[shadow_key]], 1 - alpha) if ("shadow" in controls or "block" in controls) \
            else np.ones(len(sr), bool)
        out[m] = (a, b)
    return out


def vote_counts(fl):
    return np.sum([a & b for a, b in fl.values()], axis=0)


def vote_threshold(policy, n_rankers):
    """Number of rankers that must agree: ``all`` -> M, ``majority`` -> ceil(M/2), ``any`` -> 1, or an integer."""
    if isinstance(policy, (int, np.integer)) and not isinstance(policy, bool):
        if not 1 <= int(policy) <= n_rankers:
            raise ValueError(f"An integer policy must be between 1 and the number of rankers ({n_rankers}).")
        return int(policy)
    return {"all": n_rankers, "majority": int(np.ceil(n_rankers / 2)), "any": 1}[policy]


def _vote_policy(name):
    def fn(votes, n_rankers, consensus_scores):
        return np.where(votes >= vote_threshold(name, n_rankers))[0]
    fn.__name__ = f"vote_{name}"
    return fn


POLICIES = {"all": _vote_policy("all"), "majority": _vote_policy("majority"), "any": _vote_policy("any")}
_BUILTIN_POLICIES = frozenset(POLICIES)


def register_policy(name, fn, overwrite=False):
    """Add a policy ``fn(votes, n_rankers, consensus) -> indices or boolean mask`` over the real features.

    ``votes[j]`` is the number of rankers whose score for feature ``j`` beat the decoys; ``consensus[j]`` is its Borda
    score over the augmented matrix.
    """
    if name in POLICIES and not overwrite:
        raise ValueError(f"A policy called {name!r} already exists (use overwrite=True to replace it).")
    POLICIES[name] = fn


def unregister_policy(name):
    if name in _BUILTIN_POLICIES:
        raise ValueError(f"{name!r} is a built-in policy.")
    POLICIES.pop(name, None)


def list_policies():
    return sorted(POLICIES)


def resolve_policy(policy):
    """Turn a policy name, integer or callable into a callable ``(votes, n_rankers, consensus) -> indices``."""
    if callable(policy):
        return policy
    if isinstance(policy, (int, np.integer)) and not isinstance(policy, bool):
        return lambda votes, M, c, _t=int(policy): np.where(votes >= vote_threshold(_t, M))[0]
    if policy in POLICIES:
        return POLICIES[policy]
    raise ValueError(f"Unknown policy {policy!r}. Available: {list_policies()} (or an integer / a callable).")


def ctrl_sets(scores, idx, alpha=0.05):
    """Decoy-referenced sets for one run."""
    M = len(scores)
    c = consensus(scores)
    cr, cx, cs = c[idx["real"]], c[idx["rand"]], c[idx["shad"]]
    fl = flags(scores, idx, alpha)
    votes = vote_counts(fl)
    sets = {
        "CTRL_BORDA_RAND": np.where(cr > cx.max())[0],
        "CTRL_BORDA_SHAD": np.where(cr > np.quantile(cs, 1 - alpha))[0],
        "CTRL_BORDA_BOTH": np.where((cr > cx.max()) & (cr > np.quantile(cs, 1 - alpha)))[0],
        "CTRL_VOTE_ANY": np.where(votes >= 1)[0],
        "CTRL_VOTE_MAJ": np.where(votes >= int(np.ceil(M / 2)))[0],
        "CTRL_VOTE_ALL": np.where(votes >= M)[0],
    }
    return sets, c, votes, fl


def feature_report(c, votes, fl, idx, n_methods):
    br = np.mean([a for a, _ in fl.values()], axis=0)
    bs = np.mean([b for _, b in fl.values()], axis=0)
    return dict(consensus=c[idx["real"]], votes=votes, beat_rand=br, beat_shad=bs, n_methods=n_methods)


def null_pass_rates(sel, c, idx):
    """Fraction of decoys (Random / Shadow) that would fall inside the selected set: a proxy for the false-positive rate."""
    if len(sel) == 0:
        return 0.0, 0.0
    cmin = c[idx["real"]][sel].min()
    return float((c[idx["rand"]] >= cmin).mean()), float((c[idx["shad"]] >= cmin).mean())


def ablation_sets(scores, idx, alpha=0.05):
    """Protocol ablations computed ONLY from the scores already obtained (cost ~0).

    A001/A010: Shadow quantile level alpha; RANDONLY/SHADONLY: a single decoy type; PROBES_HALF: half of the Random
    probes (every second one, keeps the 7 families); LORO_<m>: leave ranker m out (majority of the remaining M-1).
    """
    M = len(scores)
    out = {}
    for tag, a in (("ABL_A001", 0.01), ("ABL_A010", 0.10)):
        out[tag] = ctrl_sets(scores, idx, a)[0]["CTRL_VOTE_MAJ"]
    fl = flags(scores, idx, alpha)
    thr = int(np.ceil(M / 2))
    out["ABL_RANDONLY"] = np.where(np.sum([a for a, _ in fl.values()], axis=0) >= thr)[0]
    out["ABL_SHADONLY"] = np.where(np.sum([b for _, b in fl.values()], axis=0) >= thr)[0]
    half = {**idx, "rand": idx["rand"][::2]}
    out["ABL_PROBES_HALF"] = np.where(vote_counts(flags(scores, half, alpha)) >= thr)[0]
    if M > 2:
        for m in scores:
            rest = {k: v for k, v in fl.items() if k != m}
            v = np.sum([a & b for a, b in rest.values()], axis=0)
            out[f"ABL_LORO_{m}"] = np.where(v >= int(np.ceil((M - 1) / 2)))[0]
    return out


def knockoff_stat_sets(c2, scores2, idx2, q=0.10, stats=("borda", "xgb_gain")):
    """Knockoff+ on W_j = statistic(real) - statistic(knockoff). ``borda`` uses the consensus of the second pass."""
    from .dependence import knockoff_threshold
    out = {}
    for st in stats:
        s = c2 if st == "borda" else scores2.get(st)
        if s is None:
            continue
        W = s[idx2["real"]] - s[idx2["ko"]]
        out[f"KO_{st.upper()}_q{int(q * 100)}"] = knockoff_threshold(W, q)
    return out


def block_vote_set(scores2, idx2, alpha=0.05):
    """CTRL_VOTE_MAJ with Block-Shadow instead of independent Shadow (same pass as the knockoffs)."""
    M = len(scores2)
    idx = {"real": idx2["real"], "rand": idx2["rand"], "shad": idx2["blk"]}
    return np.where(vote_counts(flags(scores2, idx, alpha)) >= int(np.ceil(M / 2)))[0]


def null_rate(sel, c, idx, key):
    """Fraction of ``key`` decoys whose consensus is >= the minimum consensus of the selection (NaN if no decoys)."""
    if key not in idx or len(sel) == 0:
        return np.nan if key not in idx else 0.0
    return float((c[idx[key]] >= c[idx["real"]][sel].min()).mean())


def cv_k(Z, y, order, seed, folds=3):
    """Expensive baseline: choose k by inner CV and the one-standard-error rule (geometric grid) on a given ordering."""
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from .device import xgb_device
    from .metrics import balanced_weights, evaluate
    p = len(order)
    grid = sorted({min(p, int(round(g))) for g in np.geomspace(1, p, num=min(10, p))})
    kfold = StratifiedKFold(folds, shuffle=True, random_state=seed)
    res = {k: [] for k in grid}
    for tr, va in kfold.split(Z, y):
        for k in grid:
            cols = order[:k]
            m = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.15, tree_method="hist",
                                  device=xgb_device(), random_state=seed, n_jobs=4, verbosity=0)
            m.fit(Z[tr][:, cols], y[tr], sample_weight=balanced_weights(y[tr]))
            P = m.predict_proba(Z[va][:, cols])
            res[k].append(evaluate(y[va], P)["auroc"])
    mean = np.array([np.nanmean(res[k]) for k in grid])
    se = np.array([np.nanstd(res[k]) / np.sqrt(folds) for k in grid])
    best = int(np.nanargmax(mean))
    ok = np.where(mean >= mean[best] - se[best])[0]
    return order[:grid[ok[0]]]


def build_policies(Z, y, scores, idx, cfg, run_stab=None, seed=0, ds_truth=None, extra=None):
    """Return ``(dict policy -> (indices, extra_time_s, fallback)), consensus, votes, flags`` (benchmark helper).

    ``extra``: ``{policy: (indices, extra_time_s)}`` computed elsewhere (external baselines, Block-Shadow, knockoffs).
    """
    import time
    p = len(idx["real"])
    sets, c, votes, fl = ctrl_sets(scores, idx, cfg["alpha"])
    order = np.argsort(-c[idx["real"]], kind="stable")
    pol, fb = {}, {}

    def put(name, sel, extra=0.0):
        sel = np.asarray(sel, dtype=int)
        fell = False
        if len(sel) == 0:
            sel, fell = order[:min(3, p)], True
        pol[name] = (np.sort(sel), extra, fell)

    pol["ALL"] = (np.arange(p), 0.0, False)
    for f in FIXK_FRACS:
        put(f"FIXK_{int(f*100)}", order[:max(2, int(round(f * p)))])
    if cfg.get("cvk", True):
        t0 = time.perf_counter()
        sel = cv_k(Z, y, order, seed)
        put("CVK_1SE", sel, time.perf_counter() - t0)
    for k, v in sets.items():
        put(k, v)
    if run_stab is not None:
        put("CTRL_STAB", run_stab[0], run_stab[1])
    if cfg.get("ablations", False):
        for k, v in ablation_sets(scores, idx, cfg["alpha"]).items():
            put(k, v)
    for k, (sel, t) in (extra or {}).items():
        put(k, sel, t)

    ref = cfg["ref_policy"]
    k0 = len(pol[ref][0])
    put("BORDA_MATCH", order[:k0])
    rng = np.random.default_rng(seed + 7)
    put("RAND_MATCH", rng.choice(p, size=k0, replace=False))
    rej = np.setdiff1d(np.arange(p), pol[ref][0])
    if 0 < len(rej):
        pol["REJECTED"] = (rej, 0.0, False)
    for m in cfg["matched_methods"]:
        if m in scores:
            s = scores[m][idx["real"]]
            put(f"TOPK_MATCH_{m}", np.argsort(-s, kind="stable")[:k0])
    return pol, c, votes, fl
