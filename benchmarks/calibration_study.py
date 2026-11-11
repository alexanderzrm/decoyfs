"""Calibration study: does the null-pass rate of a decoy-referenced selection predict its false-discovery proportion?

Draws random synthetic scenarios (sample size, dimension, signal strength, dependence, nonlinearity, noise type,
classes, imbalance), runs ONE selection pass per scenario (7 rankers, Random + Shadow controls, no downstream models)
and, from the same scores, derives many selection policies. For each (scenario, policy) it stores the true FDP and the
null-based estimators of the FDP:

    FDP_hat = min(1, FPR_hat * m0_hat / max(|S|, 1))

with FPR_hat from the decoys (R: Random probes, S: Shadow copies, C: both pooled, C1: pooled with +1 correction,
V: Shadow copies counted with the same vote rule as the policy) and m0_hat in {p, p-|S|, Storey-type estimate}.

Usage:  python calibration_study.py --n-scen 1200 --out ../results/calib [--start 0] [--force-gpu]
Resumable: scenarios already present in ``<out>/calib_rows.csv`` are skipped.
"""
import argparse
import csv
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from decoyfs import policies as P  # noqa: E402
from decoyfs.bench.data import _block_noise, _finish  # noqa: E402
from decoyfs.bench.data import Dataset  # noqa: E402,F401
from decoyfs.controls import build_controls  # noqa: E402
from decoyfs.device import device  # noqa: E402
from decoyfs.preprocessing import FoldPrep  # noqa: E402
from decoyfs.rankers import DEFAULT_RANKERS, run_rankers  # noqa: E402

N_GRID = [100, 200, 500, 1000, 2000, 3000]
P_GRID = [30, 50, 100, 200, 500, 1000]
FRAC_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
DEP_GRID = ["indep", "noiseblock", "block", "proxy"]
NL_GRID = ["lin", "tanh", "mixed", "xor"]
CLASS_PROPS = {2: [[.5, .5], [.7, .3], [.9, .1]], 3: [[.4, .35, .25], [.7, .2, .1]], 4: [[.4, .3, .2, .1], [.25] * 4]}


def draw_scenario(sid):
    r = np.random.default_rng(10_000 + sid)
    n, p = int(r.choice(N_GRID)), int(r.choice(P_GRID))
    if p >= 500:
        n = min(n, 2000)
    dep = str(r.choice(DEP_GRID))
    bs = int(min(r.choice([5, 10, 20]), max(2, p // 6)))
    ppc = int(r.choice([1, 2, 3]))
    frac = float(r.choice(FRAC_GRID))
    cap = {"indep": p // 2, "noiseblock": p // 2, "block": max(2, p // bs // 2), "proxy": max(2, p // (2 * (1 + ppc)))}[dep]
    n_inf = int(max(2, min(round(frac * p), cap)))
    k = int(r.choice([2, 2, 2, 2, 3, 4]))
    props = CLASS_PROPS[k][int(r.integers(len(CLASS_PROPS[k])))]
    return dict(sid=sid, n=n, p=p, dep=dep, bs=bs, ppc=ppc, rho=float(r.choice([0.3, 0.6, 0.9])),
                rho_noise=float(r.choice([0.0, 0.5])), frac=frac, n_inf=n_inf,
                nl=str(r.choice(NL_GRID)), strength=float(np.round(r.uniform(0.6, 4.0), 2)), k=k,
                props=",".join(f"{q:.2f}" for q in props), heavy=bool(r.random() < 0.4),
                label_noise=float(r.choice([0.0, 0.02, 0.05])), seed=20_000 + sid)


def _f(kind, x, rng):
    if kind == "lin":
        return x
    if kind == "tanh":
        return np.tanh(1.5 * x)
    if kind == "quad":
        return x ** 2 - 1
    return (x > rng.uniform(-0.5, 0.5)).astype(float)


def make_scenario(sc):
    rng = np.random.default_rng(sc["seed"])
    n, p, dep, bs, ni = sc["n"], sc["p"], sc["dep"], sc["bs"], sc["n_inf"]
    truth = np.zeros(p, int)
    if dep == "indep":
        X = rng.normal(size=(n, p))
        causal = rng.choice(p, ni, replace=False)
    elif dep == "noiseblock":
        X = _block_noise(n, p, bs, sc["rho"], rng)
        causal = rng.choice(p, ni, replace=False)
        X[:, causal] = rng.normal(size=(n, ni))
    elif dep == "block":
        X = _block_noise(n, p, bs, sc["rho"], rng)
        blocks = rng.choice(p // bs, ni, replace=False)
        causal = blocks * bs
        for b in blocks:
            truth[b * bs:(b + 1) * bs] = 1
    else:
        ppc, rho = sc["ppc"], sc["rho"]
        X = rng.normal(size=(n, p))
        causal = np.arange(ni)
        for j in range(ni):
            for q in range(ppc):
                col = ni + j * ppc + q
                X[:, col] = rho * X[:, j] + np.sqrt(1 - rho ** 2) * X[:, col]
                truth[col] = 1
        s0 = ni * (1 + ppc)
        if sc["rho_noise"] > 0 and p - s0 >= 2 * bs:
            X[:, s0:] = _block_noise(n, p - s0, bs, sc["rho_noise"], rng)
    truth[causal] = 2
    w = rng.choice([-1.0, 1.0], ni) * rng.uniform(0.7, 1.3, ni)
    if sc["nl"] == "xor":
        score = np.zeros(n)
        for a in range(0, ni - 1, 2):
            score += 2.0 * w[a] * X[:, causal[a]] * X[:, causal[a + 1]]
        if ni % 2:
            score += w[-1] * X[:, causal[-1]]
    else:
        kinds = (["lin"] * ni if sc["nl"] == "lin" else ["tanh"] * ni if sc["nl"] == "tanh"
                 else list(rng.choice(["lin", "quad", "step", "tanh"], ni)))
        score = sum(w[i] * _f(kinds[i], X[:, causal[i]], rng) for i in range(ni))
    score = (score - score.mean()) / (score.std() + 1e-12) * sc["strength"] * ni ** 0.25
    s = score + rng.logistic(size=n) * 0.7
    props = np.array([float(v) for v in sc["props"].split(",")])
    cuts = np.quantile(s, np.cumsum(props)[:-1])
    y = np.digitize(s, cuts)
    flip = rng.random(n) < sc["label_noise"]
    y = np.where(flip, rng.integers(0, sc["k"], n), y)
    if len(np.unique(y)) < 2:
        y[:2] = [0, 1]
    ds = _finish(f"scn{sc['sid']}", X, y, truth, rng, sc["heavy"], sc)
    ds.meta["k"] = int(len(np.unique(ds.y)))
    return ds


def _configs(M):
    thr = int(np.ceil(M / 2))
    cfg = [(f"V{t}", 0.05, ("random", "shadow"), t) for t in range(1, M + 1)]
    cfg += [("A001_MAJ", 0.01, ("random", "shadow"), thr), ("A010_MAJ", 0.10, ("random", "shadow"), thr),
            ("RANDONLY_MAJ", 0.05, ("random",), thr), ("SHADONLY_MAJ", 0.05, ("shadow",), thr)]
    return cfg


def run_scenario(sc):
    t0 = time.perf_counter()
    ds = make_scenario(sc)
    k = ds.meta["k"]
    prep = FoldPrep().fit(ds.X)
    Z = prep.transform(ds.X)
    keep = prep.keep_
    tk = ds.truth[keep]
    pk = len(keep)
    rng = np.random.default_rng(sc["seed"])
    Xa, idx = build_controls(Z, rng, ("random", "shadow"), None, None, seed=sc["seed"])
    scores, _, errors = run_rankers(Xa, ds.y, k, sc["seed"], DEFAULT_RANKERS)
    M = len(scores)
    c = P.consensus(scores)
    cr, cR, cS = c[idx["real"]], c[idx["rand"]], c[idx["shad"]]
    cC = np.concatenate([cR, cS])
    n_inf_total = max(1, int((ds.truth == 2).sum()))
    shadow_idx = {"real": idx["shad"], "rand": idx["rand"], "shad": idx["shad"]}

    sets = []                                            # (name, selection, vote-based shadow pass fraction fn)
    for name, alpha, ctr, thr in _configs(M):
        votes = P.vote_counts(P.flags(scores, idx, alpha, ctr))
        vs = P.vote_counts(P.flags(scores, shadow_idx, alpha, ctr))
        sets.append((name, np.where(votes >= thr)[0], float((vs >= thr).mean())))
    a05 = 0.05
    sets.append(("BORDA_RAND", np.where(cr > cR.max())[0], np.nan))
    sets.append(("BORDA_SHAD", np.where(cr > np.quantile(cS, 1 - a05))[0], np.nan))
    sets.append(("BORDA_BOTH", np.where((cr > cR.max()) & (cr > np.quantile(cS, 1 - a05)))[0], np.nan))
    order = np.argsort(-cr, kind="stable")
    for f in (0.10, 0.25, 0.50):
        sets.append((f"FIXK{int(f * 100)}", order[:max(1, int(np.ceil(f * pk)))], np.nan))

    q_half = np.quantile(cS, 0.5)
    pi0 = float(min(1.0, max(1.0 / pk, (cr <= q_half).mean() / 0.5)))
    m0_true = int((tk == 0).sum())
    base = {k_: sc[k_] for k_ in ("sid", "n", "p", "dep", "bs", "ppc", "rho", "rho_noise", "frac", "n_inf", "nl",
                                  "strength", "k", "props", "heavy", "label_noise")}
    base.update(p_kept=pk, M=M, m0_true=m0_true, pi0_hat=pi0, n_rand=len(cR), n_shad=len(cS),
                n_rank_err=len(errors), t_sec=round(time.perf_counter() - t0, 2))
    rows = []
    for name, sel, fpr_v in sets:
        sel = np.asarray(sel, int)
        ks = len(sel)
        row = dict(base, policy=name, k_sel=ks)
        if ks == 0:
            row.update(fdp=np.nan, rec_inf=0.0)
            rows.append(row)
            continue
        tau = cr[sel].min()
        fr, fs = float((cR >= tau).mean()), float((cS >= tau).mean())
        fc = float((cC >= tau).mean())
        fc1 = float((1 + (cC >= tau).sum()) / (1 + len(cC)))
        row.update(fdp=float((tk[sel] == 0).mean()), rec_inf=float((tk[sel] == 2).sum() / n_inf_total),
                   fpr_R=fr, fpr_S=fs, fpr_C=fc, fpr_C1=fc1, fpr_V=fpr_v, tau=float(tau))
        for kind, fpr in (("R", fr), ("S", fs), ("C", fc), ("C1", fc1), ("V", fpr_v)):
            for mk, m0 in (("p", pk), ("pk", pk - ks), ("st", pi0 * pk)):
                row[f"fdp_hat_{kind}_{mk}"] = min(1.0, fpr * m0 / ks) if np.isfinite(fpr) else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scen", type=int, default=1200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "calib"))
    ap.add_argument("--force-gpu", action="store_true")
    a = ap.parse_args()
    device(force_gpu=a.force_gpu)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "calib_rows.csv")
    done = set()
    if os.path.exists(path):
        import pandas as pd
        done = set(pd.read_csv(path, usecols=["sid"])["sid"].unique().tolist())
    sids = [s for s in range(a.start, a.start + a.n_scen) if s not in done]
    print(f"{len(sids)} scenarios to run ({len(done)} already done)", flush=True)
    header = None
    if os.path.exists(path):
        with open(path, newline="") as fh:
            header = next(csv.reader(fh))
    t_all = time.perf_counter()
    for i, sid in enumerate(sids):
        sc = draw_scenario(sid)
        try:
            rows = run_scenario(sc)
        except Exception as e:  # noqa: BLE001
            print(f"[{sid}] FAILED {e!r}", flush=True)
            continue
        if header is None:
            header = list(rows[0].keys())
            for r_ in rows:
                for kk in r_:
                    if kk not in header:
                        header.append(kk)
        with open(path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore", restval="")
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        el = time.perf_counter() - t_all
        print(f"[{sid}] n={sc['n']} p={sc['p']} {sc['dep']:<10} {rows[0]['t_sec']:6.1f}s | {i + 1}/{len(sids)} "
              f"elapsed {el / 60:.1f} min, ETA {(el / (i + 1)) * (len(sids) - i - 1) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
