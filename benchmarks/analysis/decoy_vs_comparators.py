"""Paired comparison of DecoyFS against Boruta (and CVK_1SE): AUROC, FDP/recall, size, cost and stability.
Usage (from the ``benchmarks`` directory, after a full run): python analysis/decoy_vs_comparators.py"""
import glob

import numpy as np
import pandas as pd
from scipy import stats

from decoyfs.bench.stats import bootstrap_ci, stability_table

D = "results/full"
MODELS = ["logreg", "knn", "xgb", "mlp"]
DECOY = ["CTRL_VOTE_MAJ", "CTRL_STAB", "ABL_A001", "CTRL_VOTE_ANY"]
COMP = ["BORUTA", "CVK_1SE"]
DELTA = 0.01

res = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/res_*.csv")])
sets = pd.concat([pd.read_csv(f) for f in glob.glob(f"{D}/sets_*.csv")])
sets["syn"] = sets.dataset.str.startswith("syn")
res["syn"] = res.dataset.str.startswith("syn")


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def paired(a, b, delta=DELTA):
    """a-b por dataset: media, IC bootstrap, Wilcoxon bilateral, TOST de equivalencia (±delta)."""
    d = (a - b).dropna().values
    lo, hi = bootstrap_ci(d, 10000)
    try:
        p2 = stats.wilcoxon(d).pvalue
    except ValueError:
        p2 = 1.0
    try:
        p_lo = stats.wilcoxon(d + delta, alternative="greater").pvalue
        p_hi = stats.wilcoxon(d - delta, alternative="less").pvalue
    except ValueError:
        p_lo = p_hi = 1.0
    return dict(n=len(d), mean=d.mean(), lo=lo, hi=hi, median=np.median(d), wins=int((d > 1e-4).sum()),
                losses=int((d < -1e-4).sum()), p_two=p2, p_tost=max(p_lo, p_hi))


# ---------------- 1) AUROC pareado (promedio de folds por dataset)
rows = []
for m in MODELS:
    pv = res[res.model == m].groupby(["dataset", "policy"]).auroc.mean().unstack()
    syn = pv.index.str.startswith("syn")
    for sub, idx in [("all74", np.ones(len(pv), bool)), ("real58", ~syn), ("syn16", syn)]:
        for pol in DECOY:
            for c in COMP:
                rows.append(dict(model=m, subset=sub, DECOY=pol, comp=c, **paired(pv.loc[idx, pol], pv.loc[idx, c])))
A = pd.DataFrame(rows)
A["p_two_holm"] = A.groupby(["subset", "comp"]).p_two.transform(holm)
A["p_tost_holm"] = A.groupby(["subset", "comp"]).p_tost.transform(holm)
A.to_csv("analysis/decoy_vs_comp_auroc.csv", index=False)

# ---------------- 2) Verdad conocida (16 sintéticos): FDP, recall, k
T = sets[sets.syn].groupby(["dataset", "policy"])[["fdp", "rec_inf", "k"]].mean().unstack()
rows = []
dep = [d for d in T.index if d.split("_")[0] in ("syn13", "syn14", "syn15", "syn16")]
for sub, idx in [("syn16", list(T.index)), ("dep_syn13-16", dep)]:
    for pol in DECOY:
        for c in COMP:
            for met in ["fdp", "rec_inf"]:
                x = T[met][pol].loc[idx]; y = T[met][c].loc[idx]
                d = (x - y).dropna().values
                try:
                    p = stats.wilcoxon(d).pvalue
                except ValueError:
                    p = 1.0
                rows.append(dict(subset=sub, DECOY=pol, comp=c, metric=met, decoy_mean=x.mean(), comp_mean=y.mean(),
                                 diff=d.mean(), p=p, n=len(d)))
B = pd.DataFrame(rows)
B.to_csv("analysis/decoy_vs_comp_truth.csv", index=False)

# ---------------- 3) Costo, fracción retenida y estabilidad (74 datasets)
ts = sets.groupby(["dataset", "policy"]).agg(t=("t_select", "mean"), k=("k", "mean"), p=("p", "first")).reset_index()
tp = ts.pivot(index="dataset", columns="policy", values="t")
kp = ts.pivot(index="dataset", columns="policy", values="k").div(ts.groupby("dataset").p.first(), axis=0)
st = stability_table(sets).pivot(index="dataset", columns="policy", values="stability")
rows = []
for pol in DECOY + COMP:
    rows.append(dict(policy=pol, t_mean=tp[pol].mean(), t_median=tp[pol].median(), frac_median=kp[pol].median(),
                     stab_mean=st[pol].mean()))
C = pd.DataFrame(rows)
C.to_csv("analysis/decoy_vs_comp_cost.csv", index=False)
ratio = (tp["BORUTA"] / tp["CTRL_VOTE_MAJ"])
meta = ts.groupby("dataset").p.first()
print("\nRatio t_Boruta/t_DECOY: mediana %.1f  rango [%.1f, %.1f]; Spearman con p: %.2f" %
      (ratio.median(), ratio.min(), ratio.max(), stats.spearmanr(ratio, meta.loc[ratio.index]).correlation))
dst = pd.DataFrame({"stab_decoy": st["CTRL_VOTE_MAJ"], "stab_boruta": st["BORUTA"]}).dropna()
print("Estabilidad VOTE_MAJ vs Boruta: Wilcoxon p=%.3g, dif media %.3f" %
      (stats.wilcoxon(dst.stab_decoy - dst.stab_boruta).pvalue, (dst.stab_decoy - dst.stab_boruta).mean()))
for pol in DECOY[1:3]:
    d = (st[pol] - st["BORUTA"]).dropna()
    print(f"Estabilidad {pol} - Boruta: {d.mean():+.3f} (p={stats.wilcoxon(d).pvalue:.3g}); wins {int((d>0).sum())}/{len(d)}")

# ---------------- 4) Presupuesto de tiempo: ¿cuántos datasets completa cada método con un presupuesto fijo?
tot = pd.DataFrame({"DECOY": tp["CTRL_VOTE_MAJ"], "BORUTA": tp["BORUTA"], "CVK": tp["CVK_1SE"]}) * 10  # 10 folds
print("\nTiempo total de selección (h) en 74 datasets x 10 folds:", (tot.sum() / 3600).round(2).to_dict())
print("datasets donde Boruta tarda > 60 s por fold:", int((tp["BORUTA"] > 60).sum()), " DECOY:", int((tp["CTRL_VOTE_MAJ"] > 60).sum()))

# ---------------- 5) Frontera recall-precisión (syn16)
g = sets[sets.syn].groupby("policy")[["fdp", "rec_inf"]].mean()
print("\nFrontera (syn16):"); print(g.loc[DECOY + COMP + ["CTRL_VOTE_ALL", "FIXK_25", "FIXK_50"]].round(3))

pd.set_option("display.width", 220)
print("\nAUROC (all74/real58/syn16), mean [CI] wins/losses, p_two_holm, p_tost_holm")
for c in COMP:
    for pol in DECOY[:3]:
        for sub in ["all74", "real58", "syn16"]:
            x = A[(A.DECOY == pol) & (A.comp == c) & (A["subset"] == sub)]
            print(c, pol, sub, " | ".join(f"{r.model}:{r['mean']:+.4f}[{r.lo:+.3f},{r.hi:+.3f}] {r.wins}/{r.losses} pH={r.p_two_holm:.2f} eqH={r.p_tost_holm:.3f}" for _, r in x.iterrows()))
print("\nTruth:"); print(B.round(4).to_string())
print("\nCost:"); print(C.round(3))
