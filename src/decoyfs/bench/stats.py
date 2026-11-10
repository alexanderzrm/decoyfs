"""Análisis estadístico: inferencia entre datasets (Demšar 2006), no-inferioridad, estabilidad, calibración de FPR, costo."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

from ..metrics import nogueira_stability


def load(out_dir):
    def cat(prefix):
        fs = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.csv")))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return cat("res"), cat("sets"), cat("feat")


def dataset_table(res, metric="auroc"):
    """Media por dataset (sobre folds/repeticiones) x modelo x política."""
    return res.groupby(["dataset", "model", "policy"])[metric].mean().reset_index()


DIAGNOSTIC_POLICIES = ("REJECTED",)  # conjuntos de diagnóstico (rechazados por los controles), no políticas de selección


def pivot(res, model, metric="auroc", policies=None, min_cov=1.0):
    """Datasets x políticas. Las políticas con cobertura < min_cov (p. ej. KO_*, inviables en algún dataset) se descartan
    antes de eliminar datasets, de modo que una política parcial no reduzca la muestra de datasets (n = todos los
    datasets con resultados). Con min_cov < 1 se conservan políticas parciales y se restringe a los datasets comunes.
    Los conjuntos de diagnóstico (REJECTED) solo entran si se piden explícitamente en `policies`."""
    t = dataset_table(res[res.model == model] if model != "ALL_MODELS" else res, metric)
    pv = t.pivot_table(index="dataset", columns="policy", values=metric, aggfunc="mean")
    if policies:
        pv = pv[[c for c in policies if c in pv.columns]]
    else:
        pv = pv.drop(columns=[c for c in DIAGNOSTIC_POLICIES if c in pv.columns])
    pv = pv.loc[:, pv.notna().mean() >= min_cov - 1e-9]
    return pv.dropna()


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def friedman_ranks(pv, higher_better=True):
    ranks = pv.rank(axis=1, ascending=not higher_better)
    stat, p = stats.friedmanchisquare(*[pv[c].values for c in pv.columns]) if pv.shape[1] > 2 else (np.nan, np.nan)
    return ranks.mean().sort_values(), stat, p


def nemenyi_cd(k, n, alpha=0.05):
    from scipy.stats import studentized_range
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def vs_reference(pv, ref, delta=0.01):
    """Wilcoxon + Holm vs referencia y prueba de NO-INFERIORIDAD (H0: política peor que ref por más de delta)."""
    rows = []
    for c in pv.columns:
        if c == ref:
            continue
        d = (pv[c] - pv[ref]).values
        try:
            p2 = stats.wilcoxon(d, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            pni = stats.wilcoxon(d + delta, alternative="greater", zero_method="zsplit").pvalue
        except ValueError:
            p2 = pni = np.nan
        rows.append(dict(policy=c, mean_diff=d.mean(), median_diff=np.median(d), wins=int((d > 0).sum()),
                         ties=int((d == 0).sum()), losses=int((d < 0).sum()), p_wilcoxon=p2, p_noninf=pni))
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df.p_wilcoxon.fillna(1).values)
        df["p_noninf_holm"] = holm(df.p_noninf.fillna(1).values)
        df["noninferior"] = df.p_noninf_holm < 0.05
    return df.sort_values("mean_diff", ascending=False)


def nadeau_bengio(res, dataset, model, a, b, metric="auroc", n_splits=5):
    """t corregida (Nadeau & Bengio 2003) dentro de un dataset entre dos políticas."""
    r = res[(res.dataset == dataset) & (res.model == model)]
    x = r[r.policy == a].set_index(["rep", "fold"])[metric]
    y = r[r.policy == b].set_index(["rep", "fold"])[metric]
    d = (x - y).dropna().values
    J = len(d)
    if J < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    corr = 1 / J + 1 / (n_splits - 1)
    t = d.mean() / np.sqrt(corr * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), J - 1)


def stability_table(sets):
    rows = []
    for (ds, pol), g in sets.groupby(["dataset", "policy"]):
        p = int(g.p.iloc[0])
        allf = set()
        sel = [set(map(int, s.split(","))) if isinstance(s, str) and s else set() for s in g.selected]
        for s in sel:
            allf |= s
        universe = sorted(allf)
        if not universe or len(sel) < 2:
            continue
        Z = np.array([[f in s for f in universe] for s in sel], float)
        # estabilidad respecto al universo completo p (atributos nunca elegidos cuentan como 0)
        Z = np.concatenate([Z, np.zeros((Z.shape[0], max(0, p - Z.shape[1])))], axis=1)
        rows.append(dict(dataset=ds, policy=pol, stability=nogueira_stability(Z), k_mean=g.k.mean(), p=p))
    return pd.DataFrame(rows)


def summary(res, sets, model, ref="ALL", metric="auroc", delta=0.01, policies=None, min_cov=1.0):
    """Tabla maestra: ranking medio, Δ vs ALL, % atributos retenidos, no-inferioridad, costo, estabilidad, FPR."""
    pv = pivot(res, model, metric, policies, min_cov)
    ranks, stat, p = friedman_ranks(pv)
    vs = vs_reference(pv, ref, delta).set_index("policy")
    frac = (sets.assign(frac=sets.k / sets.p).groupby("policy").frac.median())
    cost = sets.groupby("policy").t_select.mean()
    fcols = [c for c in ("fpr_rand", "fpr_shad", "fpr_block", "fpr_ko") if c in sets]
    fpr = sets.groupby("policy")[fcols].mean()
    stab = stability_table(sets).groupby("policy").stability.mean()
    out = pd.DataFrame({"rank_mean": ranks, f"{metric}_mean": pv.mean(), "frac_feat_median": frac,
                        "t_select_s": cost, "stability": stab}).join(fpr).join(
        vs[["mean_diff", "wins", "losses", "p_holm", "noninferior"]])
    out = out[out[f"{metric}_mean"].notna()]
    out.attrs.update(friedman_stat=stat, friedman_p=p, n_datasets=len(pv), cd=nemenyi_cd(pv.shape[1], len(pv)))
    return out.sort_values("rank_mean")


def truth_table(sets):
    s = sets.dropna(subset=["prec_signal"]) if "prec_signal" in sets else pd.DataFrame()
    if s.empty:
        return s
    cols = [c for c in ("k", "fdp", "prec_signal", "rec_inf", "rec_signal", "fpr_rand", "fpr_shad", "fpr_block", "fpr_ko")
            if c in s]
    return s.groupby("policy")[cols].mean()


# ---------------------------------------------------------------- extensiones (limitaciones 1-6)
def bootstrap_ci(d, B=10000, stat=np.mean, alpha=0.05, seed=0):
    """IC percentil por bootstrap sobre datasets (la unidad de inferencia) de un estadístico de las diferencias d."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([stat(d[rng.integers(0, len(d), len(d))]) for _ in range(B)])
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def diff_table(pv, ref, B=5000):
    """Δ vs ref por política con IC95% bootstrap de la media y de la mediana."""
    rows = []
    for c in pv.columns:
        if c == ref:
            continue
        d = (pv[c] - pv[ref]).values
        lo, hi = bootstrap_ci(d, B)
        mlo, mhi = bootstrap_ci(d, B, np.median)
        rows.append(dict(policy=c, n=len(d), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi,
                         median_diff=np.median(d), med_lo=mlo, med_hi=mhi))
    return pd.DataFrame(rows).set_index("policy").sort_values("mean_diff", ascending=False)


def delta_sensitivity(pv, ref="ALL", deltas=(0.0, 0.005, 0.01, 0.02)):
    """Qué políticas son no-inferiores a ref para distintos márgenes δ (Wilcoxon unilateral + Holm por δ)."""
    out = {}
    for dl in deltas:
        v = vs_reference(pv, ref, dl).set_index("policy")
        out[f"noninf_d{dl}"] = v["noninferior"]
        out[f"p_d{dl}"] = v["p_noninf_holm"]
    return pd.DataFrame(out).sort_index()


def dataset_meta(out_dir):
    fs = glob.glob(os.path.join(out_dir, "meta_*.json"))
    return pd.DataFrame([pd.read_json(f, typ="series") for f in fs]).set_index("dataset") if fs else pd.DataFrame()


def meta_regression(res, sets, out_dir, model="xgb", policy="CTRL_VOTE_MAJ", ref="ALL", metric="auroc"):
    """EXPLORATORIO. ΔAUROC(policy - ref) ~ log n + log p + p/n + minoría + fracción de ruido estimada, con
    intercepto aleatorio por dataset sobre las observaciones por (rep, fold). Devuelve (tabla de coeficientes, n)."""
    import statsmodels.formula.api as smf
    meta = dataset_meta(out_dir)
    if meta.empty:
        return pd.DataFrame(), 0
    r = res[res.model == model]
    a = r[r.policy == policy].set_index(["dataset", "rep", "fold"])[metric]
    b = r[r.policy == ref].set_index(["dataset", "rep", "fold"])[metric]
    d = (a - b).dropna().rename("delta").reset_index()
    rej = sets[sets.policy == "REJECTED"].assign(noise_frac=lambda s: s.k / s.p).groupby("dataset").noise_frac.mean()
    d = d.join(meta, on="dataset").join(rej, on="dataset").dropna()
    if d.dataset.nunique() < 8:
        return pd.DataFrame(), d.dataset.nunique()
    d["log_n"], d["log_p"] = np.log(d.n), np.log(d.p)
    d["p_over_n"], d["minor"] = d.p / d.n, d.minority
    covs = [c for c in ("log_n", "log_p", "p_over_n", "minor", "noise_frac") if d[c].std() > 1e-9]
    for c in covs:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    try:
        m = smf.mixedlm("delta ~ " + " + ".join(covs), d, groups=d["dataset"]).fit(reml=True)
    except Exception:  # noqa: BLE001  (matriz singular / no convergencia)
        return pd.DataFrame(), d.dataset.nunique()
    ci = m.conf_int()
    t = pd.DataFrame({"coef": m.params, "se": m.bse, "ci_lo": ci[0], "ci_hi": ci[1], "p": m.pvalues})
    return t.drop(index="Group Var", errors="ignore"), d.dataset.nunique()


def fpr_vs_fdp(sets):
    """Calibración de los proxies de FPR (rand/shad/block/ko) frente al FDP real en sintéticos con verdad conocida.
    Spearman (sobre políticas x datasets) y error medio |proxy - FDP|."""
    s = sets.dropna(subset=["fdp"]) if "fdp" in sets else pd.DataFrame()
    s = s[~s.policy.isin(["ALL", "REJECTED"])] if len(s) else s
    rows = []
    for c in ("fpr_rand", "fpr_shad", "fpr_block", "fpr_ko"):
        if c not in s:
            continue
        g = s.groupby(["dataset", "policy"])[[c, "fdp"]].mean().dropna()
        if len(g) > 5:
            rho = stats.spearmanr(g[c], g["fdp"])
            rows.append(dict(proxy=c, n=len(g), spearman=rho.statistic, p=rho.pvalue,
                             mae=float((g[c] - g["fdp"]).abs().mean()), bias=float((g[c] - g["fdp"]).mean())))
    return pd.DataFrame(rows).set_index("proxy") if rows else pd.DataFrame()


def ablation_table(res, sets, model="xgb", ref="CTRL_VOTE_MAJ", metric="auroc"):
    """Δ de cada ablación del protocolo frente a la política completa `ref`, con IC bootstrap, k y FPR."""
    pol = [ref] + sorted(c for c in res.policy.unique() if c.startswith("ABL_"))
    pv = pivot(res, model, metric, pol)
    if ref not in pv.columns or pv.shape[1] < 2:
        return pd.DataFrame()
    t = diff_table(pv, ref)
    g = sets[sets.policy.isin(pol)].groupby("policy").agg(k=("k", "mean"), fpr_shad=("fpr_shad", "mean"))
    return t.join(g)


def default_vs_tuned(res, base_models=("logreg", "knn", "xgb", "mlp"), ref="ALL", metric="auroc", policies=None):
    """Para cada política, Δ vs ref con el modelo por defecto y con su versión afinada (solo datasets/folds comunes)."""
    rows = []
    for m in base_models:
        mt = m + "_t"
        if mt not in set(res.model):
            continue
        tuned = pivot(res, mt, metric, policies, min_cov=0.0)
        if ref not in tuned.columns:
            continue
        default = pivot(res, m, metric, list(tuned.columns), min_cov=0.0)
        common = tuned.index.intersection(default.index)
        for c in tuned.columns:
            if c == ref:
                continue
            dd = (default.loc[common, c] - default.loc[common, ref]).values
            dt = (tuned.loc[common, c] - tuned.loc[common, ref]).values
            rows.append(dict(model=m, policy=c, n=len(common), d_default=dd.mean(), d_tuned=dt.mean(),
                             gap=dt.mean() - dd.mean(), p_wilcoxon_diff=(
                                 stats.wilcoxon(dt - dd).pvalue if np.any(dt != dd) and len(common) > 5 else np.nan)))
    return pd.DataFrame(rows)


def maxn_sensitivity(res_sub, res_full, model="xgb", ref="ALL", metric="auroc", policies=None):
    """Compara Δ(política - ref) con submuestreo a max_n pequeño (res_sub) vs max_n completo (res_full), mismos datasets."""
    a = pivot(res_sub, model, metric, policies, min_cov=0.0)
    b = pivot(res_full, model, metric, policies, min_cov=0.0)
    common = a.index.intersection(b.index)
    rows = []
    for c in a.columns.intersection(b.columns):
        if c == ref or ref not in a or ref not in b:
            continue
        da = (a.loc[common, c] - a.loc[common, ref]).values
        db = (b.loc[common, c] - b.loc[common, ref]).values
        rows.append(dict(policy=c, n_datasets=len(common), d_sub=da.mean(), d_full=db.mean(), gap=da.mean() - db.mean(),
                         spearman_ds=stats.spearmanr(da, db).statistic if len(common) > 3 else np.nan))
    return pd.DataFrame(rows).set_index("policy") if rows else pd.DataFrame()
