"""Figuras estándar del protocolo."""
import matplotlib.pyplot as plt
import numpy as np


def cd_diagram(ranks, cd, ax=None, title="Rangos medios (menor = mejor)"):
    """Diagrama de diferencia crítica simplificado: barra de rangos y grupos no distinguibles (Nemenyi)."""
    r = ranks.sort_values()
    fig, ax = (plt.subplots(figsize=(9, 0.35 * len(r) + 1.5)) if ax is None else (ax.figure, ax))
    y = np.arange(len(r))[::-1]
    ax.hlines(y, 0, r.values, color="#9ecae1")
    ax.plot(r.values, y, "o", color="#08519c")
    best = r.iloc[0]
    for yi, (n, v) in zip(y, r.items()):
        ax.text(v + 0.1, yi, f"{n} ({v:.2f})", va="center", fontsize=8,
                color="#08519c" if v - best <= cd else "#444")
    ax.axvline(best + cd, ls="--", c="r", lw=0.8)
    ax.text(best + cd, y.max() + 0.6, f"mejor + CD={cd:.2f}", color="r", fontsize=8, ha="center")
    ax.set_yticks([]); ax.set_xlabel("rango medio"); ax.set_title(title)
    return fig


def pareto(tab, metric="auroc_mean", ax=None, ref="ALL"):
    """Rendimiento vs fracción de atributos retenida."""
    fig, ax = (plt.subplots(figsize=(7, 5)) if ax is None else (ax.figure, ax))
    for n, r in tab.iterrows():
        ctrl = n.startswith("CTRL")
        ax.scatter(r.frac_feat_median, r[metric], c="#d62728" if ctrl else "#1f77b4", s=45 if ctrl else 30)
        ax.annotate(n, (r.frac_feat_median, r[metric]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("fracción mediana de atributos retenida"); ax.set_ylabel(metric)
    ax.set_title("Frontera rendimiento vs parsimonia (rojo = referenciadas a controles)")
    return fig


def fpr_bars(sets, ax=None):
    g = sets.groupby("policy")[["fpr_rand", "fpr_shad"]].mean().sort_values("fpr_shad")
    fig, ax = (plt.subplots(figsize=(9, 4)) if ax is None else (ax.figure, ax))
    g.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("tasa de controles que 'pasan' (FPR estimada)")
    ax.set_title("Calibración de falsos positivos con Random / Shadow")
    ax.tick_params(axis="x", labelsize=7)
    return fig
