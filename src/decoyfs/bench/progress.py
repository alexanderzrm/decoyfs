"""Progreso y tiempo restante del experimento completo (todas las corridas: principal + sensibilidad a max_n).

El costo de cada fold pendiente se predice con los tiempos ya medidos (results/<dir>/timing.csv) mediante una
regresión ridge sobre log(n_train), log(p), log(k) y si es la repetición 0 (que incluye modelos afinados y RealMLP).
Con pocos datos se usa la mediana. Es una estimación: mejora conforme avanza la corrida.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
_MES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def fmt_dur(s):
    s = int(max(0, round(s)))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    if d:
        return f"{d} d {h} h {m:02d} min"
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min" if m else f"{s} s"


def fmt_dt(t):
    return f"{_DIAS[t.weekday()]} {t.day:02d}-{_MES[t.month - 1]}-{t.year} {t:%H:%M}"


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except Exception:  # noqa: BLE001
        return False


class Tracker:
    """jobs: lista de (nombres_de_datasets, out_dir, cfg, tag). tag='sens' marca corridas ligeras (sin extensiones)."""

    def __init__(self, jobs, log=print, ui=None, sens_prior=0.35):
        self.log, self.sens_prior = log, sens_prior
        self.ui = _in_notebook() if ui is None else ui
        self.jobs = [dict(names=list(n), out=os.path.abspath(o), cfg=c, tag=t) for n, o, c, t in jobs]
        self.t_start = None
        self.current = ""
        self._handle = None
        self._model_cache = None
        self._build()

    # ---------- construcción
    def _dataset_meta(self):
        from .data import load_dataset
        names = sorted({n for j in self.jobs for n in j["names"]})
        root = os.path.dirname(self.jobs[0]["out"])
        cache_f = os.path.join(root, "_dsmeta.json")
        cache = json.load(open(cache_f, encoding="utf-8")) if os.path.exists(cache_f) else {}
        new = [n for n in names if n not in cache]
        if new:
            self.log(f"Preparando metadatos de {len(new)} datasets para estimar el avance (una sola vez)…")
        for n in new:
            try:
                ds = load_dataset(n, max_n=None, seed=self.jobs[0]["cfg"].get("seed", 42))
                cache[n] = dict(n=int(ds.meta["n"]), p=int(ds.meta["p"]), k=int(ds.meta["k"]))
            except Exception as e:  # noqa: BLE001
                self.log(f"   ! sin metadatos de {n}: {repr(e)[:80]}")
        if new:
            os.makedirs(root, exist_ok=True)
            json.dump(cache, open(cache_f, "w", encoding="utf-8"))
        return cache

    def _build(self):
        meta = self._dataset_meta()
        med = {c: float(np.median([m[c] for m in meta.values()])) if meta else 1.0 for c in ("n", "p", "k")}
        self.items, self.index = [], {}
        for ji, j in enumerate(self.jobs):
            cfg = j["cfg"]
            for name in j["names"]:
                m = meta.get(name, med)
                n_tr = 0.8 * min(m["n"], cfg.get("max_n") or m["n"])
                feat = (np.log(max(n_tr, 2)), np.log(max(m["p"], 1)), np.log(max(m["k"], 2)))
                done = self._read_done(j["out"], name)
                for rep in range(cfg["n_repeats"]):
                    for fold in range(cfg["n_splits"]):
                        it = dict(job=ji, name=name, rep=rep, fold=fold, feat=feat, secs=None,
                                  done=(rep, fold) in done, light=j["tag"] == "sens")
                        self.items.append(it)
                        self.index[(ji, name, rep, fold)] = it
            f_t = os.path.join(j["out"], "timing.csv")
            if os.path.exists(f_t):
                for r in pd.read_csv(f_t).itertuples():
                    it = self.index.get((ji, r.dataset, int(r.rep), int(r.fold)))
                    if it is not None:
                        it["secs"], it["done"] = float(r.secs), True

    @staticmethod
    def _read_done(out, name):
        f = os.path.join(out, f"sets_{name}.csv")
        if not os.path.exists(f):
            return set()
        d = pd.read_csv(f, usecols=["rep", "fold"]).drop_duplicates()
        return set(zip(d.rep, d.fold))

    # ---------- modelo de costo
    @staticmethod
    def _x(it):
        return np.array([*it["feat"], 1.0 if it["rep"] == 0 else 0.0])

    def _fit(self):
        rows = [it for it in self.items if it["secs"] and not it["light"]]
        sens = [it for it in self.items if it["secs"] and it["light"]]
        self.n_timed = len(rows) + len(sens)
        mdl = dict(kind="none", sd=0.5, ratio=self.sens_prior)
        if rows:
            y = np.log([it["secs"] for it in rows])
            X = np.array([self._x(it) for it in rows])
            n_ds = len({it["name"] for it in rows})
            lo, hi = np.exp(y.min()) / 3, np.exp(y.max()) * 6
            if len(rows) >= 12 and n_ds >= 5:
                mu, sd_x = X.mean(0), X.std(0)
                sd_x[sd_x == 0] = 1.0
                Xs = (X - mu) / sd_x
                ym = y.mean()
                w = np.linalg.solve(Xs.T @ Xs + 2.0 * np.eye(Xs.shape[1]), Xs.T @ (y - ym))
                res = y - (ym + Xs @ w)
                mdl = dict(kind="regresión", mu=mu, sd_x=sd_x, ym=ym, w=w, lo=lo, hi=hi,
                           sd=max(0.15, 1.25 * float(res.std())), ratio=self.sens_prior)
            else:
                r0 = [np.log(it["secs"]) for it in rows if it["rep"] == 0]
                r1 = [np.log(it["secs"]) for it in rows if it["rep"] != 0]
                m0 = np.median(r0) if r0 else (np.median(r1) + np.log(2.0))
                m1 = np.median(r1) if r1 else (np.median(r0) - np.log(2.0))
                mdl = dict(kind="mediana", m0=m0, m1=m1, lo=lo, hi=hi, sd=0.5, ratio=self.sens_prior)
        self._model_cache = mdl
        if sens and mdl["kind"] != "none":
            ratios = [it["secs"] / self._base(it) for it in sens if self._base(it) > 0]
            if len(ratios) >= 3:
                mdl["ratio"] = float(np.median(ratios))

    def _base(self, it):
        m = self._model_cache
        if m["kind"] == "none":
            return 0.0
        if m["kind"] == "mediana":
            v = np.exp(m["m0"] if it["rep"] == 0 else m["m1"])
        else:
            v = np.exp(m["ym"] + ((self._x(it) - m["mu"]) / m["sd_x"]) @ m["w"])
        return float(np.clip(v, m["lo"], m["hi"]))

    def _pred(self, it):
        b = self._base(it)
        return b * self._model_cache["ratio"] if it["light"] else b

    # ---------- estado
    def stats(self):
        self._fit()
        done = [it for it in self.items if it["done"]]
        rem = [it for it in self.items if not it["done"]]
        w_done = sum(it["secs"] if it["secs"] else self._pred(it) for it in done)
        w_rem = sum(self._pred(it) for it in rem)
        n_ds_rem = len({(it["job"], it["name"]) for it in rem})
        s = max(0.08, min(0.6, self._model_cache["sd"] / np.sqrt(max(1, n_ds_rem))) * 1.5)
        tot_ds = len({(it["job"], it["name"]) for it in self.items})
        fin_ds = tot_ds - n_ds_rem
        return dict(n_done=len(done), n_total=len(self.items), w_done=w_done, w_rem=w_rem,
                    pct=100.0 * w_done / (w_done + w_rem) if (w_done + w_rem) > 0 else 0.0,
                    lo=w_rem * np.exp(-s), hi=w_rem * np.exp(s), ds_done=fin_ds, ds_total=tot_ds,
                    kind=self._model_cache["kind"], timed=self.n_timed)

    def short(self):
        st = self.stats()
        if st["kind"] == "none":
            return f"[{st['n_done']}/{st['n_total']} folds · calculando estimación…]"
        fin = dt.datetime.now() + dt.timedelta(seconds=st["w_rem"])
        return f"[{st['pct']:.1f}% · quedan {fmt_dur(st['w_rem'])} · fin {fmt_dt(fin)}]"

    def panel(self):
        st = self.stats()
        bar_n = 30
        fill = int(round(bar_n * st["pct"] / 100))
        bar = "█" * fill + "░" * (bar_n - fill)
        el = (dt.datetime.now() - self.t_start).total_seconds() if self.t_start else 0
        lines = [f"PROGRESO  [{bar}] {st['pct']:5.1f} %   ({st['n_done']}/{st['n_total']} folds · "
                 f"{st['ds_done']}/{st['ds_total']} datasets-corrida)",
                 f"Ahora: {self.current or '—'}   |   transcurrido en esta sesión: {fmt_dur(el)}"]
        if st["kind"] == "none":
            lines.append("Tiempo restante: calculando (se necesita al menos un fold terminado)…")
        else:
            now = dt.datetime.now()
            fin = now + dt.timedelta(seconds=st["w_rem"])
            f_lo, f_hi = now + dt.timedelta(seconds=st["lo"]), now + dt.timedelta(seconds=st["hi"])
            lines += [f"Tiempo restante estimado: {fmt_dur(st['w_rem'])}   (rango {fmt_dur(st['lo'])} – {fmt_dur(st['hi'])})",
                      f"Conclusión estimada: {fmt_dt(fin)}   (rango {fmt_dt(f_lo)} – {fmt_dt(f_hi)})",
                      f"Base: {st['timed']} folds con tiempo medido · modelo de costo: {st['kind']}"]
        return "\n".join(lines)

    # ---------- interfaz
    def show(self):
        if not self.ui:
            return
        from IPython.display import HTML, display
        html = HTML("<pre style='font-family:Consolas,monospace;line-height:1.35;margin:0'>" + self.panel() + "</pre>")
        if self._handle is None:
            self._handle = display(html, display_id=True)
        else:
            self._handle.update(html)

    def begin(self, out_dir, name):
        if self.t_start is None:
            self.t_start = dt.datetime.now()
        self.current = name
        self.show()

    def new_panel(self):
        self._handle = None
        if self.t_start is None:
            self.t_start = dt.datetime.now()
        self.show()

    def fold_done(self, out_dir, name, rep, fold, secs):
        out = os.path.abspath(out_dir)
        for ji, j in enumerate(self.jobs):
            if j["out"] == out:
                it = self.index.get((ji, name, rep, fold))
                if it is not None:
                    it["secs"], it["done"] = float(secs), True
                break
        self.current = f"{name} (rep{rep} fold{fold})"
        self.show()
        return self.short()
