"""Bucle principal del benchmark: CV externa repetida; selección + controles SOLO dentro de train; reanudable."""
from __future__ import annotations

import json
import os
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .. import policies as P
from ..comparators import run_baselines
from ..controls import augment
from ..dependence import block_shadow, gaussian_knockoffs, leader_clusters
from ..device import device
from ..metrics import evaluate
from ..preprocessing import FoldPrep
from ..rankers import run_rankers
from .data import load_dataset
from .models import MODELS, ModelUnavailable

DEFAULT_CFG = dict(
    methods=["anova", "chi2q", "mi", "relieff", "rf", "xgb_gain", "l1lr"],
    models=["logreg", "knn", "xgb", "mlp"],
    n_splits=5, n_repeats=2, seed=42, alpha=0.05, ref_policy="CTRL_VOTE_MAJ",
    matched_methods=["anova", "mi", "relieff", "xgb_gain"],
    cvk=True, stab_B=0, stab_pi=0.6, stab_frac=0.8, max_n=20000,
    # --- extensiones (ver docs/DISENO_PROTOCOLO.md §Extensiones)
    ablations=False,                       # ablaciones del protocolo (costo ~0, usa los mismos puntajes)
    baselines=[],                          # subconjunto de ["BORUTA", "NABFS"]
    boruta_iter=25, boruta_budget=420.0, nabfs_boot=20, nabfs_noise=7,
    pass2=False,                           # Block-Shadow + knockoffs gaussianos (2ª pasada de rankers)
    block_thr=0.5, ko_pmax=2000, ko_q=0.10, ko_stats=("borda", "xgb_gain"),
    model_plan=[],                         # evaluaciones extra: dict(models, policies|None, reps|None, max_p|None)
)

UNAVAILABLE = set()                        # modelos opcionales que fallaron (paquete/licencia): se omiten sin abortar
OWN_TIME = ("CTRL_STAB", "BORUTA", "NABFS")


def _append(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _done(path):
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path, usecols=["rep", "fold"]).drop_duplicates()
    return set(zip(d.rep, d.fold))


def _one_run(Ztr, ytr, k, seed, cfg):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    Xa, idx = augment(Ztr, rng)
    t_ctrl = time.perf_counter() - t0
    scores, times, errors = run_rankers(Xa, ytr, k, seed, cfg["methods"])
    return Xa, idx, scores, times, errors, t_ctrl


def _pass2(Ztr, ytr, Xa, idx, k, seed, cfg):
    """2ª pasada con controles sensibles a la dependencia: [reales | Random | Block-Shadow | knockoffs]."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed + 555)
    n, p = Ztr.shape
    lab = leader_clusters(Ztr, cfg["block_thr"], seed=seed)
    reps = int(min(10, max(1, np.ceil(40 / p))))
    parts = [Ztr, Xa[:, idx["rand"]], block_shadow(Ztr, lab, reps, rng)]
    idx2 = dict(real=np.arange(p), rand=np.arange(p, p + len(idx["rand"])))
    o = p + len(idx["rand"])
    idx2["blk"] = np.arange(o, o + parts[2].shape[1])
    o += parts[2].shape[1]
    if p <= cfg["ko_pmax"]:
        Xk, _ = gaussian_knockoffs(Ztr, rng)
        parts.append(Xk)
        idx2["ko"] = np.arange(o, o + p)
    X2 = np.ascontiguousarray(np.concatenate(parts, axis=1))
    scores2, times2, errors2 = run_rankers(X2, ytr, k, seed + 555, cfg["methods"])
    c2 = P.consensus(scores2)
    extra = {"CTRL_BLOCK_VOTE_MAJ": (P.block_vote_set(scores2, idx2, cfg["alpha"]), 0.0)}
    if "ko" in idx2:
        for nm, sel in P.knockoff_stat_sets(c2, scores2, idx2, cfg["ko_q"], tuple(cfg["ko_stats"])).items():
            extra[nm] = (sel, 0.0)
    t2 = time.perf_counter() - t0
    extra = {nm: (s, t2) for nm, (s, _) in extra.items()}
    return extra, c2, idx2, errors2


def run_dataset(name, out_dir, cfg, log=print, tracker=None):
    cfg = {**DEFAULT_CFG, **cfg}
    ds = load_dataset(name, max_n=cfg["max_n"], seed=cfg["seed"])
    X, y, k = ds.X, ds.y, ds.meta["k"]
    f_sets = os.path.join(out_dir, f"sets_{name}.csv")
    f_res = os.path.join(out_dir, f"res_{name}.csv")
    f_feat = os.path.join(out_dir, f"feat_{name}.csv")
    done = _done(f_sets)
    log(f"[{name}] n={ds.meta['n']} p={ds.meta['p']} k={k} minority={ds.meta['minority']:.2f}  hechos={len(done)}")
    with open(os.path.join(out_dir, f"meta_{name}.json"), "w") as fh:
        json.dump({"dataset": name, "n": int(ds.meta["n"]), "p": int(ds.meta["p"]), "k": int(k),
                   "minority": float(ds.meta["minority"])}, fh)
    for rep in range(cfg["n_repeats"]):
        skf = StratifiedKFold(cfg["n_splits"], shuffle=True, random_state=cfg["seed"] + rep)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            if (rep, fold) in done:
                continue
            seed = cfg["seed"] + 1000 * rep + fold
            t_fold = time.perf_counter()
            try:
                prep = FoldPrep().fit(X[tr])
                Ztr, Zte = prep.transform(X[tr]), prep.transform(X[te])
                keep = prep.keep_
                ytr, yte = y[tr], y[te]
                Xa, idx, scores, times, errors, t_ctrl = _one_run(Ztr, ytr, k, seed, cfg)
                t_rank = t_ctrl + sum(times.values())
                run_stab = None
                if cfg["stab_B"] > 0:
                    cnt, t_st = np.zeros(len(keep)), 0.0
                    for b in range(cfg["stab_B"]):
                        rs = np.random.default_rng(seed + 31 * (b + 1))
                        sub = rs.choice(len(ytr), size=int(cfg["stab_frac"] * len(ytr)), replace=False)
                        t0 = time.perf_counter()
                        Xb, ib, sb, tb, _, tcb = _one_run(Ztr[sub], ytr[sub], k, seed + 31 * (b + 1), cfg)
                        sets_b, *_ = P.ctrl_sets(sb, ib, cfg["alpha"])
                        cnt[sets_b["CTRL_VOTE_MAJ"]] += 1
                        t_st += time.perf_counter() - t0
                    run_stab = (np.where(cnt / cfg["stab_B"] >= cfg["stab_pi"])[0], t_st)
                extra, c2, idx2 = {}, None, None
                if cfg["pass2"]:
                    try:
                        extra, c2, idx2, err2 = _pass2(Ztr, ytr, Xa, idx, k, seed, cfg)
                        if err2:
                            log(f"   ! pasada 2 con errores en rankers: {list(err2)}")
                    except Exception as e:  # noqa: BLE001
                        log(f"   ! pasada 2 omitida en {name} rep{rep} fold{fold}: {repr(e)[:160]}")
                for bn, (sel, t) in run_baselines(Ztr, ytr, seed, cfg).items():
                    if bn.endswith("_ERR"):
                        log(f"   ! baseline {bn[:-4]} falló: {t}")
                    else:
                        extra[bn] = (sel, t)
                pol, c, votes, fl = P.build_policies(Ztr, ytr, scores, idx, cfg, run_stab, seed, extra=extra)
                rep_f = P.feature_report(c, votes, fl, idx, len(scores))

                # ---- tabla de selecciones
                srows = []
                for pn, (sel, extra_t, fell) in pol.items():
                    if pn == "ALL":
                        ts = 0.0
                    elif pn.startswith("TOPK_MATCH_"):
                        ts = times.get(pn[len("TOPK_MATCH_"):], 0.0)
                    elif pn in OWN_TIME:
                        ts = extra_t
                    else:
                        ts = t_rank + extra_t
                    fr, fs = P.null_pass_rates(sel, c, idx) if len(sel) else (np.nan, np.nan)
                    row = dict(dataset=name, rep=rep, fold=fold, policy=pn, k=len(sel), p=len(keep),
                               fallback=fell, t_select=ts, fpr_rand=fr, fpr_shad=fs,
                               fpr_block=P.null_rate(sel, c2, idx2, "blk") if c2 is not None else np.nan,
                               fpr_ko=P.null_rate(sel, c2, idx2, "ko") if c2 is not None else np.nan,
                               selected=",".join(map(str, keep[sel])))
                    if ds.truth is not None and len(sel):
                        tt = ds.truth[keep[sel]]
                        sig = (ds.truth > 0)
                        row.update(prec_signal=float((tt > 0).mean()), fdp=float((tt == 0).mean()),
                                   rec_inf=float((tt == 2).sum() / max(1, (ds.truth == 2).sum())),
                                   rec_signal=float((tt > 0).sum() / max(1, sig.sum())))
                    srows.append(row)

                # ---- evaluación: plan principal (todos los modelos/políticas) + planes de sensibilidad
                plans = [dict(models=cfg["models"], policies=None, reps=None, max_p=None)] + list(cfg["model_plan"])
                rrows = []
                for plan in plans:
                    if plan.get("reps") is not None and rep not in plan["reps"]:
                        continue
                    if plan.get("max_p") is not None and len(keep) > plan["max_p"]:
                        continue
                    names_ok = None if plan.get("policies") is None else set(plan["policies"])
                    uniq = {}
                    for pn, (sel, _, _) in pol.items():
                        if names_ok is None or pn in names_ok:
                            uniq.setdefault(tuple(sel.tolist()), []).append(pn)
                    for key, names in uniq.items():
                        cols = np.array(key, dtype=int)
                        for mn in plan["models"]:
                            if mn in UNAVAILABLE:
                                continue
                            try:
                                t0 = time.perf_counter()
                                mdl = MODELS[mn](seed).fit(Ztr[:, cols], ytr)
                                t_fit = time.perf_counter() - t0
                                t0 = time.perf_counter()
                                Pr = mdl.predict_proba(Zte[:, cols])
                                t_pred = time.perf_counter() - t0
                                met = evaluate(yte, Pr)
                            except ModelUnavailable as e:
                                UNAVAILABLE.add(mn)
                                log(f"   ! modelo opcional '{mn}' desactivado: {e}")
                                continue
                            except Exception as e:  # noqa: BLE001
                                met, t_fit, t_pred = dict(auroc=np.nan), np.nan, np.nan
                                log(f"   ! {name} {mn} {len(cols)} feats: {repr(e)[:100]}")
                            for pn in names:
                                rrows.append(dict(dataset=name, rep=rep, fold=fold, policy=pn, model=mn,
                                                  k=len(cols), t_fit=t_fit, t_pred=t_pred, **met))
                frows = [dict(dataset=name, rep=rep, fold=fold, feat=int(keep[j]), name=ds.feature_names[keep[j]],
                              consensus=float(rep_f["consensus"][j]), votes=int(rep_f["votes"][j]),
                              beat_rand=float(rep_f["beat_rand"][j]), beat_shad=float(rep_f["beat_shad"][j]))
                         for j in range(len(keep))]
                _append(f_res, rrows)
                _append(f_feat, frows)
                _append(f_sets, srows)       # el último: marca el fold como completado
                secs = time.perf_counter() - t_fold
                _append(os.path.join(out_dir, "timing.csv"), [dict(dataset=name, rep=rep, fold=fold, secs=round(secs, 2))])
                eta = ""
                if tracker is not None:
                    try:
                        eta = "  " + tracker.fold_done(out_dir, name, rep, fold, secs)
                    except Exception:  # noqa: BLE001
                        pass
                log(f"   rep{rep} fold{fold}: {secs:5.1f}s  rank={t_rank:4.1f}s "
                    f"k_ref={len(pol[cfg['ref_policy']][0])}/{len(keep)} err={list(errors)}{eta}")
            except Exception:  # noqa: BLE001
                log(f"   ✗ {name} rep{rep} fold{fold}\n{traceback.format_exc()[-600:]}")


def run_all(names, out_dir, cfg, log=print, tracker=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg=cfg, datasets=names, device=device()), f, indent=2, default=str)
    for n in names:
        if tracker is not None:
            try:
                tracker.begin(out_dir, n)
            except Exception:  # noqa: BLE001
                pass
        run_dataset(n, out_dir, cfg, log, tracker)
