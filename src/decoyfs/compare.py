"""Head-to-head comparison of selectors under the same folds and the same downstream model."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .validation import check_X_y


def compare_selectors(X, y, selectors, model="logreg", cv=5, random_state=0, n_repeats=1):
    """Evaluate several selectors with stratified cross-validation; selection happens inside each training fold.

    Parameters
    ----------
    selectors : dict name -> selector (:class:`DecoySelector`, :class:`FixedKSelector`, ...). Include
        ``"all": None`` to add the no-selection reference.
    model : {"logreg", "knn", "xgb", "mlp"} or a class ``M(seed)`` with ``fit(X, y)`` and ``predict_proba(X)``.
    cv, n_repeats : folds and repetitions (different fold splits).

    Returns
    -------
    DataFrame with one row per (selector, repeat, fold): ``n_selected``, ``frac_selected``, ``select_time_s`` and the
    metrics ``auroc``, ``prauc``, ``bacc``, ``f1m``, ``logloss`` of the downstream model.
    Use ``df.groupby("selector").mean(numeric_only=True)`` for a summary.
    """
    from .bench.models import MODELS
    from .metrics import evaluate
    Xa, yc, _, _ = check_X_y(X, y)
    make = MODELS[model] if isinstance(model, str) else model
    if not selectors:
        raise ValueError("selectors must be a non-empty dict.")
    rows = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(cv, shuffle=True, random_state=random_state + rep)
        for fold, (tr, te) in enumerate(skf.split(Xa, yc)):
            for name, sel in selectors.items():
                if sel is None:
                    from .preprocessing import FoldPrep
                    pre = FoldPrep().fit(Xa[tr])
                    Ztr, Zte, n_sel, p, dt = pre.transform(Xa[tr]), pre.transform(Xa[te]), len(pre.keep_), Xa.shape[1], 0.0
                else:
                    s = clone(sel)
                    if "random_state" in s.get_params():
                        s.set_params(random_state=random_state + rep * 1000 + fold)
                    t0 = time.perf_counter()
                    s.fit(Xa[tr], yc[tr])
                    dt = time.perf_counter() - t0
                    pre = s.preprocessor_
                    local = np.where(s.support_[pre.keep_])[0]
                    Ztr, Zte = pre.transform(Xa[tr])[:, local], pre.transform(Xa[te])[:, local]
                    n_sel, p = int(s.support_.sum()), Xa.shape[1]
                m = make(random_state + fold).fit(Ztr, yc[tr])
                res = evaluate(yc[te], m.predict_proba(Zte))
                rows.append(dict(selector=name, repeat=rep, fold=fold, n_selected=n_sel, frac_selected=n_sel / p,
                                 select_time_s=dt, **res))
    return pd.DataFrame(rows)
