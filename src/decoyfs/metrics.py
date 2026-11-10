"""Evaluation metrics and the selection-stability index used by the library and the benchmark."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score


def balanced_weights(y):
    """Per-sample weights inversely proportional to class frequency (``y`` holds integer codes)."""
    cnt = np.bincount(y)
    return (len(y) / (len(cnt) * cnt))[y]


def evaluate(y, P):
    """bacc, macro-F1, log-loss, AUROC (one-vs-rest macro for k > 2) and PR-AUC from class probabilities ``P``."""
    k = P.shape[1]
    P = np.clip(P, 1e-7, 1)
    P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1)
    res = dict(bacc=balanced_accuracy_score(y, pred), f1m=f1_score(y, pred, average="macro"))
    try:
        res["logloss"] = log_loss(y, P, labels=list(range(k)))
        if k == 2:
            res["auroc"] = roc_auc_score(y, P[:, 1])
            res["prauc"] = average_precision_score(y, P[:, 1])
        else:
            res["auroc"] = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(k)))
            Y = np.eye(k)[y]
            res["prauc"] = float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in range(k) if Y[:, j].sum() > 0]))
    except Exception:  # noqa: BLE001
        res.update(logloss=np.nan, auroc=np.nan, prauc=np.nan)
    return res


def nogueira_stability(Z):
    """Selection stability (Nogueira et al., 2018, JMLR). ``Z`` is an (M, p) binary matrix of M selections."""
    Z = np.asarray(Z, dtype=float)
    M, p = Z.shape
    if M < 2:
        return np.nan
    pf = Z.mean(0)
    kbar = Z.sum(1).mean()
    denom = (kbar / p) * (1 - kbar / p)
    if denom <= 0:
        return 1.0
    return float(1 - (M / (M - 1) * pf * (1 - pf)).mean() / denom)
