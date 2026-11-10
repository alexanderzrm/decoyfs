"""Downstream evaluation models (GPU when available). Metrics live in :mod:`decoyfs.metrics`."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ..device import device, xgb_device
from ..metrics import balanced_weights as _bal_weights
from ..metrics import evaluate, nogueira_stability  # noqa: F401  (re-exported for the benchmark)


class LogReg:
    def __init__(self, seed=0, C=0.5):
        from sklearn.linear_model import LogisticRegression
        self.m = LogisticRegression(C=C, max_iter=400, class_weight="balanced")

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class KNNGPU:
    def __init__(self, seed=0, k=15):
        self.k = k

    def fit(self, X, y):
        self.dev = device()
        self.Xt = torch.as_tensor(X, device=self.dev)
        self.yt = torch.as_tensor(y, dtype=torch.long, device=self.dev)
        self.nc = int(y.max()) + 1
        prior = np.bincount(y, minlength=self.nc) / len(y)
        self.w = torch.as_tensor(1.0 / np.maximum(prior, 1e-6), dtype=torch.float32, device=self.dev)
        return self

    def predict_proba(self, X):
        out, k = [], min(self.k, len(self.yt))
        T = torch.as_tensor(X, device=self.dev)
        step = max(64, int(2e7 // max(1, len(self.yt))))
        for s in range(0, len(T), step):
            d = torch.cdist(T[s:s + step], self.Xt)
            dv, di = torch.topk(d, k, dim=1, largest=False)
            wt = 1.0 / (dv + 1e-3)
            oh = torch.nn.functional.one_hot(self.yt[di], self.nc).float() * wt[..., None]
            pr = oh.sum(1) * self.w[None, :] + 1e-3
            out.append((pr / pr.sum(1, keepdim=True)).cpu().numpy())
        return np.concatenate(out).astype("float64")


class XGB:
    def __init__(self, seed=0, n_estimators=300, max_depth=5, learning_rate=0.08, colsample_bytree=0.8):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                   subsample=0.8, colsample_bytree=colsample_bytree, tree_method="hist",
                                   device=xgb_device(), random_state=seed, n_jobs=4, verbosity=0)

    def fit(self, X, y):
        self.m.fit(X, y, sample_weight=_bal_weights(y))
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class MLP:
    def __init__(self, seed=0, epochs=60, patience=8, hidden=(256, 128), dropout=0.2, wd=1e-4, lr=2e-3):
        self.seed, self.epochs, self.patience = seed, epochs, patience
        self.hidden, self.dropout, self.wd, self.lr = hidden, dropout, wd, lr

    def fit(self, X, y):
        dev = device()
        torch.manual_seed(self.seed)
        n, p = X.shape
        k = int(y.max()) + 1
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        nv = max(int(0.1 * n), 2)
        va, tr = perm[:nv], perm[nv:]
        Xt = torch.as_tensor(X, device=dev)
        yt = torch.as_tensor(y, dtype=torch.long, device=dev)
        h1, h2 = self.hidden
        self.net = torch.nn.Sequential(
            torch.nn.Linear(p, h1), torch.nn.ReLU(), torch.nn.Dropout(self.dropout),
            torch.nn.Linear(h1, h2), torch.nn.ReLU(), torch.nn.Dropout(self.dropout), torch.nn.Linear(h2, k)).to(dev)
        cnt = np.bincount(y[tr], minlength=k).clip(min=1)
        lw = torch.as_tensor(len(tr) / (k * cnt), dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = torch.nn.CrossEntropyLoss(weight=lw)
        tr_t = torch.as_tensor(tr, device=dev)
        va_t = torch.as_tensor(va, device=dev)
        bs = 256 if n < 5000 else 1024
        best, bad, state = 1e9, 0, None
        for _ in range(self.epochs):
            self.net.train()
            sh = tr_t[torch.randperm(len(tr_t), device=dev)]
            for s in range(0, len(sh), bs):
                b = sh[s:s + bs]
                opt.zero_grad()
                lossf(self.net(Xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = lossf(self.net(Xt[va_t]), yt[va_t]).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {a: v.clone() for a, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if state:
            self.net.load_state_dict(state)
        self.dev = dev
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            out = torch.softmax(self.net(torch.as_tensor(X, device=self.dev)), 1)
        return out.cpu().numpy().astype("float64")


class ModelUnavailable(RuntimeError):
    """Modelo opcional no utilizable (paquete/pesos/licencia): el runner lo desactiva y sigue."""


class RealMLP:
    """RealMLP-TD (Holzmüller et al., 2025) vía pytabkit. Opcional."""

    def __init__(self, seed=0, n_epochs=48, max_rows=10000):
        self.seed, self.n_epochs, self.max_rows = seed, n_epochs, max_rows

    def fit(self, X, y):
        try:
            from pytabkit import RealMLP_TD_Classifier
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"pytabkit no disponible: {repr(e)[:120]}") from e
        self._k = int(y.max()) + 1
        if len(y) > self.max_rows:
            sub = np.random.default_rng(self.seed).choice(len(y), self.max_rows, replace=False)
            X, y = X[sub], y[sub]
        self.m = RealMLP_TD_Classifier(device=device(), random_state=self.seed, n_epochs=self.n_epochs, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        P = self.m.predict_proba(X)
        return _full_proba(P, self.classes_, self._k)

    _k = None


def _full_proba(P, classes, k):
    if k is None or P.shape[1] == k:
        return P
    out = np.zeros((P.shape[0], k))
    out[:, classes] = P
    return out


class Tuned:
    """Ajusta hiperparámetros con una partición interna (<=4000 filas, estratificada) y reentrena con todo el train.

    Mide la sensibilidad de las conclusiones a los hiperparámetros fijos del protocolo por defecto.
    """
    GRIDS = {
        "logreg": (LogReg, [dict(C=c) for c in (0.05, 0.5, 5.0)]),
        "knn": (KNNGPU, [dict(k=k) for k in (5, 15, 40)]),
        "xgb": (XGB, [dict(n_estimators=200, max_depth=3, learning_rate=0.10),
                      dict(n_estimators=300, max_depth=5, learning_rate=0.08),
                      dict(n_estimators=400, max_depth=7, learning_rate=0.05),
                      dict(n_estimators=400, max_depth=4, learning_rate=0.05, colsample_bytree=0.5)]),
        "mlp": (MLP, [dict(hidden=(256, 128), dropout=0.2, wd=1e-4),
                      dict(hidden=(128, 64), dropout=0.3, wd=1e-3),
                      dict(hidden=(512, 256), dropout=0.4, wd=1e-4)]),
    }

    def __init__(self, base, seed=0, max_rows=4000):
        self.base, self.seed, self.max_rows = base, seed, max_rows

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        cls, grid = self.GRIDS[self.base]
        k = int(y.max()) + 1
        idx = np.arange(len(y))
        if len(y) > self.max_rows:
            idx, _ = train_test_split(idx, train_size=self.max_rows, stratify=y, random_state=self.seed)
        try:
            a, b = train_test_split(idx, test_size=0.25, stratify=y[idx], random_state=self.seed)
        except ValueError:
            a, b = train_test_split(idx, test_size=0.25, random_state=self.seed)
        best, best_kw = -np.inf, grid[0]
        for kw in grid:
            try:
                P = cls(self.seed, **kw).fit(X[a], y[a]).predict_proba(X[b])
                s = evaluate(y[b], P)
                s = -s["logloss"] if np.isfinite(s.get("logloss", np.nan)) else -np.inf
            except Exception:  # noqa: BLE001
                continue
            if s > best:
                best, best_kw = s, kw
        self.best_kw = best_kw
        self.m = cls(self.seed, **best_kw).fit(X, y)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


MODELS = {"logreg": LogReg, "knn": KNNGPU, "xgb": XGB, "mlp": MLP,
          "logreg_t": lambda s: Tuned("logreg", s), "knn_t": lambda s: Tuned("knn", s),
          "xgb_t": lambda s: Tuned("xgb", s), "mlp_t": lambda s: Tuned("mlp", s),
          "realmlp": RealMLP}
CORE_MODELS = ("logreg", "knn", "xgb", "mlp")
TUNED_MODELS = ("logreg_t", "knn_t", "xgb_t", "mlp_t")
FOUNDATION_MODELS = ("realmlp",)
