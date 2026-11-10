"""Dataset registry (OpenML) and synthetic generator with known ground truth."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

# Cache of downloaded datasets: override with DECOYFS_CACHE (e.g. to reuse an existing cache).
CACHE = os.environ.get("DECOYFS_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "decoyfs")

# short name -> OpenML data_id (verified). Tier order: smoke ⊂ laptop ⊂ full.
REAL = {
    # --- tabulares clásicos (p pequeño/mediano)
    "wdbc": 1510, "diabetes": 37, "credit-g": 31, "ilpd": 1480, "kc1": 1067, "pc4": 1049,
    "climate": 1467, "steel-plates": 1504, "ozone": 1487, "spambase": 44, "qsar-biodeg": 1494,
    "vehicle": 54, "segment": 36, "satimage": 182, "optdigits": 28, "texture": 40499,
    "magic": 1120, "eeg-eye": 1471, "wall-robot": 1497,
    # --- con atributos-ruido conocidos (AutoML/NIPS-2003)
    "madelon": 1485, "philippine": 41145, "madeline": 41144, "sylvine": 41146, "jasmine": 41143,
    "christine": 41142, "gina": 1038,
    # --- alta dimensión
    "cnae-9": 1468, "har": 1478, "isolet": 300, "bioresponse": 4134, "arcene": 1458,
    "internet-ads": 40978, "micro-mass": 1514, "mfeat-factors": 12, "dna": 40670,
    # --- grandes en n
    "bank": 1461, "nomao": 1486, "electricity": 151,
}

# Extensión de validez externa: 20 datasets de OpenML-CC18 (estudio 99) no incluidos arriba, con n<=50k, p<=~1.6k,
# >=10 atributos, sin duplicados de REAL (segment/steel-plates/climate ya están bajo otros ids).
REAL_EXT = {
    "vowel": 307, "adult": 1590, "letter": 6, "pendigits": 32, "eucalyptus": 188, "churn": 40701,
    "jm1": 1053, "pc1": 1068, "sick": 38, "phishing": 4534, "gesture": 4538, "kr-vs-kp": 3,
    "cylinder-bands": 6332, "mfeat-zernike": 22, "first-order-tp": 1475, "splice": 46,
    "authorship": 458, "mice-protein": 40966, "semeion": 1501, "mfeat-fourier": 14,
}
REAL.update(REAL_EXT)

# (n, p, n_inf, n_red, clusters, heavy_tail, minority_frac, label_noise)
SYN = {
    "syn01": (1000, 50, 5, 5, 1, False, 0.5, 0.01),
    "syn02": (1000, 50, 5, 5, 2, True, 0.5, 0.01),
    "syn03": (2000, 200, 10, 10, 2, False, 0.5, 0.01),
    "syn04": (2000, 200, 10, 10, 2, True, 0.2, 0.03),
    "syn05": (500, 500, 10, 10, 2, False, 0.5, 0.01),
    "syn06": (500, 500, 10, 10, 2, True, 0.1, 0.03),
    "syn07": (5000, 100, 8, 8, 3, True, 0.1, 0.01),
    "syn08": (5000, 1000, 20, 20, 2, False, 0.5, 0.01),
    "syn09": (200, 1000, 5, 5, 1, False, 0.5, 0.0),
    "syn10": (200, 1000, 5, 5, 1, True, 0.5, 0.0),
    "syn11": (3000, 300, 15, 15, 4, True, 0.3, 0.05),
    "syn12": (1500, 30, 6, 0, 2, False, 0.5, 0.0),
}

# Sintéticos dependientes (bloques correlacionados, XOR, proxies): truth 2 causal, 1 proxy, 0 ruido (independiente o
# en bloques). kind: "block" (factor por bloques), "xor" (interacciones sin señal marginal), "proxy".
# dict(n, p, kind, block_size, rho, n_causal, proxies_per_causal, minority, label_noise, heavy)
SYN2 = {
    "syn13": dict(n=1000, p=200, kind="block", block_size=10, rho=0.6, n_causal=8, minority=0.4, noise=0.02, heavy=False),
    "syn14": dict(n=300, p=1000, kind="block", block_size=20, rho=0.8, n_causal=6, minority=0.5, noise=0.02, heavy=True),
    "syn15": dict(n=2000, p=100, kind="xor", n_xor=3, n_lin=2, minority=0.5, noise=0.02, heavy=False),
    "syn16": dict(n=2000, p=100, kind="proxy", n_causal=10, proxies_per_causal=2, rho=0.7, noise_block=10,
                  minority=0.3, noise=0.03, heavy=True),
}
SYN_ALL = list(SYN) + list(SYN2)

TIERS = {
    "smoke": ["wdbc", "qsar-biodeg", "syn01", "syn02"],
    "laptop": ["wdbc", "diabetes", "credit-g", "ilpd", "kc1", "pc4", "climate", "steel-plates", "ozone",
               "spambase", "qsar-biodeg", "vehicle", "segment", "satimage", "optdigits", "texture",
               "madelon", "philippine", "madeline", "sylvine", "jasmine", "gina", "cnae-9", "magic",
               ] + list(SYN),
    "full": list(REAL) + SYN_ALL,
}
# datasets grandes en n para el análisis de sensibilidad a max_n (se corren aparte, ver notebook)
MAXN_SENS = ["bank", "nomao", "electricity", "magic", "letter", "eeg-eye", "adult", "gina", "jm1"]


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # float32, puede tener NaN
    y: np.ndarray                      # int 0..k-1
    feature_names: list
    truth: np.ndarray | None = None    # 0 ruido, 1 redundante, 2 informativo (solo sintéticos)
    meta: dict = field(default_factory=dict)


def _encode(df):
    """Binarias -> 0/1; categóricas de 3..12 niveles -> one-hot; resto -> códigos ordinales."""
    import pandas as pd
    cols, names = [], []
    for c in df.columns:
        s = df[c]
        if s.dtype.name in ("category", "object", "bool", "string"):
            s = s.astype("category")
            cats = list(s.cat.categories)
            codes = s.cat.codes.to_numpy().astype("float32")
            codes[codes < 0] = np.nan
            if len(cats) <= 2 or len(cats) > 12:
                cols.append(codes); names.append(str(c))
            else:
                for lv in cats:
                    cols.append((s == lv).to_numpy().astype("float32")); names.append(f"{c}={lv}")
        else:
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy().astype("float32")); names.append(str(c))
    return np.column_stack(cols), names


def _load_real(name):
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return z["X"], z["y"], list(z["names"])
    from sklearn.datasets import fetch_openml
    d = fetch_openml(data_id=REAL[name], as_frame=True, parser="auto")
    X, names = _encode(d.data)
    classes, y = np.unique(d.target.astype(str).to_numpy(), return_inverse=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, names=np.array(names, dtype=object))
    return X, y, names


def make_synthetic(name, seed=0):
    from sklearn.datasets import make_classification
    n, p, n_inf, n_red, clus, heavy, minority, noise = SYN[name]
    rs = int(name[3:]) * 1000 + seed
    rng = np.random.default_rng(rs)
    X, y = make_classification(n_samples=n, n_features=p, n_informative=n_inf, n_redundant=n_red,
                               n_repeated=0, n_classes=2, n_clusters_per_class=clus,
                               weights=[1 - minority, minority], flip_y=noise, class_sep=0.9,
                               shuffle=False, random_state=rs)
    truth = np.zeros(p, int)
    truth[:n_inf] = 2
    truth[n_inf:n_inf + n_red] = 1
    return _finish(name, X, y, truth, rng, heavy, SYN[name])


def _finish(name, X, y, truth, rng, heavy, cfg):
    """Permuta columnas, escalas heterogéneas, colas pesadas y ruido discreto (común a todos los sintéticos)."""
    p = X.shape[1]
    perm = rng.permutation(p)
    X, truth = X[:, perm].astype("float64"), truth[perm]
    # escalas heterogéneas (provoca el artefacto de escala observado en ReliefF/MI sin normalizar)
    X = X * 10 ** rng.uniform(-1, 2, p)
    if heavy:
        cols = rng.choice(p, size=int(0.4 * p), replace=False)
        for j in cols:
            z = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-12)
            X[:, j] = np.sinh(1.8 * z)          # colas pesadas, transformación monótona
    disc = rng.choice(np.where(truth == 0)[0], size=max(1, int(0.2 * (truth == 0).sum())), replace=False)
    for j in disc:                                 # ruido discreto (empates / baja cardinalidad)
        X[:, j] = np.floor(np.clip((X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-12) * 2, -4, 4))
    names = [f"f{j}" for j in range(p)]
    return Dataset(name, X.astype("float32"), y.astype(int), names, truth,
                   dict(kind="synthetic", cfg=cfg))


def _block_noise(n, p, bs, rho, rng):
    """p columnas N(0,1) agrupadas en bloques de tamaño bs con correlación intra-bloque rho (modelo de un factor)."""
    X = rng.normal(size=(n, p))
    for s in range(0, p, bs):
        f = rng.normal(size=(n, 1))
        X[:, s:s + bs] = np.sqrt(rho) * f + np.sqrt(1 - rho) * X[:, s:s + bs]
    return X


def _label(score, minority, noise, rng):
    s = score + rng.logistic(size=len(score)) * 0.5
    y = (s > np.quantile(s, 1 - minority)).astype(int)
    flip = rng.random(len(y)) < noise
    return np.where(flip, 1 - y, y)


def make_synthetic2(name, seed=0):
    c = SYN2[name]
    rs = int(name[3:]) * 1000 + seed
    rng = np.random.default_rng(rs)
    n, p = c["n"], c["p"]
    truth = np.zeros(p, int)
    if c["kind"] == "block":
        bs = c["block_size"]
        X = _block_noise(n, p, bs, c["rho"], rng)
        blocks = rng.choice(p // bs, size=c["n_causal"], replace=False)
        causal = blocks * bs                                  # primer miembro de cada bloque elegido
        for b in blocks:
            truth[b * bs:(b + 1) * bs] = 1                    # el resto del bloque: proxies correlacionados
        truth[causal] = 2
        w = rng.choice([-1.0, 1.0], size=len(causal)) * rng.uniform(0.7, 1.3, len(causal))
        score = X[:, causal] @ w
        score = score + 0.5 * np.tanh(X[:, causal[0]] * X[:, causal[1]])
    elif c["kind"] == "xor":
        X = rng.normal(size=(n, p))
        npair = c["n_xor"]
        for j in range(npair):
            truth[2 * j:2 * j + 2] = 2
        lin = np.arange(2 * npair, 2 * npair + c["n_lin"])
        truth[lin] = 2
        score = sum(2.0 * X[:, 2 * j] * X[:, 2 * j + 1] for j in range(npair)) + X[:, lin].sum(1)
    elif c["kind"] == "proxy":
        n_c, ppc = c["n_causal"], c["proxies_per_causal"]
        X = rng.normal(size=(n, p))
        rho = c["rho"]
        for j in range(n_c):
            for q in range(ppc):
                col = n_c + j * ppc + q
                X[:, col] = rho * X[:, j] + np.sqrt(1 - rho ** 2) * X[:, col]
                truth[col] = 1
        truth[:n_c] = 2
        s0 = n_c * (1 + ppc)
        X[:, s0:] = _block_noise(n, p - s0, c["noise_block"], 0.5, rng)   # ruido que se agrupa en bloques
        w = rng.choice([-1.0, 1.0], size=n_c) * rng.uniform(0.7, 1.3, n_c)
        score = np.tanh(X[:, :n_c]) @ w
    else:
        raise ValueError(c["kind"])
    score = (score - score.mean()) / (score.std() + 1e-12) * 1.5
    y = _label(score, c["minority"], c["noise"], rng)
    return _finish(name, X, y, truth, rng, c["heavy"], c)


def load_dataset(name, max_n=None, seed=0, min_class=10):
    if name in SYN2:
        ds = make_synthetic2(name, seed)
    elif name in SYN:
        ds = make_synthetic(name, seed)
    else:
        X, y, names = _load_real(name)
        ds = Dataset(name, X, y, names, None, dict(kind="real"))
    # clases demasiado pequeñas para CV estratificada
    cnt = np.bincount(ds.y)
    keep_c = np.where(cnt >= min_class)[0]
    if len(keep_c) < len(cnt):
        m = np.isin(ds.y, keep_c)
        remap = {c: i for i, c in enumerate(keep_c)}
        ds.X, ds.y = ds.X[m], np.array([remap[v] for v in ds.y[m]])
    if max_n and len(ds.y) > max_n:
        from sklearn.model_selection import train_test_split
        idx, _ = train_test_split(np.arange(len(ds.y)), train_size=max_n, stratify=ds.y, random_state=seed)
        idx = np.sort(idx)
        ds.X, ds.y = ds.X[idx], ds.y[idx]
    ds.meta.update(n=int(ds.X.shape[0]), p=int(ds.X.shape[1]), k=int(ds.y.max() + 1),
                   minority=float(np.bincount(ds.y).min() / len(ds.y)))
    return ds


def describe(names):
    rows = []
    for n in names:
        d = load_dataset(n)
        rows.append(dict(dataset=n, **d.meta))
    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(describe(list(REAL)[:3]).to_string())
    json.dumps(TIERS)
