# DecoyFS

**Decoy-referenced feature selection** for tabular classification. DecoyFS keeps a feature only if it beats *decoys* — columns whose relationship with the target is known to be null — according to several heterogeneous rankers. The result is a small, auditable subset: for every feature you can see which rankers voted for it and how it compared with the decoys.

* **Random probes**: pure-noise columns. A feature must outscore the best probe.
* **Shadow features**: each real column independently permuted. A feature must outscore the `1 - alpha` quantile of its shadows.
* **Heterogeneous rankers** (ANOVA, chi², mutual information, ReliefF, random forest, XGBoost gain, L1 logistic regression) vote; a **policy** (`all`, `majority`, `any`, an integer or your own function) turns the votes into the subset.
* **Every block is swappable**: add or remove rankers, controls and policies. Boruta-style, CV-k and fixed-k comparators are bundled so you can compare on your own data.

> Honest scope. In our benchmark (74 datasets, 5×2 CV, 4 downstream models) DecoyFS is **non-inferior (margin 0.01 AUROC) to using all features and to the comparators**, keeps roughly half of the features (median), is **~10× cheaper than Boruta-style selection** (median time ratio), and is **auditable and tunable** (`policy` moves you along the precision–recall frontier). It does **not** beat Boruta in false-discovery proportion or parsimony; the empirical null-pass rate it reports is a proxy, not a formal FDR guarantee. See the paper for details.

## Install

```bash
pip install decoyfs            # core (numpy, scipy, pandas, scikit-learn, xgboost, torch)
pip install "decoyfs[bench]"   # + benchmark harness (OpenML, statsmodels, matplotlib, jupyter)
```

Python ≥ 3.10. A GPU is optional: if `torch` / `xgboost` can use CUDA they will, otherwise everything runs on CPU. For exactly pinned versions use `benchmarks/requirements-lock.txt` (tested on Python 3.11).

## Quick start

```python
from decoyfs import DecoySelector

sel = DecoySelector(policy="majority", alpha=0.05, random_state=0).fit(X_train, y_train)
X_small = sel.transform(X_test)

sel.selected_features_      # names (DataFrame input) or column indices
sel.report_                 # per-feature audit trail: votes, beat_random / beat_shadow fractions, per-ranker scores
sel.decoy_thresholds_       # the bar each ranker set (Random max, Shadow quantile)
sel.null_pass_rate_         # fraction of decoys that would have been selected (empirical FPR proxy)
```

It is a scikit-learn transformer, so it works in a `Pipeline`:

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
model = make_pipeline(DecoySelector(), LogisticRegression(max_iter=500)).fit(X_train, y_train)
```

### Choose your trade-off

| `policy` | Rule | Use when |
|---|---|---|
| `"all"` | every ranker must beat the decoys | precision first (few false features) |
| `"majority"` | at least half of the rankers (reference) | balanced default |
| `"any"` | at least one ranker | recall first, e.g. you must not lose a critical signal (screening, quality control) |
| `3` | at least 3 rankers | custom |
| callable | `(votes, n_rankers, consensus) -> indices` | anything else |

`stability_B=5` adds a subsampling stability filter; `controls=("random", "block")` replaces Shadow by Block-Shadow for strongly correlated features.

### Swap the pieces

```python
import numpy as np
from decoyfs import register_ranker, register_policy, DecoySelector, list_rankers

register_ranker("mean_gap", lambda X, y: np.abs(X[y == 1].mean(0) - X[y == 0].mean(0)))
register_policy("top10", lambda votes, M, consensus: np.argsort(-consensus)[:10])

DecoySelector(rankers=("anova", "rf", "mean_gap"), policy="top10").fit(X, y)
```

A ranker is any function `fn(X, y) -> one score per column` (higher = more relevant). See [docs/extending.md](docs/extending.md) and the block diagram in [docs/architecture.md](docs/architecture.md).

### Compare with other selectors on your data

```python
from decoyfs import DecoySelector, FixedKSelector, CVKSelector, BorutaSelector, compare_selectors

df = compare_selectors(X, y, {
    "all": None,
    "decoyfs": DecoySelector(),
    "fixed25": FixedKSelector(k=0.25),
    "cvk_1se": CVKSelector(),
    "boruta_style": BorutaSelector(),
}, model="xgb", cv=5)
df.groupby("selector").mean(numeric_only=True)
```

`BorutaSelector` is a **Boruta-style re-implementation** (XGBoost gain importance, shadow features, binomial test with Bonferroni correction), the one used in the paper. It is not the `BorutaPy` package.

## Input rules

Failing early beats failing silently. DecoyFS raises `DecoyInputError` with an actionable message when:

* `X` is not numeric (encode categoricals first), is not 2-D, or contains `±inf` (NaN is allowed: median imputation is applied inside the selector);
* `y` is not 1-D, has missing values, looks continuous (regression is not supported), or has fewer than 2 classes or fewer than 5 samples in a class;
* there are fewer than 20 rows or fewer than 2 columns; feature names are duplicated;
* `transform` receives different columns than `fit` (count or names).

Constant columns are dropped (reported with `NaN` in `report_`). If a policy returns an empty set, the 3 top-consensus features are returned with a warning (`fallback_ = True`).

## GPU

The rankers and the downstream models in the benchmark use `torch` / `xgboost` on CUDA when available (`decoyfs.device.env_report()` shows what was detected). On NVIDIA Blackwell cards (RTX 50xx) install a CUDA 12.8 build of torch: `pip install torch --index-url https://download.pytorch.org/whl/cu128`.

## Reproduce the paper

```bash
git clone <this repo> && cd decoyfs
pip install -e ".[bench]"
python benchmarks/build_notebook.py        # creates benchmarks/notebooks/DecoyFS_benchmark.ipynb
python benchmarks/run_benchmark.py --tier smoke          # minutes
python benchmarks/run_benchmark.py --tier full --force-gpu   # 74 datasets, 5x2 CV (about 23 h on an RTX 5060 Ti)
python benchmarks/analysis/decoy_vs_comparators.py        # paired comparison vs Boruta-style and CV-k
```

Datasets are downloaded from OpenML into `~/.cache/decoyfs` (override with `DECOYFS_CACHE`). The benchmark's policy identifiers (`CTRL_VOTE_MAJ`, `CTRL_STAB`, `BORUTA`, `CVK_1SE`, …) are kept as in the released result tables. The Spanish protocol design document is in [docs/DISENO_PROTOCOLO.es.md](docs/DISENO_PROTOCOLO.es.md).

## Cite

See [CITATION.cff](CITATION.cff). A software DOI is minted by Zenodo for every release.

## License

MIT © Nestor A. Zermeño Campos, Universidad Autónoma de Baja California.
