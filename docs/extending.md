# Extending DecoyFS

## Add a ranker

A ranker is a function `fn(X, y) -> scores` returning one finite score per column of `X` (higher = more relevant). `X` is the preprocessed matrix **including decoys**; `y` are integer class codes.

```python
import numpy as np
from decoyfs import register_ranker, DecoySelector

def mean_gap(X, y):
    return np.abs(X[y == 1].mean(0) - X[y == 0].mean(0))

register_ranker("mean_gap", mean_gap)
DecoySelector(rankers=("anova", "rf", "mean_gap")).fit(X, y)
```

Rules: register under a new name (`overwrite=True` to replace); built-in rankers cannot be unregistered; a ranker that raises is skipped with a warning and listed in `ranker_errors_`. Use `register_ranker(name, fn, takes_ctx=True)` if you need the context object (`ctx.X`, `ctx.y`, `ctx.k`, `ctx.seed`, `ctx.dev`).

## Add a policy

A policy receives `votes` (number of rankers beating the decoys, per real feature), `n_rankers` and the Borda `consensus` of the real features, and returns indices or a boolean mask.

```python
from decoyfs import register_policy
register_policy("strict_or_top5", lambda votes, M, c: (votes == M).nonzero()[0] if (votes == M).any() else c.argsort()[::-1][:5])
```

Policies can also be passed directly as an integer (`policy=3`) or a callable.

## Add a comparator

Subclass `decoyfs._base.BaseSelector` and implement `_select(Z, y, n_classes)` returning indices into the preprocessed matrix `Z`. You get validation, preprocessing, the scikit-learn interface and compatibility with `compare_selectors` for free.
