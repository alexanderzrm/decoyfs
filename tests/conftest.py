import numpy as np
import pandas as pd
import pytest


def make_data(n=400, p_signal=5, p_noise=25, seed=0, n_classes=2, nan_frac=0.0, as_df=False):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    sig = rng.normal(size=(n, p_signal)) + 1.6 * y[:, None]
    noise = rng.normal(size=(n, p_noise))
    X = np.hstack([sig, noise])
    if nan_frac:
        X[rng.random(X.shape) < nan_frac] = np.nan
    if as_df:
        X = pd.DataFrame(X, columns=[f"s{i}" for i in range(p_signal)] + [f"n{i}" for i in range(p_noise)])
    return X, y


@pytest.fixture
def data():
    return make_data()


@pytest.fixture
def small_rankers():
    return ("anova", "mi", "rf", "l1lr")
