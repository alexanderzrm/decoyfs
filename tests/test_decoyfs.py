import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from conftest import make_data

import decoyfs
from decoyfs import (BorutaSelector, CVKSelector, DecoyInputError, DecoySelector, FixedKSelector, compare_selectors,
                     list_policies, list_rankers, register_policy, register_ranker, unregister_policy,
                     unregister_ranker)
from decoyfs import policies as P
from decoyfs.controls import augment
from decoyfs.preprocessing import FoldPrep
from decoyfs.rankers import run_rankers

FAST = ("anova", "mi", "rf", "l1lr")


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("bad", ["text", "inf", "continuous_y", "nan_y", "one_class", "tiny", "one_col", "mismatch",
                                 "tiny_class", "dup_names", "str_col"])
def test_input_errors(bad):
    X, y = make_data(n=100)
    if bad == "text":
        X = np.array([["a"] * 5] * 100)
    elif bad == "inf":
        X[0, 0] = np.inf
    elif bad == "continuous_y":
        y = np.random.default_rng(0).normal(size=100)
    elif bad == "nan_y":
        y = y.astype(float)
        y[0] = np.nan
    elif bad == "one_class":
        y = np.zeros(100, int)
    elif bad == "tiny":
        X, y = X[:10], y[:10]
    elif bad == "one_col":
        X = X[:, :1]
    elif bad == "mismatch":
        y = y[:-1]
    elif bad == "tiny_class":
        y = np.zeros(100, int)
        y[:2] = 1
    elif bad == "dup_names":
        X = pd.DataFrame(X[:, :3], columns=["a", "a", "b"])
    elif bad == "str_col":
        X = pd.DataFrame(X[:, :3], columns=list("abc"))
        X["c"] = "x"
    with pytest.raises(DecoyInputError):
        DecoySelector(rankers=FAST).fit(X, y)


def test_invalid_parameters(data):
    X, y = data
    for kw in (dict(rankers=("nope",)), dict(policy="bogus"), dict(alpha=2), dict(controls=("block", "shadow")),
               dict(controls=()), dict(rankers=())):
        with pytest.raises(ValueError):
            DecoySelector(**kw).fit(X, y)


# ---------------------------------------------------------------- behaviour
def test_selects_signal_not_noise(data):
    X, y = data
    s = DecoySelector(rankers=FAST).fit(X, y)
    assert s.support_[:5].sum() >= 4
    assert s.support_[5:].sum() <= 3
    assert s.transform(X).shape[1] == s.support_.sum()


def test_policy_ordering(data):
    X, y = data
    n = {p: DecoySelector(rankers=FAST, policy=p).fit(X, y).support_.sum() for p in ("all", "majority", "any")}
    assert n["all"] <= n["majority"] <= n["any"]


def test_matches_benchmark_policy(data):
    """Same seed -> same set as the benchmark function ``ctrl_sets`` (CTRL_VOTE_MAJ)."""
    X, y = data
    sel = DecoySelector(rankers=FAST, random_state=3).fit(X, y)
    pre = FoldPrep().fit(X)
    Z = pre.transform(X)
    Xa, idx = augment(Z, np.random.default_rng(3))
    scores, _, _ = run_rankers(Xa, y, 2, 3, FAST)
    ref = P.ctrl_sets(scores, idx, 0.05)[0]["CTRL_VOTE_MAJ"]
    if len(ref):
        assert np.array_equal(np.where(sel.support_)[0], pre.keep_[ref])


def test_report_and_dataframe(data):
    X, y = make_data(as_df=True)
    s = DecoySelector(rankers=FAST).fit(X, y)
    assert isinstance(s.transform(X), pd.DataFrame)
    assert list(s.get_feature_names_out()) == s.selected_features_
    r = s.report_
    assert len(r) == X.shape[1] and r["selected"].sum() == s.support_.sum()
    assert {"votes", "consensus", "frac_beat_random", "frac_beat_shadow"} <= set(r.columns)
    assert set(s.decoy_thresholds_.index) == set(FAST)
    with pytest.raises(DecoyInputError):
        s.transform(X.iloc[:, ::-1])  # same names, different order -> must be rejected or aligned
    with pytest.raises(DecoyInputError):
        s.transform(X.iloc[:, :10])


def test_constant_and_nan_columns():
    X, y = make_data(nan_frac=0.05)
    X = np.hstack([X, np.ones((len(X), 2))])
    s = DecoySelector(rankers=FAST).fit(X, y)
    assert not s.support_[-2:].any()
    assert s.support_.shape == (X.shape[1],)
    assert np.isnan(s.report_["consensus"].iloc[-1])


def test_reproducible(data):
    X, y = data
    a = DecoySelector(rankers=FAST, random_state=1).fit(X, y).support_
    b = DecoySelector(rankers=FAST, random_state=1).fit(X, y).support_
    assert np.array_equal(a, b)


def test_fallback_on_pure_noise():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 20)), rng.integers(0, 2, 300)
    with pytest.warns(UserWarning):
        s = DecoySelector(rankers=FAST, policy="all", alpha=0.001).fit(X, y)
    if s.fallback_:
        assert s.support_.sum() == 3


def test_multiclass():
    X, y = make_data(n=450, n_classes=3)
    s = DecoySelector(rankers=FAST).fit(X, y)
    assert s.support_[:5].sum() >= 3


def test_stability_and_block(data):
    X, y = data
    s = DecoySelector(rankers=("anova", "rf"), stability_B=3).fit(X, y)
    assert "stability_freq" in s.report_
    b = DecoySelector(rankers=("anova", "rf"), controls=("random", "block")).fit(X, y)
    assert b.support_.any()
    r = DecoySelector(rankers=("anova", "rf"), controls=("random",)).fit(X, y)
    assert np.isnan(r.null_pass_rate_["shadow"])


# ---------------------------------------------------------------- extensibility
def test_ranker_registry(data):
    X, y = data
    register_ranker("absmean_diff", lambda Xm, ym: np.abs(Xm[ym == 1].mean(0) - Xm[ym == 0].mean(0)))
    try:
        assert "absmean_diff" in list_rankers()
        with pytest.raises(ValueError):
            register_ranker("absmean_diff", lambda a, b: a.sum(0))
        s = DecoySelector(rankers=("anova", "absmean_diff")).fit(X, y)
        assert s.support_[:5].sum() >= 4
    finally:
        unregister_ranker("absmean_diff")
    assert "absmean_diff" not in list_rankers()
    with pytest.raises(ValueError):
        unregister_ranker("anova")


def test_failing_ranker_is_skipped(data):
    X, y = data

    def boom(Xm, ym):
        raise RuntimeError("boom")
    register_ranker("boom", boom)
    try:
        with pytest.warns(UserWarning):
            s = DecoySelector(rankers=("anova", "rf", "boom")).fit(X, y)
        assert "boom" in s.ranker_errors_ and s.rankers_used_ == ["anova", "rf"]
    finally:
        unregister_ranker("boom")


def test_policy_registry_and_callable(data):
    X, y = data
    register_policy("top3", lambda votes, M, c: np.argsort(-c)[:3])
    try:
        assert "top3" in list_policies()
        assert DecoySelector(rankers=FAST, policy="top3").fit(X, y).support_.sum() == 3
    finally:
        unregister_policy("top3")
    assert DecoySelector(rankers=FAST, policy=lambda v, M, c: np.where(v >= 1)[0]).fit(X, y).support_.any()
    assert DecoySelector(rankers=FAST, policy=2).fit(X, y).support_.any()
    with pytest.raises(ValueError):
        DecoySelector(rankers=FAST, policy=9).fit(X, y)


# ---------------------------------------------------------------- scikit-learn integration
def test_sklearn_clone_pipeline(data):
    X, y = data
    s = DecoySelector(rankers=FAST, alpha=0.1)
    assert clone(s).get_params() == s.get_params()
    pipe = Pipeline([("sel", DecoySelector(rankers=FAST)), ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.7


# ---------------------------------------------------------------- comparators
def test_comparators(data):
    X, y = data
    f = FixedKSelector(k=0.2, rankers=FAST).fit(X, y)
    assert f.support_.sum() == int(np.ceil(0.2 * X.shape[1]))
    assert FixedKSelector(k=4, rankers=FAST).fit(X, y).support_.sum() == 4
    assert CVKSelector(rankers=FAST).fit(X, y).support_.any()
    assert BorutaSelector(n_iter=25, budget_s=60).fit(X, y).support_[:5].sum() >= 3


def test_compare_selectors(data):
    X, y = data
    df = compare_selectors(X, y, {"all": None, "decoy": DecoySelector(rankers=FAST), "fixed": FixedKSelector(k=5)},
                           model="logreg", cv=3)
    assert set(df["selector"]) == {"all", "decoy", "fixed"} and len(df) == 9
    assert df.groupby("selector")["auroc"].mean().min() > 0.8


def test_fdp_estimate(data):
    X, y = data
    s = DecoySelector(rankers=FAST, random_state=0).fit(X, y)
    est = s.fdp_estimate_
    assert set(est) == {"shadow", "vote"}
    assert 0.0 <= est["vote"] <= 1.0 and 0.0 <= est["shadow"] <= 1.0
    # recompute by hand from the audit quantities
    p = int(s.report_["used"].sum())
    k = int(s.support_.sum())
    assert est["shadow"] == pytest.approx(min(1.0, s.null_pass_rate_["shadow"] * (p - k) / k))
    # stricter policy -> no larger estimated FDP on average signal data is not guaranteed, but it must stay defined
    for pol in ("all", "any", 2):
        e = DecoySelector(rankers=FAST, policy=pol, random_state=0).fit(X, y).fdp_estimate_
        assert np.isfinite(e["vote"])


def test_fdp_estimate_not_available():
    X, y = make_data()
    assert np.isnan(DecoySelector(rankers=("anova", "rf"), policy=lambda v, M, c: np.where(v >= 1)[0]).fit(X, y)
                    .fdp_estimate_["vote"])
    assert np.isnan(DecoySelector(rankers=("anova", "rf"), stability_B=3).fit(X, y).fdp_estimate_["vote"])
    assert np.isnan(DecoySelector(rankers=("anova", "rf"), controls=("random", "block")).fit(X, y)
                    .fdp_estimate_["shadow"])


def test_estimate_fdp_function():
    assert P.estimate_fdp(0.1, 10, 110) == pytest.approx(1.0)
    assert P.estimate_fdp(0.05, 10, 110) == pytest.approx(0.5)
    assert P.estimate_fdp(0.5, 5, 100) == 1.0           # clipped
    assert np.isnan(P.estimate_fdp(0.1, 0, 100))
    assert np.isnan(P.estimate_fdp(float("nan"), 5, 100))


def test_version():
    assert decoyfs.__version__ == "0.1.1"
