"""Minimal end-to-end example: select features, inspect the audit trail and compare with other selectors."""
from sklearn.datasets import make_classification

from decoyfs import BorutaSelector, DecoySelector, FixedKSelector, compare_selectors

X, y = make_classification(n_samples=600, n_features=40, n_informative=6, n_redundant=0, random_state=0)

sel = DecoySelector(policy="majority", random_state=0).fit(X, y)
print("selected columns:", sel.selected_features_)
print(sel.report_.sort_values("votes", ascending=False).head(10)[["feature", "votes", "frac_beat_random",
                                                                    "frac_beat_shadow", "selected"]])
print("null-pass rate:", sel.null_pass_rate_)

res = compare_selectors(
    X, y,
    {"all": None, "decoyfs": DecoySelector(), "fixed25": FixedKSelector(k=0.25), "boruta_style": BorutaSelector()},
    model="logreg", cv=5)
print(res.groupby("selector")[["auroc", "n_selected", "select_time_s"]].mean().round(3))
