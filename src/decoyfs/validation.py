"""Input rules enforced before any computation, with actionable error messages.

Rules (classification only):

* ``X`` is 2-D and numeric (arrays, lists or pandas DataFrames). Non-numeric columns must be encoded first. ``NaN`` is
  allowed (median imputation inside the selector); ``+/-inf`` is not.
* ``y`` is 1-D, has no missing values, at least two classes and at least ``min_per_class`` samples in every class.
  Continuous targets are rejected (regression is not supported).
* ``X`` and ``y`` have the same number of rows, at least ``min_samples`` rows and at least two columns.
* Feature names, when present, are unique.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

MIN_SAMPLES = 20
MIN_PER_CLASS = 5


class DecoyInputError(ValueError):
    """Raised when the data violate the input contract of the library."""


def check_X_y(X, y, min_samples=MIN_SAMPLES, min_per_class=MIN_PER_CLASS):
    """Validate and normalise the inputs. Returns ``(X float64 ndarray, y int codes, classes, feature_names)``."""
    if X is None or y is None:
        raise DecoyInputError("X and y are required.")
    names = None
    if isinstance(X, pd.DataFrame):
        names = np.array([str(c) for c in X.columns], dtype=object)
        bad = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
        if bad:
            raise DecoyInputError(
                f"Non-numeric columns {bad[:5]}{'...' if len(bad) > 5 else ''}: encode them first "
                "(e.g. one-hot / ordinal encoding); DecoyFS ranks numeric columns only.")
        Xa = X.to_numpy(dtype="float64", na_value=np.nan)
    else:
        try:
            Xa = np.asarray(X, dtype="float64")
        except (TypeError, ValueError) as e:
            raise DecoyInputError(f"X must be numeric (could not convert to float): {e}") from e
    if Xa.ndim != 2:
        raise DecoyInputError(f"X must be 2-D (n_samples, n_features); got shape {Xa.shape}.")
    ya = np.asarray(y)
    if ya.ndim == 2 and ya.shape[1] == 1:
        ya = ya.ravel()
    if ya.ndim != 1:
        raise DecoyInputError(f"y must be 1-D; got shape {ya.shape}.")
    n, p = Xa.shape
    if len(ya) != n:
        raise DecoyInputError(f"X has {n} rows but y has {len(ya)} elements.")
    if np.isinf(Xa).any():
        raise DecoyInputError("X contains +/-inf: replace or clip the infinite values first.")
    if pd.isna(ya).any():
        raise DecoyInputError("y contains missing values: drop those rows first.")
    if ya.dtype.kind == "f" and not np.all(np.mod(ya, 1) == 0):
        raise DecoyInputError("y looks continuous (non-integer floats). DecoyFS supports classification only.")
    if n < min_samples:
        raise DecoyInputError(f"At least {min_samples} samples are required; got {n}.")
    if p < 2:
        raise DecoyInputError(f"At least 2 features are required; got {p}.")
    classes, codes = np.unique(ya, return_inverse=True)
    if len(classes) < 2:
        raise DecoyInputError("y has a single class; at least two are required.")
    counts = np.bincount(codes)
    small = {str(c): int(m) for c, m in zip(classes, counts) if m < min_per_class}
    if small:
        raise DecoyInputError(
            f"Every class needs at least {min_per_class} samples; too small: {small}. "
            "Merge or drop rare classes.")
    if len(classes) > max(50, n // 5):
        raise DecoyInputError(f"y has {len(classes)} distinct values for {n} rows: it does not look like a class label.")
    if names is not None and len(set(names)) != len(names):
        raise DecoyInputError("DataFrame column names must be unique.")
    if np.isnan(Xa).all(axis=0).any():
        warnings.warn("Some columns are entirely NaN; they are treated as constant and dropped.", stacklevel=3)
    if n < 100:
        warnings.warn(f"Only {n} samples: decoy thresholds and feature scores are very noisy.", stacklevel=3)
    return Xa, codes.astype(int), classes, names


def check_transform_X(X, n_features, names=None):
    """Validate X at transform time (same width as at fit time)."""
    if isinstance(X, pd.DataFrame):
        if names is not None and list(map(str, X.columns)) != list(names):
            raise DecoyInputError("Column names/order differ from those seen at fit time.")
        Xo = X
    else:
        Xo = np.asarray(X)
        if Xo.ndim != 2:
            raise DecoyInputError(f"X must be 2-D; got shape {Xo.shape}.")
    if Xo.shape[1] != n_features:
        raise DecoyInputError(f"X has {Xo.shape[1]} features; the selector was fitted with {n_features}.")
    return Xo
