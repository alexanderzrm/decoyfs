"""Common machinery of the selectors: validation, per-fold preprocessing and a scikit-learn compatible interface."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .preprocessing import FoldPrep
from .validation import DecoyInputError, check_transform_X, check_X_y


class BaseSelector(TransformerMixin, BaseEstimator):
    """Base class: ``fit`` validates and preprocesses, subclasses implement :meth:`_select` on the preprocessed matrix.

    After ``fit``: ``support_`` (boolean mask over the ORIGINAL columns), ``selected_features_`` (names or indices),
    ``n_features_in_``, ``classes_``, ``preprocessor_`` and ``fit_time_`` (seconds).
    """

    def _select(self, Z, y, k):
        """Return the indices (into the columns of ``Z``) of the selected features."""
        raise NotImplementedError

    def fit(self, X, y):
        import time
        t0 = time.perf_counter()
        Xa, yc, classes, names = check_X_y(X, y)
        self.n_features_in_ = Xa.shape[1]
        if names is not None:
            self.feature_names_in_ = names
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        self.classes_ = classes
        self.preprocessor_ = FoldPrep().fit(Xa)
        keep = self.preprocessor_.keep_
        if len(keep) < 2:
            raise DecoyInputError("Fewer than two non-constant features remain after preprocessing.")
        Z = self.preprocessor_.transform(Xa)
        sel = np.asarray(self._select(Z, yc, len(classes)), dtype=int)
        mask = np.zeros(Xa.shape[1], bool)
        mask[keep[sel]] = True
        self.support_ = mask
        self.fit_time_ = time.perf_counter() - t0
        return self

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return np.where(self.support_)[0] if indices else self.support_.copy()

    @property
    def selected_features_(self):
        check_is_fitted(self, "support_")
        if hasattr(self, "feature_names_in_"):
            return list(self.feature_names_in_[self.support_])
        return list(np.where(self.support_)[0])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            names = self.feature_names_in_
        else:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(names[self.support_], dtype=object)

    def transform(self, X):
        check_is_fitted(self, "support_")
        Xo = check_transform_X(X, self.n_features_in_, getattr(self, "feature_names_in_", None))
        if isinstance(Xo, pd.DataFrame):
            return Xo.loc[:, self.support_]
        return Xo[:, self.support_]

    def inverse_transform(self, X):
        check_is_fitted(self, "support_")
        X = np.asarray(X)
        out = np.zeros((X.shape[0], self.n_features_in_), dtype=X.dtype)
        out[:, self.support_] = X
        return out

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)
