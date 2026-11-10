"""Per-fold preprocessing, fitted on the training data only."""
from __future__ import annotations

import numpy as np
from scipy.stats import skew


class FoldPrep:
    """Median imputation, constant-column removal, Yeo-Johnson on strongly skewed columns, robust scaling, clip."""

    def __init__(self, skew_thr=1.0, clip=10.0):
        self.skew_thr, self.clip = skew_thr, clip

    def fit(self, X):
        X = np.asarray(X, dtype="float64")
        self.med_ = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.keep_ = np.where(Xi.std(0) > 0)[0]
        Xi = Xi[:, self.keep_]
        sk = np.abs(np.nan_to_num(skew(Xi, axis=0)))
        nun = np.array([len(np.unique(Xi[:, j])) for j in range(Xi.shape[1])])
        self.yj_cols_ = np.where((sk > self.skew_thr) & (nun > 10))[0]
        self.pt_ = None
        if len(self.yj_cols_):
            from sklearn.preprocessing import PowerTransformer
            self.pt_ = PowerTransformer(method="yeo-johnson", standardize=False).fit(Xi[:, self.yj_cols_])
            Xi[:, self.yj_cols_] = self.pt_.transform(Xi[:, self.yj_cols_])
        q1, q3 = np.percentile(Xi, [25, 75], axis=0)
        self.center_ = np.median(Xi, axis=0)
        iqr = q3 - q1
        std = Xi.std(0)
        self.scale_ = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype="float64")
        Xi = np.where(np.isnan(X), self.med_, X)[:, self.keep_]
        if self.pt_ is not None:
            Xi[:, self.yj_cols_] = self.pt_.transform(Xi[:, self.yj_cols_])
        Xi = (Xi - self.center_) / self.scale_
        return np.clip(Xi, -self.clip, self.clip).astype("float32")
