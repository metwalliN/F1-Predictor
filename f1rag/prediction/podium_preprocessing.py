"""The Milestone 1 preprocessing class, copied verbatim from the notebook (section P1).

The saved model (models/podium_model.joblib) is a sklearn Pipeline whose first step is an instance of this class,
pickled under the module name `podium_preprocessing`. This file must stay identical to the notebook's class:
predictor.py checks that the loaded model reproduces the notebook's test-set probabilities exactly.
Nothing here is ever fitted by the app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

LOG_FEATURE = 'q1_gap_pct'               # >= 0 and right-skewed (a few drivers 5-10%+ off the pace): log1p helps
CLIP_FEATURE = 'q1_gap_to_teammate_pct'  # long tails on both sides (a teammate with a wrecked lap): clip at 1st/99th pct
CLIP_QUANTILES = (0.01, 0.99)
# Missing flags only where the gap itself carries information and no other feature already says so
# (missing form rates are already told apart by driver_form_races / constructor_form_races = 0).
MISSING_FLAGS = ['q1_gap_pct', 'q1_gap_to_teammate_pct', 'grid_minus_quali', 'driver_champ_rank', 'constructor_champ_rank']

class PodiumPreprocessor(BaseEstimator, TransformerMixin):
    """fit() learns fill values, clip bounds and scaling from the rows it is given (training rows only).
    transform() only applies those stored numbers, row by row; it never looks at statistics of new data.
    log_order : 'impute_first'  -> fill LOG_FEATURE with its raw mean, then log1p
                'transform_first' -> log1p, then fill with the mean of the logged values
    clip_order: 'impute_first'  -> fill CLIP_FEATURE with its raw mean, then clip at quantiles of the filled column
                'clip_first'    -> clip observed values at their quantiles, then fill with the mean of clipped values
    All other features: median fill. scale=True standardises every output column (mean 0, sd 1 on train)."""
    def __init__(self, features, log_order='transform_first', clip_order='clip_first', scale=True):
        self.features, self.log_order, self.clip_order, self.scale = features, log_order, clip_order, scale

    def _flags(self, X):
        return pd.DataFrame({f'{f}_missing': X[f].isna().astype(float) for f in self.flags_}, index=X.index)

    def fit(self, X, y=None):
        assert self.log_order in ('impute_first', 'transform_first') and self.clip_order in ('impute_first', 'clip_first')
        X = X[self.features].astype(float)
        assert X[LOG_FEATURE].dropna().ge(0).all(), 'log1p needs non-negative values'
        self.flags_ = [f for f in MISSING_FLAGS if f in self.features]
        raw = X[LOG_FEATURE]
        self.log_fill_ = (float(np.log1p(raw.mean())) if self.log_order == 'impute_first'   # log of the raw mean
                          else float(np.log1p(raw).mean()))                              # mean of the logs
        c = X[CLIP_FEATURE]
        if self.clip_order == 'impute_first':
            filled = c.fillna(c.mean())
            self.clip_fill_ = float(c.mean())
            self.clip_bounds_ = tuple(float(v) for v in filled.quantile(CLIP_QUANTILES))
        else:
            self.clip_bounds_ = tuple(float(v) for v in c.quantile(CLIP_QUANTILES))   # observed values only
            self.clip_fill_ = float(c.clip(*self.clip_bounds_).mean())
        others = [f for f in self.features if f not in (LOG_FEATURE, CLIP_FEATURE)]
        self.medians_ = X[others].median().to_dict()
        assert all(np.isfinite(v) for v in self.medians_.values()), 'a feature has no training values'
        Z = self._apply(X)
        self.mean_, self.sd_ = Z.mean(), Z.std(ddof=0).replace(0, 1.0)
        self.feature_names_out_ = list(Z.columns)
        return self

    def _apply(self, X):
        X = X[self.features].astype(float)
        out = X.copy()
        out[LOG_FEATURE] = np.log1p(X[LOG_FEATURE]).fillna(self.log_fill_)     # same result for both orders, given the fill
        if self.clip_order == 'impute_first':
            out[CLIP_FEATURE] = X[CLIP_FEATURE].fillna(self.clip_fill_).clip(*self.clip_bounds_)
        else:
            out[CLIP_FEATURE] = X[CLIP_FEATURE].clip(*self.clip_bounds_).fillna(self.clip_fill_)
        for f, m in self.medians_.items(): out[f] = X[f].fillna(m)
        return pd.concat([out, self._flags(X)], axis=1)

    def transform(self, X):
        check_is_fitted(self, 'mean_')
        Z = self._apply(X)
        return ((Z - self.mean_) / self.sd_) if self.scale else Z

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


