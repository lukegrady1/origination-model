"""Shared Ridge score regression with training-only preprocessing (spec section 9, M1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nfl_origination.config import FeatureSet, ModelFamily
from nfl_origination.errors import ModelValidationError
from nfl_origination.schemas import (
    REQUIRED_METRIC_FEATURES,
    assert_feature_allowlist,
    assert_no_market_columns,
    feature_columns,
)

ROW_WEIGHT = 0.5  # each game contributes total weight 1 across its two perspective rows


@dataclass(frozen=True)
class FitSpec:
    family: ModelFamily
    feature_set: FeatureSet
    alpha: float | None
    seed: int = 42
    max_unexpected_null_rate: float = 0.05

    @property
    def model_id(self) -> str:
        if self.family == "league_baseline":
            return "B0_league_baseline"
        suffix = "" if self.feature_set == "full" else f"_{self.feature_set}"
        alpha = "na" if self.alpha is None else f"{self.alpha:g}"
        return f"M1_ridge{suffix}_alpha{alpha}"


class Preprocessor:
    """Training-only median imputation, missingness indicators, and standard scaling."""

    def __init__(self, features: list[str]) -> None:
        self.features = list(features)
        self.medians: dict[str, float] = {}
        self.indicator_features: list[str] = []
        self.means: np.ndarray = np.zeros(0)
        self.scales: np.ndarray = np.ones(0)
        self.null_rates: dict[str, float] = {}

    @property
    def output_columns(self) -> list[str]:
        return self.features + [f"{c}__missing" for c in self.indicator_features]

    def fit(self, X: pd.DataFrame, max_null_rate: float) -> Preprocessor:
        assert_no_market_columns(list(X.columns), "Preprocessor.fit")
        frame = X[self.features].astype(float)
        self.null_rates = {c: float(frame[c].isna().mean()) for c in self.features}
        all_null = [c for c, r in self.null_rates.items() if r >= 1.0]
        if all_null:
            raise ModelValidationError(f"required feature columns are entirely null: {all_null}")
        too_many = {
            c: r
            for c, r in self.null_rates.items()
            if c in REQUIRED_METRIC_FEATURES and r > max_null_rate
        }
        if too_many:
            raise ModelValidationError(
                f"unexpected null rate above {max_null_rate:.0%} in required metrics: {too_many}; "
                "investigate the source before training"
            )
        self.medians = {c: float(np.nanmedian(frame[c].to_numpy())) for c in self.features}
        self.indicator_features = [c for c, r in self.null_rates.items() if r > 0.0]
        matrix = self._assemble(frame)
        scaler = StandardScaler().fit(matrix)
        self.means = np.asarray(scaler.mean_, dtype=float)
        self.scales = np.asarray(scaler.scale_, dtype=float)
        return self

    def _assemble(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.features:
            cols.append(frame[c].fillna(self.medians[c]).to_numpy(dtype=float))
        for c in self.indicator_features:
            cols.append(frame[c].isna().to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(frame), 0))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ModelValidationError(f"prediction features missing columns {missing}")
        frame = X[self.features].astype(float)
        matrix = self._assemble(frame)
        return (matrix - self.means) / self.scales

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "medians": self.medians,
            "indicator_features": self.indicator_features,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "null_rates": self.null_rates,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessor:
        p = cls(list(d["features"]))
        p.medians = {k: float(v) for k, v in d["medians"].items()}
        p.indicator_features = list(d["indicator_features"])
        p.means = np.asarray(d["means"], dtype=float)
        p.scales = np.asarray(d["scales"], dtype=float)
        p.null_rates = {k: float(v) for k, v in d.get("null_rates", {}).items()}
        return p


class RidgeScoreModel:
    """One shared Ridge pipeline predicting points for each perspective row."""

    family: ModelFamily = "ridge_score"

    def __init__(self, spec: FitSpec) -> None:
        if spec.alpha is None:
            raise ModelValidationError("ridge_score requires a resolved alpha")
        self.spec = spec
        self.features = feature_columns(spec.feature_set)
        self.preprocessor = Preprocessor(self.features)
        self.coef: np.ndarray = np.zeros(0)
        self.intercept: float = 0.0
        self.n_rows: int = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RidgeScoreModel:
        assert_feature_allowlist(
            [c for c in X.columns if c in self.features], self.spec.feature_set
        )
        self.preprocessor.fit(X, self.spec.max_unexpected_null_rate)
        matrix = self.preprocessor.transform(X)
        weights = np.full(len(matrix), ROW_WEIGHT)
        model = Ridge(alpha=float(self.spec.alpha or 0.0), fit_intercept=True)
        model.fit(matrix, np.asarray(y, dtype=float), sample_weight=weights)
        self.coef = np.asarray(model.coef_, dtype=float)
        self.intercept = float(model.intercept_)
        self.n_rows = len(matrix)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self.preprocessor.transform(X)
        return matrix @ self.coef + self.intercept

    def standardized_coefficients(self) -> dict[str, float]:
        return dict(zip(self.preprocessor.output_columns, self.coef.tolist(), strict=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "feature_set": self.spec.feature_set,
            "alpha": self.spec.alpha,
            "preprocessor": self.preprocessor.to_dict(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], seed: int = 42) -> RidgeScoreModel:
        spec = FitSpec("ridge_score", d["feature_set"], float(d["alpha"]), seed)
        m = cls(spec)
        m.preprocessor = Preprocessor.from_dict(d["preprocessor"])
        m.coef = np.asarray(d["coef"], dtype=float)
        m.intercept = float(d["intercept"])
        m.n_rows = int(d.get("n_rows", 0))
        return m
