"""Baseline B0: pooled training mean score plus estimated home/away offset (spec section 9)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ModelFamily
from nfl_origination.errors import ModelValidationError
from nfl_origination.models.ridge import FitSpec


class LeagueBaselineModel:
    family: ModelFamily = "league_baseline"
    features = ["venue_advantage"]

    def __init__(self, spec: FitSpec) -> None:
        self.spec = spec
        self.mean_score = float("nan")
        self.home_offset = 0.0
        self.away_offset = 0.0
        self.n_rows = 0

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LeagueBaselineModel:
        venue = X["venue_advantage"].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ModelValidationError("baseline requires training rows")
        self.mean_score = float(y.mean())
        home, away = venue > 0, venue < 0
        self.home_offset = float(y[home].mean() - self.mean_score) if home.any() else 0.0
        self.away_offset = float(y[away].mean() - self.mean_score) if away.any() else 0.0
        self.n_rows = len(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        venue = X["venue_advantage"].to_numpy(dtype=float)
        offset = np.where(venue > 0, self.home_offset, np.where(venue < 0, self.away_offset, 0.0))
        return self.mean_score + offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "feature_set": self.spec.feature_set,
            "alpha": None,
            "mean_score": self.mean_score,
            "home_offset": self.home_offset,
            "away_offset": self.away_offset,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], seed: int = 42) -> LeagueBaselineModel:
        m = cls(FitSpec("league_baseline", d.get("feature_set", "full"), None, seed))
        m.mean_score = float(d["mean_score"])
        m.home_offset = float(d["home_offset"])
        m.away_offset = float(d["away_offset"])
        m.n_rows = int(d.get("n_rows", 0))
        return m
