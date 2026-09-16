"""Model bundles: fitted score model + residual distribution + provenance (spec section 14).

Bundles are plain JSON generated locally. Loading never unpickles arbitrary objects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.config import DistributionConfig
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.models.baseline import LeagueBaselineModel
from nfl_origination.models.distribution import (
    ResidualParams,
    ScoreDistribution,
    predict_distribution,
)
from nfl_origination.models.ridge import FitSpec, RidgeScoreModel
from nfl_origination.provenance import (
    dependency_versions,
    iso_utc,
    read_json,
    utc_now,
    write_json,
)
from nfl_origination.schemas import FEATURE_VERSION, SCHEMA_VERSION, assert_no_market_columns

ScoreModel = RidgeScoreModel | LeagueBaselineModel


class ModelBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    family: str
    feature_set: str
    features: list[str]
    alpha: float | None
    model_params: dict[str, Any]
    residual: dict[str, Any] | None
    residual_game_ids: list[str] = Field(default_factory=list)
    training_seasons: list[int]
    training_games: int
    training_rows: int
    training_game_ids_hash: str
    training_cutoff_utc: str | None
    forecast_season: int | None
    data_mode: str
    schema_version: int = SCHEMA_VERSION
    feature_version: str = FEATURE_VERSION
    config_hash: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dependency_versions)
    created_at_utc: str = Field(default_factory=lambda: iso_utc(utc_now()) or "")
    notes: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> Path:
        write_json(path, self.model_dump())
        return path

    @classmethod
    def load(cls, path: Path) -> ModelBundle:
        if not Path(path).exists():
            raise MissingDataError(f"model bundle not found: {path}")
        bundle = cls.model_validate(read_json(Path(path)))
        if bundle.schema_version != SCHEMA_VERSION or bundle.feature_version != FEATURE_VERSION:
            raise ModelValidationError(
                f"bundle {bundle.model_id} has schema/feature version "
                f"{bundle.schema_version}/{bundle.feature_version}; expected "
                f"{SCHEMA_VERSION}/{FEATURE_VERSION}"
            )
        return bundle

    def score_model(self, seed: int = 42) -> ScoreModel:
        if self.family == "ridge_score":
            return RidgeScoreModel.from_dict(self.model_params, seed)
        if self.family == "league_baseline":
            return LeagueBaselineModel.from_dict(self.model_params, seed)
        raise ModelValidationError(f"unknown model family {self.family!r}")

    def residual_params(self) -> ResidualParams:
        if self.residual is None:
            raise ModelValidationError(f"bundle {self.model_id} has no residual distribution")
        return ResidualParams.from_dict(self.residual)


def make_model(spec: FitSpec) -> ScoreModel:
    if spec.family == "ridge_score":
        return RidgeScoreModel(spec)
    if spec.family == "league_baseline":
        return LeagueBaselineModel(spec)
    raise ModelValidationError(f"unknown model family {spec.family!r}")


def align_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels to feature rows by (game_id, perspective); rows without labels are dropped."""
    if "points" in features.columns and "label_available_utc" in features.columns:
        return features  # already aligned
    merged = features.merge(
        labels[["game_id", "perspective", "points", "label_available_utc"]],
        on=["game_id", "perspective"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def fit_score_model(
    features: pd.DataFrame, labels: pd.DataFrame, spec: FitSpec
) -> tuple[ScoreModel, pd.DataFrame]:
    """Fit on labeled rows only. Returns the model and the training frame used."""
    assert_no_market_columns(list(features.columns), "fit_score_model")
    train = align_labels(features, labels)
    train = train[~train["insufficient_warmup"]]
    if train.empty:
        raise MissingDataError("no labeled feature rows available for training")
    model = make_model(spec)
    model.fit(train, train["points"].to_numpy(dtype=float))
    return model, train


def predict_location(model: ScoreModel, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(features), dtype=float)


def game_locations(rows: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Combine perspective-row predictions into one (mu_home, mu_away) row per game."""
    frame = rows[["game_id", "perspective"]].copy()
    frame["mu"] = predicted
    wide = frame.pivot(index="game_id", columns="perspective", values="mu")
    if not {"home", "away"} <= set(wide.columns) or wide.isna().any().any():
        raise ModelValidationError("each game needs both home and away perspective predictions")
    out = wide.rename(columns={"home": "mu_home_score", "away": "mu_away_score"}).reset_index()
    return out[["game_id", "mu_home_score", "mu_away_score"]]


def game_ids_hash(game_ids: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(set(game_ids))).encode()).hexdigest()


def build_bundle(
    model: ScoreModel,
    spec: FitSpec,
    train: pd.DataFrame,
    residual: ResidualParams | None,
    residual_game_ids: list[str],
    *,
    forecast_season: int | None,
    data_mode: str,
    config_hash: str | None,
    notes: list[str] | None = None,
) -> ModelBundle:
    cutoff = train["label_available_utc"].max() if "label_available_utc" in train else None
    return ModelBundle(
        model_id=spec.model_id,
        family=spec.family,
        feature_set=spec.feature_set,
        features=list(getattr(model, "features", [])),
        alpha=spec.alpha,
        model_params=model.to_dict(),
        residual=None if residual is None else residual.to_dict(),
        residual_game_ids=sorted(residual_game_ids),
        training_seasons=sorted(int(s) for s in train["season"].unique()),
        training_games=int(train["game_id"].nunique()),
        training_rows=len(train),
        training_game_ids_hash=game_ids_hash(train["game_id"].tolist()),
        training_cutoff_utc=iso_utc(pd.Timestamp(cutoff).to_pydatetime())
        if cutoff is not None
        else None,
        forecast_season=forecast_season,
        data_mode=data_mode,
        config_hash=config_hash,
        notes=notes or [],
    )


def predict_game(
    bundle: ModelBundle, game_features: pd.DataFrame, cfg: DistributionConfig
) -> ScoreDistribution:
    """Score distribution for one game from its two perspective feature rows."""
    if bundle.feature_version != FEATURE_VERSION:
        raise ModelValidationError("feature version mismatch between bundle and features")
    if (
        "feature_version" in game_features.columns
        and (game_features["feature_version"] != bundle.feature_version).any()
    ):
        raise ModelValidationError("feature rows were built with a different feature version")
    if len(game_features) != 2 or set(game_features["perspective"]) != {"home", "away"}:
        raise ModelValidationError("predict_game needs exactly one home and one away row")
    if game_features["insufficient_warmup"].any():
        raise MissingDataError("game lacks league prior warm-up; cannot forecast")
    model = bundle.score_model()
    mu = predict_location(model, game_features)
    loc = game_locations(game_features, mu).iloc[0]
    location = np.array([loc["mu_home_score"], loc["mu_away_score"]])
    return predict_distribution(location, bundle.residual_params(), cfg)
