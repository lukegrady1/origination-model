"""Configuration contract (spec section 20) with strict validation.

Unknown keys fail. YAML files are the source of defaults; CLI flags override via
dotted keys. The fully resolved config and its hash are saved with every run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from nfl_origination.errors import InvalidInputError

DataMode = Literal["historical_reconstruction", "recorded_asof"]
RunKind = Literal["development", "confirmation", "holdout", "forecast", "fit", "demo"]
FeatureSet = Literal["full", "epa_free"]
ModelFamily = Literal["ridge_score", "league_baseline"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictModel):
    kind: RunKind
    label: str = "v1"
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


class DataConfig(StrictModel):
    seasons: tuple[int, int] = (2010, 2025)
    game_type: Literal["REG"] = "REG"
    mode: DataMode = "historical_reconstruction"
    completed_game_lag_hours: float = 48.0
    source: Literal["nflverse", "synthetic"] = "nflverse"
    synthetic_fixture: Path | None = None
    cache_dir: Path = Path("data/raw")
    normalized_dir: Path = Path("data/normalized")
    features_dir: Path = Path("data/features")

    @field_validator("seasons")
    @classmethod
    def _check_seasons(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] > v[1]:
            raise ValueError("data.seasons must be [first, last] with first <= last")
        return v

    @property
    def season_list(self) -> list[int]:
        return list(range(self.seasons[0], self.seasons[1] + 1))


class ForecastConfig(StrictModel):
    cutoff_hours_before_kickoff: float = 24.0


class FeaturesConfig(StrictModel):
    history_games: int = 16
    half_life_games: float = 8.0
    offseason_weight: float = 0.5
    prior_seasons: int = 2
    shrinkage_plays: float = 200.0
    shrinkage_split_plays: float = 100.0
    shrinkage_games: float = 4.0
    rest_default_days: float = 7.0
    rest_min_days: float = 3.0
    rest_max_days: float = 14.0
    max_unexpected_null_rate: float = 0.05


class ModelConfig(StrictModel):
    family: ModelFamily = "ridge_score"
    feature_set: FeatureSet = "full"
    alpha_candidates: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    selected_alpha: float | None = None
    ablation_selected_alpha: float | None = None
    train_start_season: int = 2012
    residual_first_season: int = 2016
    residual_window_seasons: int = 5
    min_residual_games: int = 500
    covariance_diagonal_shrinkage: float = 0.1
    covariance_floor: float = 1e-6
    alpha_tie_tolerance: float = 0.01


class DistributionConfig(StrictModel):
    initial_max_score: int = 100
    score_step: int = 25
    hard_max_score: int = 250
    max_omitted_mass: float = 1.0e-8
    quadrature_nodes_per_cell: int = 12
    negative_latent_warn: float = 0.01
    negative_cell_tolerance: float = 1e-10
    pmf_sum_tolerance: float = 1e-8


class EvaluationConfig(StrictModel):
    development_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023)
    confirmation_season: int = 2024
    holdout_season: int = 2025
    bootstrap_replicates: int = 1000
    sparse_bin_threshold: int = 30
    ablation: bool = True


class MarketConfig(StrictModel):
    enabled: bool = False
    bookmaker: str | None = None
    odds_path: Path | None = None
    max_quote_age_minutes: float = 60.0
    min_ev: float = 0.03
    stake_units: float = 1.0
    max_selections_per_game: int = 1
    moneyline_settlement_rule: Literal["two_way_tie_void"] = "two_way_tie_void"
    closing_proxy_window_minutes: float = 30.0
    synthetic: bool = False


class ExperimentConfig(StrictModel):
    schema_version: int = 1
    seed: int = 42
    run: RunConfig
    data: DataConfig = DataConfig()
    forecast: ForecastConfig = ForecastConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    distribution: DistributionConfig = DistributionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    market: MarketConfig = MarketConfig()

    @model_validator(mode="after")
    def _check_consistency(self) -> ExperimentConfig:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        ev = self.evaluation
        if self.run.kind in ("holdout", "confirmation", "forecast") and (
            self.model.selected_alpha is None and self.model.family == "ridge_score"
        ):
            raise ValueError(
                f"run.kind={self.run.kind} requires model.selected_alpha resolved from the "
                "development decision record"
            )
        if ev.confirmation_season <= max(ev.development_seasons):
            raise ValueError("confirmation_season must follow development seasons")
        if ev.holdout_season <= ev.confirmation_season:
            raise ValueError("holdout_season must follow confirmation_season")
        if self.model.residual_first_season < self.model.train_start_season:
            raise ValueError("residual_first_season cannot precede train_start_season")
        return self

    def score_seasons(self) -> list[int]:
        """Seasons scored by this run kind."""
        ev = self.evaluation
        if self.run.kind == "development":
            return list(ev.development_seasons)
        if self.run.kind == "confirmation":
            return [ev.confirmation_season]
        if self.run.kind == "holdout":
            return [ev.holdout_season]
        if self.run.kind == "demo":
            return [*ev.development_seasons, ev.confirmation_season, ev.holdout_season]
        return []

    def resolved_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def config_hash(self) -> str:
        payload = json.dumps(self.resolved_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = target
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise InvalidInputError(f"override {key!r} conflicts with a scalar config value")
    node[parts[-1]] = value


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config, apply dotted overrides, and validate strictly."""
    if not path.exists():
        raise InvalidInputError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise InvalidInputError(f"config file {path} must contain a mapping")
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, _parse_override_value(value) if isinstance(value, str) else value)
    try:
        return ExperimentConfig.model_validate(raw)
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise InvalidInputError(f"invalid config {path}: {exc}") from exc


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise InvalidInputError(f"override must look like key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_season_range(text: str) -> list[int]:
    """Parse '2010:2025' (inclusive) or '2024' into a season list."""
    text = text.strip()
    try:
        if ":" in text:
            first, last = (int(part) for part in text.split(":", 1))
        else:
            first = last = int(text)
    except ValueError as exc:
        raise InvalidInputError(f"invalid season range {text!r}; use YYYY or YYYY:YYYY") from exc
    if first > last:
        raise InvalidInputError(f"invalid season range {text!r}: first season after last")
    return list(range(first, last + 1))
