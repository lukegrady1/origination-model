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
RunKind = Literal[
    "development",
    "confirmation",
    "holdout",
    "forecast",
    "fit",
    "demo",
    "v2_research",
    "v2_prospective",
    "v2_demo",
]
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


# ---------------------------------------------------------------------------------------------
# V2 (prospective collection, market benchmarks, distribution challenger)
# ---------------------------------------------------------------------------------------------

EvidenceMode = Literal["research_reconstruction", "observed_prospective"]


class V2StorageConfig(StrictModel):
    artifacts_dir: Path = Path("artifacts/v2")
    reports_dir: Path = Path("reports/v2")
    receipts_cache_dir: Path = Path("data/raw")
    odds_dir: Path = Path("data/odds")


class V2EvidenceConfig(StrictModel):
    mode: EvidenceMode = "research_reconstruction"
    synthetic: bool = False


class V2ModelsConfig(StrictModel):
    champion_model_id: str = "M1_ridge_alpha100"
    champion_bundle_hash: str | None = None
    challenger_decision_ref: str | None = None  # research run id holding the decision record
    challenger_model_id: str | None = None
    challenger_bundle_hash: str | None = None
    score_alpha: float = 100.0


class V2ChallengerConfig(StrictModel):
    features: tuple[str, ...] = ("tie", "abs_margin_3", "abs_margin_7", "total_even")
    lambdas: tuple[float, ...] = (0.01, 0.1, 1.0)
    bound: float = 3.0
    min_calibration_games: int = 250
    max_calibration_seasons: int = 3
    development_seasons: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)
    retrospective_check_seasons: tuple[int, ...] = (2024, 2025)
    crps_tie_tolerance: float = 1e-6
    max_worse_fraction: float = 0.01


class V2HorizonConfig(StrictModel):
    target_hours_before_kickoff: float = 24.0
    window_minutes: float = 5.0
    horizon_policy_id: str = "kickoff_minus_24h_pm5m"


class V2SourcesConfig(StrictModel):
    completed_game_lag_hours: float = 48.0
    required_datasets: tuple[str, ...] = ("schedules", "pbp")
    allow_unknown_availability_for_research: bool = True


class V2MarketConfig(StrictModel):
    enabled: bool = False
    provider: Literal["csv", "the_odds_api"] = "csv"
    bookmaker: str | None = None
    bookmakers: tuple[str, ...] = ()
    settlement_rules: dict[str, dict[str, str]] = {}  # bookmaker -> market -> rule
    max_quote_age_minutes: float = 60.0
    main_markets: tuple[str, ...] = ("h2h", "spreads", "totals")


class V2CollectionConfig(StrictModel):
    timeout_seconds: float = 20.0
    max_attempts_per_request: int = 3
    request_budget_per_invocation: int = 4
    quota_reserve: int = 50
    sport_key: str = "americanfootball_nfl"
    regions: str = "us"


class V2ClosingProxyConfig(StrictModel):
    window_minutes: float = 30.0


class V2PaperConfig(StrictModel):
    min_ev: float = 0.03
    stake_units: float = 1.0
    max_selections_per_game_per_model: int = 1
    market_order: tuple[str, ...] = ("moneyline", "spread", "total")


class V2EvaluationConfig(StrictModel):
    bootstrap_replicates: int = 2000
    seed: int = 42
    min_blocks: int = 8
    planned_review_games: int = 100
    planned_review_blocks: int = 8
    log_loss_epsilon: float = 1e-12


class V2Config(StrictModel):
    storage: V2StorageConfig = V2StorageConfig()
    evidence: V2EvidenceConfig = V2EvidenceConfig()
    models: V2ModelsConfig = V2ModelsConfig()
    challenger: V2ChallengerConfig = V2ChallengerConfig()
    horizon: V2HorizonConfig = V2HorizonConfig()
    sources: V2SourcesConfig = V2SourcesConfig()
    market: V2MarketConfig = V2MarketConfig()
    collection: V2CollectionConfig = V2CollectionConfig()
    closing_proxy: V2ClosingProxyConfig = V2ClosingProxyConfig()
    paper: V2PaperConfig = V2PaperConfig()
    evaluation: V2EvaluationConfig = V2EvaluationConfig()

    @model_validator(mode="after")
    def _check_market(self) -> V2Config:
        m = self.market
        if m.enabled:
            books = tuple(m.bookmakers) or ((m.bookmaker,) if m.bookmaker else ())
            if not books:
                raise ValueError(
                    "v2.market.enabled requires an explicit bookmaker (v2.market.bookmaker or "
                    "v2.market.bookmakers); no default sportsbook is assumed"
                )
            if m.bookmaker is None:
                raise ValueError("v2.market.bookmaker (the reference book) must be explicit")
            for book in books:
                rules = m.settlement_rules.get(book)
                if not rules:
                    raise ValueError(
                        f"v2.market.settlement_rules must document rules for bookmaker {book!r} "
                        "(e.g. moneyline: two_way_tie_void, spread: push_refund, "
                        "total: push_refund)"
                    )
        if self.models.score_alpha != 100.0:
            raise ValueError(
                "V2 keeps the V1 Ridge alpha fixed at 100 for the distribution experiment"
            )
        if self.challenger.features != ("tie", "abs_margin_3", "abs_margin_7", "total_even"):
            raise ValueError("the challenger feature set is fixed by the V2 protocol")
        if any(lam <= 0 for lam in self.challenger.lambdas):
            raise ValueError("challenger lambdas must be positive")
        return self


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
    v2: V2Config | None = None

    def require_v2(self) -> V2Config:
        if self.v2 is None:
            raise InvalidInputError("this command needs a `v2:` section in the config")
        return self.v2

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
