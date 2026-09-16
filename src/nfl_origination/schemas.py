"""Versioned canonical data contracts (spec section 7) and feature allowlists.

Every artifact has a named schema with key columns and required columns. Validation is
explicit and raises typed errors instead of silently coercing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nfl_origination.errors import ModelValidationError

SCHEMA_VERSION = 2  # v2: features carry unobserved_inputs and prior provenance
FEATURE_VERSION = "2"  # v2: priors/rest/labels/observation times obey the as-of policy

# Market-derived columns are quarantined and can never be model inputs.
MARKET_COLUMN_PREFIXES = ("market_", "ref_", "book_", "quote_")
MARKET_COLUMN_NAMES = frozenset(
    {
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
        "decimal_odds",
        "line",
        "closing_line",
        "opening_line",
    }
)

# Approved model feature columns (spec section 8, "Model row"), in fixed order.
TEAM_OFFENSE_FEATURES = [
    "team_off_epa_per_play",
    "team_off_success_rate",
    "team_off_dropback_epa",
    "team_off_rush_epa",
    "team_off_explosive_rate",
    "team_plays_per_game",
    "team_points_for",
]
OPP_DEFENSE_FEATURES = [
    "opp_def_epa_allowed",
    "opp_def_success_allowed",
    "opp_def_dropback_epa_allowed",
    "opp_def_rush_epa_allowed",
    "opp_def_explosive_allowed",
    "opp_points_against",
]
CONTEXT_FEATURES = [
    "opp_plays_per_game",
    "venue_advantage",
    "rest_difference",
    "team_rest_missing",
    "opp_rest_missing",
    "team_history_games",
    "opp_history_games",
    "team_cold_start",
    "opp_cold_start",
]
FEATURE_COLUMNS_FULL = TEAM_OFFENSE_FEATURES + OPP_DEFENSE_FEATURES + CONTEXT_FEATURES

EPA_FEATURES = {
    "team_off_epa_per_play",
    "team_off_success_rate",
    "team_off_dropback_epa",
    "team_off_rush_epa",
    "opp_def_epa_allowed",
    "opp_def_success_allowed",
    "opp_def_dropback_epa_allowed",
    "opp_def_rush_epa_allowed",
}
FEATURE_COLUMNS_EPA_FREE = [c for c in FEATURE_COLUMNS_FULL if c not in EPA_FEATURES]

# Metrics that must have <=5% unexpected nulls in training (spec section 8).
REQUIRED_METRIC_FEATURES = TEAM_OFFENSE_FEATURES + OPP_DEFENSE_FEATURES + ["opp_plays_per_game"]

FEATURE_SETS = {"full": FEATURE_COLUMNS_FULL, "epa_free": FEATURE_COLUMNS_EPA_FREE}

# Rolling metric names shared between team_games aggregation and feature rolling.
RATE_METRICS = {
    # name: (numerator column, denominator column, shrinkage kind)
    "off_epa_per_play": ("off_epa_sum", "off_epa_plays", "plays"),
    "off_success_rate": ("off_success_count", "off_epa_plays", "plays"),
    "off_dropback_epa": ("off_dropback_epa_sum", "off_dropback_epa_plays", "split"),
    "off_rush_epa": ("off_rush_epa_sum", "off_rush_epa_plays", "split"),
    "off_explosive_rate": ("off_explosive_count", "off_yard_plays", "plays"),
    "def_epa_allowed": ("def_epa_sum", "def_epa_plays", "plays"),
    "def_success_allowed": ("def_success_count", "def_epa_plays", "plays"),
    "def_dropback_epa_allowed": ("def_dropback_epa_sum", "def_dropback_epa_plays", "split"),
    "def_rush_epa_allowed": ("def_rush_epa_sum", "def_rush_epa_plays", "split"),
    "def_explosive_allowed": ("def_explosive_count", "def_yard_plays", "plays"),
    "plays_per_game": ("off_eligible_plays", "game_count", "games"),
    "points_for": ("points_for", "game_count", "games"),
    "points_against": ("points_against", "game_count", "games"),
}

TEAM_GAME_NUMERIC_COLUMNS = sorted(
    {num for num, _den, _k in RATE_METRICS.values()}
    | {den for _n, den, _k in RATE_METRICS.values()}
)


@dataclass(frozen=True)
class ArtifactSchema:
    name: str
    version: int
    keys: list[str]
    required: dict[str, str]  # column -> dtype kind: str|int|float|bool|datetime
    optional: dict[str, str] = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return list(self.required) + list(self.optional)


GAMES_SCHEMA = ArtifactSchema(
    name="games",
    version=SCHEMA_VERSION,
    keys=["game_id"],
    required={
        "game_id": "str",
        "season": "int",
        "week": "int",
        "game_type": "str",
        "home_team": "str",
        "away_team": "str",
        "home_team_original": "str",
        "away_team_original": "str",
        "kickoff_utc": "datetime",
        "neutral_site": "bool",
        "status": "str",
        "source_dataset": "str",
    },
    optional={"kickoff_local_text": "str", "kickoff_tz": "str"},
)

RESULTS_SCHEMA = ArtifactSchema(
    name="results",
    version=SCHEMA_VERSION,
    keys=["game_id"],
    required={
        "game_id": "str",
        "home_score": "int",
        "away_score": "int",
        "status": "str",
        "eligible_from_utc": "datetime",
        "overtime": "bool",
    },
)

TEAM_GAMES_SCHEMA = ArtifactSchema(
    name="team_games",
    version=SCHEMA_VERSION,
    keys=["game_id", "team_id"],
    required={
        "game_id": "str",
        "team_id": "str",
        "opponent_id": "str",
        "season": "int",
        "week": "int",
        "kickoff_utc": "datetime",
        "is_home": "bool",
        "neutral_site": "bool",
        "points_for": "int",
        "points_against": "int",
        "game_count": "int",
        **{
            c: "float"
            for c in TEAM_GAME_NUMERIC_COLUMNS
            if c not in {"points_for", "points_against", "game_count"}
        },
        "source_hash": "str",
    },
)

FEATURES_SCHEMA = ArtifactSchema(
    name="features",
    version=SCHEMA_VERSION,
    keys=["game_id", "cutoff_utc", "perspective"],
    required={
        "game_id": "str",
        "cutoff_utc": "datetime",
        "perspective": "str",
        "season": "int",
        "week": "int",
        "team_id": "str",
        "opponent_id": "str",
        "kickoff_utc": "datetime",
        "data_mode": "str",
        "max_source_eligible_utc": "datetime",
        "source_games_hash": "str",
        "feature_version": "str",
        **{c: "float" for c in FEATURE_COLUMNS_FULL},
        "insufficient_warmup": "bool",
        "unobserved_inputs": "bool",
        "prior_games_hash": "str",
    },
    optional={
        "max_observed_utc": "datetime",
        "team_source_game_ids": "str",
        "opp_source_game_ids": "str",
    },
)

LABELS_SCHEMA = ArtifactSchema(
    name="labels",
    version=SCHEMA_VERSION,
    keys=["game_id", "perspective"],
    required={
        "game_id": "str",
        "perspective": "str",
        "points": "int",
        "label_available_utc": "datetime",
    },
)

PREDICTIONS_SCHEMA = ArtifactSchema(
    name="predictions",
    version=SCHEMA_VERSION,
    keys=["run_id", "game_id", "cutoff_utc"],
    required={
        "run_id": "str",
        "game_id": "str",
        "cutoff_utc": "datetime",
        "season": "int",
        "week": "int",
        "home_team": "str",
        "away_team": "str",
        "kickoff_utc": "datetime",
        "model_id": "str",
        "data_mode": "str",
        "forecast_policy": "str",
        "mu_home_score": "float",
        "mu_away_score": "float",
        "dist_mean_home_score": "float",
        "dist_mean_away_score": "float",
        "mean_margin": "float",
        "mean_total": "float",
        "p_home_win": "float",
        "p_tie": "float",
        "p_away_win": "float",
        "p_home_win_given_no_tie": "float",
        "fair_decimal_home": "float",
        "fair_decimal_away": "float",
        "fair_american_home": "float",
        "fair_american_away": "float",
        "mean_home_handicap": "float",
        "fair_home_handicap": "float",
        "fair_home_handicap_p_win": "float",
        "fair_home_handicap_p_push": "float",
        "fair_home_handicap_p_loss": "float",
        "fair_total": "float",
        "fair_total_p_over": "float",
        "fair_total_p_push": "float",
        "fair_total_p_under": "float",
        "margin_q50_lo": "int",
        "margin_q50_hi": "int",
        "margin_q80_lo": "int",
        "margin_q80_hi": "int",
        "margin_q95_lo": "int",
        "margin_q95_hi": "int",
        "total_q50_lo": "int",
        "total_q50_hi": "int",
        "total_q80_lo": "int",
        "total_q80_hi": "int",
        "total_q95_lo": "int",
        "total_q95_hi": "int",
        "negative_latent_mass": "float",
        "omitted_upper_mass": "float",
        "max_score_support": "int",
        "clipped_negative_cells": "int",
        "warnings": "str",
        "data_quality_flags": "str",
        "schema_version": "int",
        "feature_version": "str",
        "config_hash": "str",
        "features_hash": "str",
    },
)

ODDS_SCHEMA = ArtifactSchema(
    name="odds",
    version=SCHEMA_VERSION,
    keys=["quote_id"],
    required={
        "quote_id": "str",
        "provider_event_id": "str",
        "game_id": "str",
        "bookmaker": "str",
        "market": "str",
        "selection": "str",
        "line": "float",
        "decimal_odds": "float",
        "snapshot_at_utc": "datetime",
        "bookmaker_updated_at_utc": "datetime",
        "kickoff_at_snapshot_utc": "datetime",
        "settlement_rule": "str",
        "ingested_at_utc": "datetime",
        "synthetic": "bool",
    },
)

PAPER_BETS_SCHEMA = ArtifactSchema(
    name="paper_bets",
    version=SCHEMA_VERSION,
    keys=["run_id", "quote_id"],
    required={
        "run_id": "str",
        "quote_id": "str",
        "prediction_id": "str",
        "policy_id": "str",
        "game_id": "str",
        "market": "str",
        "selection": "str",
        "line": "float",
        "decimal_odds": "float",
        "p_win": "float",
        "p_push": "float",
        "p_loss": "float",
        "ev": "float",
        "stake": "float",
        "settlement": "str",
        "profit": "float",
    },
)


def _kind_ok(series: pd.Series, kind: str) -> bool:
    dtype = series.dtype
    if kind == "str":
        return pd.api.types.is_string_dtype(dtype) or dtype is np.dtype(object)
    if kind == "int":
        return pd.api.types.is_integer_dtype(dtype)
    if kind == "float":
        return pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(dtype)
    if kind == "datetime":
        return isinstance(dtype, pd.DatetimeTZDtype) and str(dtype.tz) == "UTC"
    raise ValueError(f"unknown dtype kind {kind}")


def validate_frame(df: pd.DataFrame, schema: ArtifactSchema, *, allow_extra: bool = True) -> None:
    """Validate required columns, dtype kinds, key uniqueness, and non-null keys."""
    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        raise ModelValidationError(f"{schema.name} v{schema.version}: missing columns {missing}")
    bad = [c for c, kind in schema.required.items() if not _kind_ok(df[c], kind)]
    if bad:
        raise ModelValidationError(
            f"{schema.name} v{schema.version}: wrong dtype for {bad}: "
            f"{ {c: str(df[c].dtype) for c in bad} }"
        )
    for c in schema.keys:
        if df[c].isna().any():
            raise ModelValidationError(f"{schema.name}: key column {c} contains nulls")
    if len(df) and df.duplicated(subset=schema.keys).any():
        dups = df[df.duplicated(subset=schema.keys, keep=False)][schema.keys].head(5)
        raise ModelValidationError(f"{schema.name}: duplicate keys on {schema.keys}:\n{dups}")
    if not allow_extra:
        extra = [c for c in df.columns if c not in schema.columns]
        if extra:
            raise ModelValidationError(f"{schema.name}: unexpected extra columns {extra}")


def assert_no_market_columns(columns: list[str] | pd.Index, context: str) -> None:
    """Fail if any market-derived column would enter the feature/model boundary."""
    offenders = [
        c
        for c in columns
        if c in MARKET_COLUMN_NAMES or any(str(c).startswith(p) for p in MARKET_COLUMN_PREFIXES)
    ]
    if offenders:
        raise ModelValidationError(f"{context}: market-derived columns are forbidden: {offenders}")


def feature_columns(feature_set: str) -> list[str]:
    try:
        return list(FEATURE_SETS[feature_set])
    except KeyError as exc:
        raise ModelValidationError(f"unknown feature set {feature_set!r}") from exc


def assert_feature_allowlist(columns: list[str], feature_set: str) -> None:
    """Training inputs must be exactly the approved allowlist for the feature set."""
    allowed = feature_columns(feature_set)
    extra = [c for c in columns if c not in allowed]
    missing = [c for c in allowed if c not in columns]
    if extra or missing:
        raise ModelValidationError(
            f"feature allowlist violation for {feature_set!r}: extra={extra} missing={missing}"
        )


def ensure_utc(series: pd.Series) -> pd.Series:
    """Coerce a datetime series to tz-aware UTC; naive input is rejected."""
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        if pd.api.types.is_datetime64_any_dtype(series):
            raise ModelValidationError(f"column {series.name} is a naive datetime; UTC required")
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
        if parsed.isna().sum() > series.isna().sum():
            raise ModelValidationError(f"column {series.name} has unparseable timestamps")
        return parsed
    return series.dt.tz_convert("UTC")


def float_array(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=np.float64)
