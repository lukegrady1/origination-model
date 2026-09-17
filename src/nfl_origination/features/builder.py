"""As-of feature builder producing two perspective rows per game (spec section 8).

Features depend only on source games eligible before each game's own cutoff. Labels live in a
separate table and are joined only at fit/evaluation time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_origination.config import FeaturesConfig
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.rolling import ShrinkageParams, league_priors, rolling_metrics
from nfl_origination.provenance import hash_frame
from nfl_origination.schemas import (
    FEATURE_VERSION,
    FEATURES_SCHEMA,
    LABELS_SCHEMA,
    RATE_METRICS,
    assert_no_market_columns,
    validate_frame,
)

OFFENSE_METRIC_TO_FEATURE = {
    "off_epa_per_play": "team_off_epa_per_play",
    "off_success_rate": "team_off_success_rate",
    "off_dropback_epa": "team_off_dropback_epa",
    "off_rush_epa": "team_off_rush_epa",
    "off_explosive_rate": "team_off_explosive_rate",
    "plays_per_game": "team_plays_per_game",
    "points_for": "team_points_for",
}
DEFENSE_METRIC_TO_FEATURE = {
    "def_epa_allowed": "opp_def_epa_allowed",
    "def_success_allowed": "opp_def_success_allowed",
    "def_dropback_epa_allowed": "opp_def_dropback_epa_allowed",
    "def_rush_epa_allowed": "opp_def_rush_epa_allowed",
    "def_explosive_allowed": "opp_def_explosive_allowed",
    "points_against": "opp_points_against",
    "plays_per_game": "opp_plays_per_game",
}


@dataclass
class TeamSnapshot:
    metrics: dict[str, float]
    history_games: int
    cold_start: bool
    rest_days: float
    rest_missing: bool
    source_game_ids: list[str]
    max_source_eligible_utc: pd.Timestamp | None
    max_observed_utc: pd.Timestamp | None
    games_without_pbp: int


class TeamHistoryIndex:
    """Per-team completed team-game history sorted by kickoff for fast as-of slicing."""

    def __init__(self, team_games: pd.DataFrame, policy: AsOfPolicy) -> None:
        self.policy = policy
        self.by_team: dict[str, pd.DataFrame] = {}
        tg = team_games.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
        tg["eligible_from_utc"] = policy.eligible_from(tg["kickoff_utc"])
        for team, part in tg.groupby("team_id", sort=False):
            self.by_team[str(team)] = part.reset_index(drop=True)

    def eligible_history(self, team: str, cutoff: pd.Timestamp) -> pd.DataFrame:
        part = self.by_team.get(team)
        if part is None or part.empty:
            return pd.DataFrame(columns=["game_id", "season", "kickoff_utc"])
        first_observed = (
            part["first_observed_utc"] if "first_observed_utc" in part.columns else None
        )
        mask = self.policy.eligible_mask(part["kickoff_utc"], cutoff, first_observed)
        return part[mask]

    def prior_kickoff(self, team: str, cutoff: pd.Timestamp, season: int) -> pd.Timestamp | None:
        """Most recent same-season game that has already kicked off before the cutoff.

        Rest is schedule knowledge; in ``recorded_asof`` the schedule row itself must have been
        observed at or before the cutoff.
        """
        part = self.by_team.get(team)
        if part is None or part.empty:
            return None
        known = (part["kickoff_utc"] < cutoff) & (part["season"] == season)
        if self.policy.mode == "recorded_asof":
            col = (
                "schedule_first_observed_utc"
                if "schedule_first_observed_utc" in part.columns
                else "first_observed_utc"
            )
            if col not in part.columns:
                return None
            known &= (part[col] <= cutoff).fillna(False)
        prior = part[known]
        if prior.empty:
            return None
        return pd.Timestamp(prior["kickoff_utc"].iloc[-1])


def _snapshot(
    index: TeamHistoryIndex,
    team: str,
    season: int,
    cutoff: pd.Timestamp,
    kickoff: pd.Timestamp,
    cfg: FeaturesConfig,
    priors: dict[str, float],
) -> TeamSnapshot:
    hist = index.eligible_history(team, cutoff)
    recent = hist.tail(cfg.history_games)
    metrics = rolling_metrics(
        recent,
        season,
        history_games=cfg.history_games,
        half_life=cfg.half_life_games,
        offseason_weight=cfg.offseason_weight,
        shrinkage=ShrinkageParams(
            cfg.shrinkage_plays, cfg.shrinkage_split_plays, cfg.shrinkage_games
        ),
        priors=priors,
    )
    prior_kick = index.prior_kickoff(team, cutoff, season)
    if prior_kick is None:
        rest, rest_missing = cfg.rest_default_days, True
    else:
        rest = (kickoff - prior_kick).total_seconds() / 86400.0
        rest = float(min(max(rest, cfg.rest_min_days), cfg.rest_max_days))
        rest_missing = False
    max_elig = pd.Timestamp(recent["eligible_from_utc"].max()) if len(recent) else None
    max_obs = None
    if len(recent) and "first_observed_utc" in recent.columns:
        max_obs = pd.Timestamp(recent["first_observed_utc"].max())
    without_pbp = (
        int((~recent["has_pbp"]).sum()) if "has_pbp" in recent.columns and len(recent) else 0
    )
    return TeamSnapshot(
        metrics=metrics,
        history_games=int(min(len(hist), cfg.history_games)),
        cold_start=len(hist) == 0,
        rest_days=rest,
        rest_missing=rest_missing,
        source_game_ids=[str(g) for g in recent["game_id"].tolist()],
        max_source_eligible_utc=max_elig,
        max_observed_utc=max_obs,
        games_without_pbp=without_pbp,
    )


@dataclass
class PriorSnapshot:
    priors: dict[str, float]
    seasons: list[int]
    games_hash: str
    max_source_eligible_utc: pd.Timestamp | None
    max_observed_utc: pd.Timestamp | None
    fully_eligible: bool


def _nan_priors() -> dict[str, float]:
    return {m: float("nan") for m in RATE_METRICS}


def season_priors(
    team_games: pd.DataFrame,
    target_season: int,
    prior_seasons: int,
    *,
    policy: AsOfPolicy | None = None,
    cutoff: pd.Timestamp | None = None,
) -> PriorSnapshot:
    """League priors from the preceding seasons, restricted to rows eligible at ``cutoff``.

    Priors are feature inputs too: under ``recorded_asof`` only observed rows may enter them.
    """
    prior_rows = team_games[team_games["season"] < target_season]
    if policy is not None and cutoff is not None and len(prior_rows):
        first_observed = (
            prior_rows["first_observed_utc"] if "first_observed_utc" in prior_rows.columns else None
        )
        mask = policy.eligible_mask(prior_rows["kickoff_utc"], cutoff, first_observed)
        fully = bool(mask.all())
        prior_rows = prior_rows[mask]
    else:
        fully = True
    available = sorted(int(s) for s in prior_rows["season"].unique())
    use = [s for s in available if s >= target_season - prior_seasons]
    if not use:  # warm-up only: fall back to earlier available seasons
        use = available[-prior_seasons:] if available else []
    if not use:
        return PriorSnapshot(_nan_priors(), [], "", None, None, fully)
    used = prior_rows[prior_rows["season"].isin(use)]
    ids = sorted(set(used["game_id"].astype(str)))
    max_elig = None
    max_obs = None
    if policy is not None and len(used):
        max_elig = pd.Timestamp(policy.eligible_from(used["kickoff_utc"]).max())
        if "first_observed_utc" in used.columns and used["first_observed_utc"].notna().any():
            max_obs = pd.Timestamp(used["first_observed_utc"].max())
    return PriorSnapshot(
        league_priors(used, use),
        use,
        hashlib.sha256(",".join(ids).encode()).hexdigest(),
        max_elig,
        max_obs,
        fully,
    )


def build_features(
    games: pd.DataFrame,
    team_games: pd.DataFrame,
    policy: AsOfPolicy,
    cfg: FeaturesConfig,
    *,
    cutoff_override_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build feature rows for every game in ``games`` at its own cutoff.

    ``cutoff_override_utc`` fixes one explicit research time for all games (custom horizon).
    """
    assert_no_market_columns(list(games.columns), "build_features(games)")
    assert_no_market_columns(list(team_games.columns), "build_features(team_games)")
    index = TeamHistoryIndex(team_games, policy)
    priors_cache: dict[tuple[int, pd.Timestamp | None], PriorSnapshot] = {}
    records: list[dict[str, object]] = []
    has_game_observation = "first_observed_utc" in games.columns
    for game in games.sort_values(["kickoff_utc", "game_id"]).itertuples(index=False):
        kickoff = pd.Timestamp(game.kickoff_utc)
        season = int(game.season)
        cutoff = (
            cutoff_override_utc if cutoff_override_utc is not None else policy.cutoff_for(kickoff)
        )
        cutoff = pd.Timestamp(cutoff)
        # priors depend on the cutoff whenever some prior-season rows are not yet eligible;
        # once every prior row is eligible the snapshot is identical for the whole season
        snap = priors_cache.get((season, None))
        if snap is None:
            snap = priors_cache.get((season, cutoff))
        if snap is None:
            snap = season_priors(
                team_games, season, cfg.prior_seasons, policy=policy, cutoff=cutoff
            )
            priors_cache[(season, None if snap.fully_eligible else cutoff)] = snap
        priors, prior_seasons_used = snap.priors, snap.seasons
        insufficient = len(prior_seasons_used) == 0
        unobserved = False
        if policy.mode == "recorded_asof":
            if not has_game_observation:
                unobserved = True
            else:
                observed_at = getattr(game, "first_observed_utc", pd.NaT)
                unobserved = pd.isna(observed_at) or pd.Timestamp(observed_at) > cutoff
        home = _snapshot(index, str(game.home_team), season, cutoff, kickoff, cfg, priors)
        away = _snapshot(index, str(game.away_team), season, cutoff, kickoff, cfg, priors)
        neutral = bool(game.neutral_site)
        for perspective, team, opp, tsnap, osnap, venue in (
            ("home", game.home_team, game.away_team, home, away, 0.0 if neutral else 1.0),
            ("away", game.away_team, game.home_team, away, home, 0.0 if neutral else -1.0),
        ):
            row: dict[str, object] = {
                "game_id": str(game.game_id),
                "cutoff_utc": cutoff,
                "perspective": perspective,
                "season": season,
                "week": int(game.week),
                "team_id": str(team),
                "opponent_id": str(opp),
                "kickoff_utc": kickoff,
                "data_mode": policy.mode,
                "feature_version": FEATURE_VERSION,
                "venue_advantage": venue,
                "rest_difference": tsnap.rest_days - osnap.rest_days,
                "team_rest_missing": float(tsnap.rest_missing),
                "opp_rest_missing": float(osnap.rest_missing),
                "team_history_games": float(tsnap.history_games),
                "opp_history_games": float(osnap.history_games),
                "team_cold_start": float(tsnap.cold_start),
                "opp_cold_start": float(osnap.cold_start),
                "insufficient_warmup": insufficient,
                "unobserved_inputs": unobserved,
                "prior_seasons_used": ",".join(map(str, prior_seasons_used)),
                "prior_games_hash": snap.games_hash,
                "team_source_game_ids": ",".join(tsnap.source_game_ids),
                "opp_source_game_ids": ",".join(osnap.source_game_ids),
                "games_without_pbp": tsnap.games_without_pbp + osnap.games_without_pbp,
            }
            for metric, col in OFFENSE_METRIC_TO_FEATURE.items():
                row[col] = tsnap.metrics[metric]
            for metric, col in DEFENSE_METRIC_TO_FEATURE.items():
                row[col] = osnap.metrics[metric]
            elig = [
                t
                for t in (
                    tsnap.max_source_eligible_utc,
                    osnap.max_source_eligible_utc,
                    snap.max_source_eligible_utc,
                )
                if t is not None
            ]
            row["max_source_eligible_utc"] = max(elig) if elig else pd.NaT
            obs = [
                t
                for t in (tsnap.max_observed_utc, osnap.max_observed_utc, snap.max_observed_utc)
                if t is not None
            ]
            if has_game_observation and pd.notna(getattr(game, "first_observed_utc", pd.NaT)):
                obs.append(pd.Timestamp(game.first_observed_utc))
            row["max_observed_utc"] = max(obs) if obs else pd.NaT
            ids = sorted(set(tsnap.source_game_ids) | set(osnap.source_game_ids))
            row["source_games_hash"] = hashlib.sha256(",".join(ids).encode()).hexdigest()
            records.append(row)
    if not records:
        return _empty_features()
    feats = pd.DataFrame.from_records(records)
    feats["cutoff_utc"] = pd.to_datetime(feats["cutoff_utc"], utc=True)
    feats["kickoff_utc"] = pd.to_datetime(feats["kickoff_utc"], utc=True)
    feats["max_source_eligible_utc"] = pd.to_datetime(feats["max_source_eligible_utc"], utc=True)
    feats["max_observed_utc"] = pd.to_datetime(feats["max_observed_utc"], utc=True)
    feats["insufficient_warmup"] = feats["insufficient_warmup"].astype(bool)
    feats["unobserved_inputs"] = feats["unobserved_inputs"].astype(bool)
    validate_frame(feats, FEATURES_SCHEMA)
    return feats


def _empty_features() -> pd.DataFrame:
    cols: dict[str, object] = {}
    for c, kind in FEATURES_SCHEMA.required.items():
        dtype = {"str": "object", "int": "int64", "float": "float64", "bool": "bool"}.get(kind)
        cols[c] = pd.Series(dtype=dtype) if dtype else pd.Series(dtype="datetime64[ns, UTC]")
    return pd.DataFrame(cols)


def build_labels(games: pd.DataFrame, results: pd.DataFrame, policy: AsOfPolicy) -> pd.DataFrame:
    """Labels are separate from features and carry their own availability time.

    Historical reconstruction uses ``kickoff + lag``; ``recorded_asof`` additionally requires the
    result's source version to have been observed, so availability is the later of the two.
    """
    cols = ["game_id", "home_score", "away_score"]
    merged = games.merge(results[cols], on="game_id", how="inner")
    available = policy.eligible_from(merged["kickoff_utc"])
    if policy.mode == "recorded_asof":
        if "first_observed_utc" not in merged.columns:
            raise MissingDataError("recorded_asof labels need the source observation time")
        observed = pd.to_datetime(merged["first_observed_utc"], utc=True)
        available = pd.concat([available, observed], axis=1).max(axis=1)
        available = available.where(observed.notna(), pd.NaT)
    rows = []
    for perspective, col in (("home", "home_score"), ("away", "away_score")):
        rows.append(
            pd.DataFrame(
                {
                    "game_id": merged["game_id"].astype(str),
                    "perspective": perspective,
                    "points": merged[col].astype(np.int64),
                    "label_available_utc": pd.to_datetime(available, utc=True),
                }
            )
        )
    labels = pd.concat(rows, ignore_index=True)
    labels["perspective"] = labels["perspective"].astype(str)
    if policy.mode == "recorded_asof":
        labels = labels[labels["label_available_utc"].notna()].reset_index(drop=True)
    validate_frame(labels, LABELS_SCHEMA)
    return labels


def features_hash(features: pd.DataFrame) -> str:
    return hash_frame(features)


AVAILABILITY_FIELDS = ("insufficient_warmup", "unobserved_inputs")


def require_availability_fields(features: pd.DataFrame, context: str) -> None:
    """The availability flags are mandatory at every training/forecast boundary."""
    missing = [c for c in (*AVAILABILITY_FIELDS, "feature_version") if c not in features.columns]
    if missing:
        raise ModelValidationError(
            f"{context}: feature rows lack availability fields {missing}; they were built by an "
            f"older feature version and must be rebuilt (current feature version {FEATURE_VERSION})"
        )
    for col in AVAILABILITY_FIELDS:
        if not pd.api.types.is_bool_dtype(features[col]) or features[col].isna().any():
            raise ModelValidationError(
                f"{context}: {col} must contain non-null boolean availability flags"
            )
    if len(features):
        versions = set(features["feature_version"].astype(str))
        if versions != {FEATURE_VERSION}:
            raise ModelValidationError(
                f"{context}: feature rows have feature_version {sorted(versions)}; "
                f"current is {FEATURE_VERSION}"
            )


def usable_rows(features: pd.DataFrame) -> pd.Series:
    """Rows whose every input satisfied the availability policy at the cutoff."""
    require_availability_fields(features, "usable_rows")
    ok = ~features["insufficient_warmup"].astype(bool)
    ok &= ~features["unobserved_inputs"].astype(bool)
    return ok


def require_forecastable(features: pd.DataFrame) -> None:
    require_availability_fields(features, "require_forecastable")
    bad_prior = features[features["insufficient_warmup"]]
    if len(bad_prior):
        raise MissingDataError(
            f"{bad_prior['game_id'].nunique()} games lack an eligible league prior warm-up; "
            "cannot forecast"
        )
    if features["unobserved_inputs"].any():
        n = features[features["unobserved_inputs"]]["game_id"].nunique()
        raise MissingDataError(
            f"{n} games have schedule rows not observed at the cutoff (recorded_asof); "
            "cannot forecast"
        )


def feature_dictionary() -> list[dict[str, str]]:
    """Generated feature dictionary (spec section 8) for docs/data_dictionary.md."""
    rows: list[dict[str, str]] = []
    rate_doc = {
        "off_epa_per_play": ("EPA per play", "sum(off EPA)/count(valid EPA plays)", "k=200 plays"),
        "off_success_rate": ("rate", "count(EPA>0)/count(valid EPA plays)", "k=200 plays"),
        "off_dropback_epa": (
            "EPA per dropback",
            "sum(dropback EPA)/count(valid dropback EPA plays)",
            "k=100",
        ),
        "off_rush_epa": (
            "EPA per designed rush",
            "sum(rush EPA)/count(valid rush EPA plays)",
            "k=100",
        ),
        "off_explosive_rate": (
            "rate",
            "count(dropback>=20yd or rush>=10yd)/count(valid yardage plays)",
            "k=200",
        ),
        "plays_per_game": ("plays", "eligible plays per completed team game", "k=4 games"),
        "points_for": ("points", "final points scored incl. OT/ST/defense", "k=4 games"),
        "def_epa_allowed": (
            "EPA per play",
            "opponent offensive EPA sum/opp valid plays (higher = weaker)",
            "k=200",
        ),
        "def_success_allowed": ("rate", "opponent success count/opp valid plays", "k=200"),
        "def_dropback_epa_allowed": (
            "EPA per dropback",
            "opponent dropback EPA/opp dropbacks",
            "k=100",
        ),
        "def_rush_epa_allowed": ("EPA per rush", "opponent rush EPA/opp designed rushes", "k=100"),
        "def_explosive_allowed": ("rate", "opponent explosive count/opp yardage plays", "k=200"),
        "points_against": ("points", "final points allowed", "k=4 games"),
    }
    for metric, col in OFFENSE_METRIC_TO_FEATURE.items():
        unit, formula, k = rate_doc[metric]
        rows.append(_dict_row(col, unit, formula, k, "team perspective (offense)"))
    for metric, col in DEFENSE_METRIC_TO_FEATURE.items():
        unit, formula, k = rate_doc[metric]
        rows.append(_dict_row(col, unit, formula, k, "opponent perspective"))
    rows += [
        {
            "column": "venue_advantage",
            "unit": "indicator",
            "formula": "+1 home, -1 away, 0 neutral",
            "null_policy": "never null; unknown neutral status fails normalization",
            "source": "schedule.location",
            "availability": "schedule metadata",
        },
        {
            "column": "rest_difference",
            "unit": "days",
            "formula": "clip(team rest,3,14) - clip(opp rest,3,14); season opener uses 7",
            "null_policy": "never null; missing rest flagged",
            "source": "schedule kickoff times before cutoff",
            "availability": "prior game kickoff < cutoff, same season",
        },
        {
            "column": "team_rest_missing",
            "unit": "indicator",
            "formula": "1 if no same-season prior game before cutoff",
            "null_policy": "never null",
            "source": "schedule",
            "availability": "cutoff",
        },
        {
            "column": "opp_rest_missing",
            "unit": "indicator",
            "formula": "as above for opponent",
            "null_policy": "never null",
            "source": "schedule",
            "availability": "cutoff",
        },
        {
            "column": "team_history_games",
            "unit": "games",
            "formula": "min(eligible completed games, 16)",
            "null_policy": "never null",
            "source": "team_games",
            "availability": "kickoff+48h<=cutoff (and observed<=cutoff in recorded_asof)",
        },
        {
            "column": "opp_history_games",
            "unit": "games",
            "formula": "as above for opponent",
            "null_policy": "never null",
            "source": "team_games",
            "availability": "as above",
        },
        {
            "column": "team_cold_start",
            "unit": "indicator",
            "formula": "1 if zero eligible history games (prior only)",
            "null_policy": "never null",
            "source": "team_games",
            "availability": "as above",
        },
        {
            "column": "opp_cold_start",
            "unit": "indicator",
            "formula": "as above for opponent",
            "null_policy": "never null",
            "source": "team_games",
            "availability": "as above",
        },
    ]
    return rows


def _dict_row(col: str, unit: str, formula: str, k: str, perspective: str) -> dict[str, str]:
    return {
        "column": col,
        "unit": unit,
        "formula": (
            f"shrunk ({k}) recency-weighted (half-life 8 games, offseason x0.5, last 16): {formula}"
        ),
        "null_policy": "prior r0 when no history (cold start); null only if prior unavailable",
        "source": f"nflverse PBP epa/yards_gained/pass/rush and schedule scores; {perspective}",
        "availability": (
            "source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff)"
        ),
    }
