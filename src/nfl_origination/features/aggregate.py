"""Play filtering and team-game aggregation (spec section 8).

Every metric keeps its numerator and denominator so rolling aggregation is weighted by
opportunities rather than averaging per-game rates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_origination.errors import ModelValidationError
from nfl_origination.provenance import hash_frame
from nfl_origination.schemas import TEAM_GAMES_SCHEMA, validate_frame

SPECIAL_TEAMS_PLAY_TYPES = ("kickoff", "punt", "field_goal", "extra_point")
EXPLOSIVE_DROPBACK_YARDS = 20.0
EXPLOSIVE_RUSH_YARDS = 10.0

OFFENSE_NUMERIC = [
    "off_eligible_plays",
    "off_epa_plays",
    "off_epa_sum",
    "off_success_count",
    "off_dropback_epa_plays",
    "off_dropback_epa_sum",
    "off_rush_epa_plays",
    "off_rush_epa_sum",
    "off_yard_plays",
    "off_explosive_count",
]
DEFENSE_NUMERIC = [
    c.replace("off_", "def_", 1) for c in OFFENSE_NUMERIC if c != "off_eligible_plays"
]


def _utc_or_fail(series: pd.Series, name: str) -> pd.Series:
    """Coerce an observation-time column to tz-aware UTC; unparseable values are an error."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if series.isna().all():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if pd.api.types.is_datetime64_any_dtype(series):
        raise ModelValidationError(f"{name} is a naive datetime; observation times must be UTC")
    try:
        return pd.to_datetime(series, utc=True, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ModelValidationError(f"{name} has unparseable observation timestamps") from exc


def eligible_play_mask(plays: pd.DataFrame) -> pd.Series:
    """Scrimmage plays with valid teams in Q1–Q4 that are a dropback or a designed rush."""
    qtr = plays["qtr"].astype("Float64")
    in_regulation = (qtr >= 1) & (qtr <= 4)
    in_regulation = in_regulation.fillna(False).astype(bool)
    has_teams = plays["posteam"].notna() & plays["defteam"].notna()
    not_special = ~plays["play_type"].isin(SPECIAL_TEAMS_PLAY_TYPES).fillna(False).astype(bool)
    scrimmage = plays["is_dropback"] | plays["is_designed_rush"]
    excluded = plays["is_no_play"] | plays["is_kneel"] | plays["is_spike"] | plays["is_two_point"]
    return (in_regulation & has_teams & not_special & scrimmage & ~excluded).astype(bool)


def aggregate_offense(plays: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, posteam) offensive numerators and denominators from eligible plays."""
    p = plays[eligible_play_mask(plays)].copy()
    epa_ok = np.isfinite(p["epa"].to_numpy(dtype=float))
    yds = p["yards_gained"].to_numpy(dtype=float)
    yds_ok = np.isfinite(yds)
    epa = p["epa"].to_numpy(dtype=float)
    db = p["is_dropback"].to_numpy(dtype=bool)
    ru = p["is_designed_rush"].to_numpy(dtype=bool)
    p["off_eligible_plays"] = 1.0
    p["off_epa_plays"] = epa_ok.astype(float)
    p["off_epa_sum"] = np.where(epa_ok, epa, 0.0)
    p["off_success_count"] = (epa_ok & (epa > 0)).astype(float)
    p["off_dropback_epa_plays"] = (epa_ok & db).astype(float)
    p["off_dropback_epa_sum"] = np.where(epa_ok & db, epa, 0.0)
    p["off_rush_epa_plays"] = (epa_ok & ru).astype(float)
    p["off_rush_epa_sum"] = np.where(epa_ok & ru, epa, 0.0)
    p["off_yard_plays"] = yds_ok.astype(float)
    p["off_explosive_count"] = (
        yds_ok & ((db & (yds >= EXPLOSIVE_DROPBACK_YARDS)) | (ru & (yds >= EXPLOSIVE_RUSH_YARDS)))
    ).astype(float)
    agg = p.groupby(["game_id", "posteam"], as_index=False)[OFFENSE_NUMERIC].sum()
    if "first_observed_utc" in p.columns:
        observed = p.groupby(["game_id", "posteam"], as_index=False)["first_observed_utc"].max()
        observed = observed.rename(columns={"first_observed_utc": "pbp_first_observed_utc"})
        agg = agg.merge(observed, on=["game_id", "posteam"], how="left")
    return agg.rename(columns={"posteam": "team_id"})


def aggregate_team_games(
    games: pd.DataFrame, plays: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    """Build one row per (game, team) for completed games with final scores.

    Games without play-by-play keep points and schedule information with null efficiency
    columns; they are reported in the exclusion audit and contribute no opportunities to rates.
    """
    final = games.merge(results[["game_id", "home_score", "away_score"]], on="game_id", how="inner")
    offense = (
        aggregate_offense(plays) if len(plays) else pd.DataFrame(columns=["game_id", "team_id"])
    )
    rows = []
    for is_home in (True, False):
        team_col, opp_col = ("home_team", "away_team") if is_home else ("away_team", "home_team")
        pf_col, pa_col = ("home_score", "away_score") if is_home else ("away_score", "home_score")
        part = pd.DataFrame(
            {
                "game_id": final["game_id"].astype(str),
                "team_id": final[team_col].astype(str),
                "opponent_id": final[opp_col].astype(str),
                "season": final["season"].astype(np.int64),
                "week": final["week"].astype(np.int64),
                "kickoff_utc": final["kickoff_utc"],
                "is_home": is_home,
                "neutral_site": final["neutral_site"].astype(bool),
                "points_for": final[pf_col].astype(np.int64),
                "points_against": final[pa_col].astype(np.int64),
                "game_count": 1,
            }
        )
        rows.append(part)
    tg = pd.concat(rows, ignore_index=True)
    tg["game_count"] = tg["game_count"].astype(np.int64)
    pbp_observed = None
    if "pbp_first_observed_utc" in offense.columns:
        pbp_observed = offense.groupby("game_id", as_index=False)["pbp_first_observed_utc"].max()
        offense = offense.drop(columns=["pbp_first_observed_utc"])
    tg = tg.merge(offense, on=["game_id", "team_id"], how="left")
    opp_off = offense.rename(columns={"team_id": "opponent_id"}).rename(
        columns={
            c: c.replace("off_", "def_", 1) for c in OFFENSE_NUMERIC if c != "off_eligible_plays"
        }
    )
    opp_off = opp_off.drop(columns=["off_eligible_plays"], errors="ignore")
    tg = tg.merge(opp_off, on=["game_id", "opponent_id"], how="left")
    for c in OFFENSE_NUMERIC + DEFENSE_NUMERIC:
        tg[c] = tg[c].astype(float)
    tg["has_pbp"] = tg["off_eligible_plays"].notna()
    if "first_observed_utc" in games.columns or pbp_observed is not None:
        if "first_observed_utc" in games.columns:
            sched = games[["game_id", "first_observed_utc"]].rename(
                columns={"first_observed_utc": "schedule_first_observed_utc"}
            )
            tg = tg.merge(sched, on="game_id", how="left")
        else:
            tg["schedule_first_observed_utc"] = pd.NaT
        if pbp_observed is not None:
            tg = tg.merge(pbp_observed, on="game_id", how="left")
        else:
            tg["pbp_first_observed_utc"] = pd.NaT
        sched_obs = _utc_or_fail(tg["schedule_first_observed_utc"], "schedule_first_observed_utc")
        pbp_obs = _utc_or_fail(tg["pbp_first_observed_utc"], "pbp_first_observed_utc")
        tg["schedule_first_observed_utc"] = sched_obs
        tg["pbp_first_observed_utc"] = pbp_obs
        # A team-game row is observed only once BOTH its schedule row and its PBP were observed.
        # An unknown observation time for either source leaves the row's time unknown (NaT), which
        # the as-of policy treats as ineligible; one source never stands in for the other.
        both_known = sched_obs.notna() & pbp_obs.notna()
        combined = pd.concat([sched_obs, pbp_obs], axis=1).max(axis=1)
        tg["first_observed_utc"] = pd.to_datetime(combined.where(both_known, pd.NaT), utc=True)
    tg = tg.sort_values(["kickoff_utc", "game_id", "team_id"]).reset_index(drop=True)
    tg["source_hash"] = hash_frame(tg[["game_id", "team_id", "points_for", "points_against"]])
    validate_frame(tg, TEAM_GAMES_SCHEMA)
    dup = tg.duplicated(subset=["game_id", "team_id"]).sum()
    if dup:
        raise ModelValidationError(f"team_games has {dup} duplicate game/team rows")
    return tg
