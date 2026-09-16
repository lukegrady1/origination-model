"""Normalize nflverse schedule and play-by-play into canonical tables (spec sections 5–7).

Betting columns present in the schedule are quarantined into ``untimed_reference`` and never
touch the games/results/feature tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nfl_origination.data.download import PBP_DATASET, SCHEDULE_DATASET, SourceManifest, read_entry
from nfl_origination.errors import ModelValidationError
from nfl_origination.provenance import hash_frame
from nfl_origination.schemas import GAMES_SCHEMA, RESULTS_SCHEMA, validate_frame

# Explicit, tested franchise relocation aliases: original code -> canonical franchise ID.
# Distinct franchises with similar names (LA vs LAC) are never merged.
FRANCHISE_ALIASES: dict[str, str] = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
SCHEDULE_TZ = "America/New_York"
EXPECTED_REG_GAMES = {"pre_2021": 256, "from_2021": 272}

SCHEDULE_REQUIRED = [
    "game_id",
    "season",
    "game_type",
    "week",
    "gameday",
    "gametime",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "location",
    "overtime",
]
SCHEDULE_MARKET_FIELDS = [
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
]

# canonical column -> nflverse source column (adapter detail; verified against the PBP dictionary)
PBP_FIELD_MAP: dict[str, str] = {
    "game_id": "game_id",
    "play_id": "play_id",
    "season": "season",
    "week": "week",
    "season_type": "season_type",
    "posteam": "posteam",
    "defteam": "defteam",
    "qtr": "qtr",
    "play_type": "play_type",
    "epa": "epa",
    "yards_gained": "yards_gained",
    "is_dropback": "pass",  # includes sacks and scrambles per nflfastR definition
    "is_designed_rush": "rush",  # designed rushes only; scrambles are dropbacks
    "is_kneel": "qb_kneel",
    "is_spike": "qb_spike",
    "is_two_point": "two_point_attempt",
    "is_scramble": "qb_scramble",
    "is_sack": "sack",
}
PBP_SOURCE_COLUMNS = list(PBP_FIELD_MAP.values())
PBP_INT_FLAGS = ["is_dropback", "is_designed_rush", "is_kneel", "is_spike", "is_two_point"]


def expected_regular_season_games(season: int) -> int:
    return EXPECTED_REG_GAMES["from_2021"] if season >= 2021 else EXPECTED_REG_GAMES["pre_2021"]


def canonical_team(code: pd.Series) -> pd.Series:
    return code.map(lambda c: FRANCHISE_ALIASES.get(c, c) if isinstance(c, str) else c)


def eastern_to_utc(gameday: pd.Series, gametime: pd.Series) -> pd.Series:
    """Combine schedule date/time strings in Eastern time and convert to UTC with DST awareness."""
    text = gameday.astype("string") + " " + gametime.astype("string")
    naive = pd.to_datetime(text, format="%Y-%m-%d %H:%M", errors="coerce")
    localized = naive.dt.tz_localize(SCHEDULE_TZ, ambiguous="raise", nonexistent="raise")
    return localized.dt.tz_convert("UTC")


@dataclass
class NormalizedSchedule:
    games: pd.DataFrame
    results: pd.DataFrame
    untimed_reference: pd.DataFrame
    exclusions: pd.DataFrame


def normalize_schedule(
    raw: pd.DataFrame,
    seasons: list[int],
    *,
    game_type: str = "REG",
    completed_game_lag_hours: float = 48.0,
) -> NormalizedSchedule:
    missing = [c for c in SCHEDULE_REQUIRED if c not in raw.columns]
    if missing:
        raise ModelValidationError(f"schedule source is missing required fields {missing}")
    df = raw[raw["season"].isin(seasons) & (raw["game_type"] == game_type)].copy()
    df = df.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    excl: list[dict[str, object]] = []
    kickoff = eastern_to_utc(df["gameday"], df["gametime"])
    df["kickoff_utc"] = kickoff
    bad_time = df["kickoff_utc"].isna()
    for _, row in df[bad_time].iterrows():
        excl.append(
            {
                "game_id": row["game_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "reason": "unresolved_kickoff_time",
                "detail": f"gameday={row['gameday']} gametime={row['gametime']}",
            }
        )
    loc = df["location"]
    bad_loc = loc.isna() | ~loc.isin(["Home", "Neutral"])
    for _, row in df[bad_loc & ~bad_time].iterrows():
        excl.append(
            {
                "game_id": row["game_id"],
                "season": int(row["season"]),
                "week": int(row["week"]),
                "reason": "unknown_neutral_status",
                "detail": f"location={row['location']}",
            }
        )
    keep = ~(bad_time | bad_loc)
    df = df[keep].copy()

    games = pd.DataFrame(
        {
            "game_id": df["game_id"].astype(str),
            "season": df["season"].astype(np.int64),
            "week": df["week"].astype(np.int64),
            "game_type": df["game_type"].astype(str),
            "home_team": canonical_team(df["home_team"]).astype(str),
            "away_team": canonical_team(df["away_team"]).astype(str),
            "home_team_original": df["home_team"].astype(str),
            "away_team_original": df["away_team"].astype(str),
            "kickoff_utc": df["kickoff_utc"],
            "neutral_site": (df["location"] == "Neutral").astype(bool),
            "status": np.where(
                df["home_score"].notna() & df["away_score"].notna(), "final", "not_final"
            ),
            "source_dataset": "nflverse_schedules",
            "kickoff_local_text": (df["gameday"].astype(str) + " " + df["gametime"].astype(str)),
            "kickoff_tz": SCHEDULE_TZ,
        }
    ).reset_index(drop=True)
    games["status"] = games["status"].astype(str)
    if (games["home_team"] == games["away_team"]).any():
        raise ModelValidationError(
            "schedule has a game where home and away map to the same franchise"
        )

    final = df[df["home_score"].notna() & df["away_score"].notna()]
    hs = final["home_score"].to_numpy(dtype=float)
    as_ = final["away_score"].to_numpy(dtype=float)
    if (
        (hs < 0).any()
        or (as_ < 0).any()
        or (hs != np.round(hs)).any()
        or (as_ != np.round(as_)).any()
    ):
        raise ModelValidationError("final scores must be nonnegative integers")
    results = pd.DataFrame(
        {
            "game_id": final["game_id"].astype(str),
            "home_score": hs.astype(np.int64),
            "away_score": as_.astype(np.int64),
            "status": "final",
            "eligible_from_utc": final["kickoff_utc"]
            + pd.Timedelta(hours=completed_game_lag_hours),
            "overtime": final["overtime"].fillna(0).astype(int).astype(bool),
        }
    ).reset_index(drop=True)
    results["status"] = results["status"].astype(str)

    ref_cols = [c for c in SCHEDULE_MARKET_FIELDS if c in df.columns]
    untimed = df[["game_id", *ref_cols]].copy().reset_index(drop=True)
    untimed["reference_kind"] = "untimed_reference"
    untimed["game_id"] = untimed["game_id"].astype(str)

    exclusions = pd.DataFrame(excl, columns=["game_id", "season", "week", "reason", "detail"])
    validate_frame(games, GAMES_SCHEMA)
    validate_frame(results, RESULTS_SCHEMA)
    return NormalizedSchedule(games, results, untimed, exclusions)


def normalize_pbp(raw: pd.DataFrame, season: int, *, game_type: str = "REG") -> pd.DataFrame:
    """Map nflverse PBP fields to the canonical play table and check semantic invariants."""
    missing = [src for src in PBP_SOURCE_COLUMNS if src not in raw.columns]
    if missing:
        raise ModelValidationError(f"PBP season {season}: cannot map required fields {missing}")
    src = raw[PBP_SOURCE_COLUMNS]
    plays = src.rename(columns={v: k for k, v in PBP_FIELD_MAP.items()}).copy()
    plays = plays[plays["season_type"] == game_type].copy()
    if (plays["season"] != season).any():
        raise ModelValidationError(f"PBP file for {season} contains other seasons")
    for flag in [*PBP_INT_FLAGS, "is_scramble", "is_sack"]:
        plays[flag] = plays[flag].fillna(0).astype(int).astype(bool)
    plays["is_no_play"] = (plays["play_type"] == "no_play").fillna(False).astype(bool)
    plays["posteam"] = canonical_team(plays["posteam"])
    plays["defteam"] = canonical_team(plays["defteam"])
    plays["game_id"] = plays["game_id"].astype(str)
    plays["play_id"] = plays["play_id"].astype(np.int64)
    plays["qtr"] = plays["qtr"].astype("Int64")
    plays["epa"] = plays["epa"].astype(float)
    plays["yards_gained"] = plays["yards_gained"].astype(float)
    plays["week"] = plays["week"].astype(np.int64)
    plays["season"] = plays["season"].astype(np.int64)

    # Semantic invariants are checked on real scrimmage plays; placeholder rows such as
    # "*** play under review ***" carry neither flag and never enter aggregation.
    live = plays[~plays["is_no_play"] & (plays["is_dropback"] | plays["is_designed_rush"])]
    both = (live["is_dropback"] & live["is_designed_rush"]).sum()
    if both:
        raise ModelValidationError(f"PBP {season}: {both} plays flagged as both dropback and rush")
    sack_not_dropback = (live["is_sack"] & ~live["is_dropback"]).sum()
    scramble_bad = (live["is_scramble"] & (~live["is_dropback"] | live["is_designed_rush"])).sum()
    if sack_not_dropback or scramble_bad:
        raise ModelValidationError(
            f"PBP {season}: sack/scramble semantics violated "
            f"(sacks not dropbacks={sack_not_dropback}, scrambles misflagged={scramble_bad})"
        )
    return plays.reset_index(drop=True)


@dataclass
class NormalizedData:
    games: pd.DataFrame
    results: pd.DataFrame
    untimed_reference: pd.DataFrame
    plays: pd.DataFrame
    exclusions: pd.DataFrame
    source_hashes: dict[str, str] = field(default_factory=dict)
    observed_at: dict[str, str] = field(default_factory=dict)


def normalize_sources(
    manifest: SourceManifest,
    seasons: list[int],
    *,
    game_type: str = "REG",
    completed_game_lag_hours: float = 48.0,
) -> NormalizedData:
    schedule_entry = manifest.entry(SCHEDULE_DATASET, None)
    raw_schedule = read_entry(schedule_entry)
    sched = normalize_schedule(
        raw_schedule,
        seasons,
        game_type=game_type,
        completed_game_lag_hours=completed_game_lag_hours,
    )
    frames: list[pd.DataFrame] = []
    observed: dict[str, str] = {}
    for season in seasons:
        entry = manifest.entry(PBP_DATASET, season)
        raw = read_entry(entry, columns=PBP_SOURCE_COLUMNS)
        plays = normalize_pbp(raw, season, game_type=game_type)
        plays["first_observed_utc"] = pd.Timestamp(entry.first_observed_at_utc)
        observed[f"pbp/{season}"] = entry.first_observed_at_utc
        frames.append(plays)
    all_plays = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    observed["schedules/all"] = schedule_entry.first_observed_at_utc
    sched.games["first_observed_utc"] = pd.Timestamp(schedule_entry.first_observed_at_utc)
    return NormalizedData(
        games=sched.games,
        results=sched.results,
        untimed_reference=sched.untimed_reference,
        plays=all_plays,
        exclusions=sched.exclusions,
        source_hashes=manifest.file_hashes(),
        observed_at=observed,
    )


def games_hash(games: pd.DataFrame) -> str:
    return hash_frame(games.drop(columns=[c for c in ["first_observed_utc"] if c in games.columns]))
