"""Mini raw fixture: nflverse-shaped CSVs -> normalized tables -> team-game aggregates."""

from __future__ import annotations

import pandas as pd

from nfl_origination.data.normalize import normalize_schedule
from nfl_origination.features.aggregate import aggregate_team_games
from nfl_origination.fixtures import MINI_SCHEDULE, load_mini_raw_fixture
from nfl_origination.schemas import GAMES_SCHEMA, RESULTS_SCHEMA, TEAM_GAMES_SCHEMA, validate_frame


def test_mini_fixture_pipeline():
    games, plays, results = load_mini_raw_fixture()
    validate_frame(games, GAMES_SCHEMA)
    validate_frame(results, RESULTS_SCHEMA)
    assert len(games) == 5 and set(games["game_type"]) == {"REG"}
    assert str(games["kickoff_utc"].dtype).endswith(", UTC]")
    tg = aggregate_team_games(games, plays, results)
    validate_frame(tg, TEAM_GAMES_SCHEMA)
    assert len(tg) == 10 and tg["has_pbp"].all()
    assert (tg["points_for"] >= 0).all()


def test_schedule_market_fields_are_quarantined():
    raw = pd.read_csv(MINI_SCHEDULE)
    sched = normalize_schedule(raw, [2023])
    assert "spread_line" not in sched.games.columns
    assert "spread_line" in sched.untimed_reference.columns
    assert (sched.untimed_reference["reference_kind"] == "untimed_reference").all()
    assert sched.exclusions.empty


def test_unresolved_kickoff_and_unknown_neutral_are_excluded():
    raw = pd.read_csv(MINI_SCHEDULE)
    raw.loc[0, "gametime"] = None
    raw.loc[1, "location"] = None
    sched = normalize_schedule(raw, [2023])
    reasons = set(sched.exclusions["reason"])
    assert reasons == {"unresolved_kickoff_time", "unknown_neutral_status"}
    assert len(sched.games) == 3
