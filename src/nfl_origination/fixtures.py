"""Committed miniature raw fixtures (nflverse-shaped schedule + PBP CSV) for integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_origination.data.normalize import normalize_pbp, normalize_schedule

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "synthetic"
MINI_SCHEDULE = FIXTURE_DIR / "mini_schedule.csv"
MINI_PBP = FIXTURE_DIR / "mini_pbp.csv"


def load_mini_raw_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize the committed mini schedule and PBP; returns (games, plays, results)."""
    raw_sched = pd.read_csv(MINI_SCHEDULE)
    raw_pbp = pd.read_csv(MINI_PBP)
    seasons = sorted(int(s) for s in raw_sched["season"].unique())
    sched = normalize_schedule(raw_sched, seasons)
    plays = pd.concat(
        [normalize_pbp(raw_pbp[raw_pbp["season"] == s], s) for s in seasons], ignore_index=True
    )
    return sched.games, plays, sched.results
