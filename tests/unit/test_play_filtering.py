"""Play filtering and team-game aggregation against the hand-written mini PBP fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_origination.data.normalize import normalize_pbp
from nfl_origination.errors import ModelValidationError
from nfl_origination.features.aggregate import (
    aggregate_offense,
    aggregate_team_games,
    eligible_play_mask,
)
from nfl_origination.fixtures import load_mini_raw_fixture


@pytest.fixture(scope="module")
def mini():
    return load_mini_raw_fixture()


def test_eligible_play_mask_rules(mini):
    _games, plays, _results = mini
    g = plays[plays["game_id"] == "2023_01_AAA_BBB"].copy()
    raw = pd.read_csv("tests/fixtures/synthetic/mini_pbp.csv")[["play_id", "desc"]]
    g = g.merge(raw, on="play_id", how="left")
    g["eligible"] = eligible_play_mask(g)
    by_desc = g.set_index("desc")["eligible"]
    assert by_desc["completed pass"]
    assert by_desc["designed run"]
    assert by_desc["sack counts as dropback"]
    assert by_desc["scramble: dropback, not rush, explosive (>=20)"]
    assert by_desc["missing EPA: yards count, EPA does not; explosive run (>=10)"]
    assert not by_desc["penalty no play excluded"]
    assert not by_desc["kneel excluded"]
    assert not by_desc["spike excluded"]
    assert not by_desc["two point attempt excluded"]
    assert not by_desc["punt excluded"]
    assert not by_desc["field goal excluded"]
    assert not by_desc["overtime excluded from efficiency"]
    assert not by_desc["kickoff no posteam"]


def test_offense_numerators_and_denominators(mini):
    _games, plays, _results = mini
    off = aggregate_offense(plays[plays["game_id"] == "2023_01_AAA_BBB"]).set_index("team_id")
    aaa = off.loc["AAA"]
    assert aaa["off_eligible_plays"] == 5
    assert aaa["off_epa_plays"] == 4  # missing-EPA play excluded from EPA denominators
    assert aaa["off_epa_sum"] == pytest.approx(0.5 - 0.3 - 1.2 + 0.9)
    assert aaa["off_success_count"] == 2
    assert aaa["off_dropback_epa_plays"] == 3  # pass + sack + scramble
    assert aaa["off_dropback_epa_sum"] == pytest.approx(0.5 - 1.2 + 0.9)
    assert aaa["off_rush_epa_plays"] == 1  # designed run only; scramble not double-counted
    assert aaa["off_rush_epa_sum"] == pytest.approx(-0.3)
    assert aaa["off_yard_plays"] == 5
    assert aaa["off_explosive_count"] == 2  # scramble 22yd dropback + 11yd designed run
    bbb = off.loc["BBB"]
    assert bbb["off_success_count"] == 2  # zero EPA is not success
    assert bbb["off_explosive_count"] == 1


def test_team_game_rows_and_defense_mirror(mini):
    games, plays, results = mini
    tg = aggregate_team_games(games, plays, results).set_index(["game_id", "team_id"])
    aaa = tg.loc[("2023_01_AAA_BBB", "AAA")]
    bbb = tg.loc[("2023_01_AAA_BBB", "BBB")]
    assert aaa["points_for"] == 20 and aaa["points_against"] == 27
    assert aaa["def_epa_sum"] == pytest.approx(bbb["off_epa_sum"])
    assert aaa["def_epa_plays"] == bbb["off_epa_plays"]
    assert bool(aaa["is_home"]) is False and bool(bbb["is_home"]) is True
    assert bool(tg.loc[("2023_02_BBB_CCC", "CCC")]["neutral_site"]) is True
    assert "2023_19_AAA_BBB" not in tg.index.get_level_values(0)  # playoff game excluded
    assert len(tg) == 10


def test_pbp_semantic_mapping_fails_on_contradiction(mini):
    raw = pd.read_csv("tests/fixtures/synthetic/mini_pbp.csv")
    bad = raw.copy()
    bad.loc[bad["desc"] == "completed pass", "rush"] = 1  # both dropback and rush
    with pytest.raises(ModelValidationError):
        normalize_pbp(bad, 2023)
    missing = raw.drop(columns=["epa"])
    with pytest.raises(ModelValidationError):
        normalize_pbp(missing, 2023)


def test_missing_epa_does_not_poison_sums(mini):
    _games, plays, _results = mini
    off = aggregate_offense(plays)
    assert np.isfinite(off["off_epa_sum"]).all()
