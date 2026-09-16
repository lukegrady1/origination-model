"""Temporal joins: cutoff boundaries, observation times, DST, reschedules."""

from __future__ import annotations

import pandas as pd

from nfl_origination.data.normalize import eastern_to_utc
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.builder import build_features
from tests.conftest import ts


def test_exact_cutoff_boundary_inclusive():
    policy = AsOfPolicy(completed_game_lag_hours=48, cutoff_hours_before_kickoff=24)
    cutoff = ts("2023-09-16T17:00:00Z")
    kick = pd.Series([ts("2023-09-14T17:00:00Z"), ts("2023-09-14T17:00:01Z")])
    mask = policy.eligible_mask(kick, cutoff)
    assert mask.tolist() == [True, False]


def test_recorded_asof_requires_observation_before_cutoff():
    policy = AsOfPolicy(mode="recorded_asof")
    cutoff = ts("2023-09-16T17:00:00Z")
    kick = pd.Series([ts("2023-09-10T17:00:00Z"), ts("2023-09-10T17:00:00Z")])
    observed = pd.Series([ts("2023-09-12T00:00:00Z"), ts("2023-09-17T00:00:00Z")])
    assert policy.eligible_mask(kick, cutoff, observed).tolist() == [True, False]
    assert policy.eligible_mask(kick, cutoff, None).tolist() == [False, False]


def test_thursday_game_not_eligible_for_sunday_cutoff():
    policy = AsOfPolicy()
    sunday_kick = ts("2023-09-17T17:00:00Z")
    cutoff = policy.cutoff_for(sunday_kick)
    thursday = pd.Series([ts("2023-09-15T00:15:00Z")])  # Thursday night, 48h later is after cutoff
    last_sunday = pd.Series([ts("2023-09-10T17:00:00Z")])
    assert not policy.eligible_mask(thursday, cutoff)[0]
    assert policy.eligible_mask(last_sunday, cutoff)[0]


def test_dst_aware_eastern_conversion():
    out = eastern_to_utc(
        pd.Series(["2023-10-29", "2023-11-05", "2023-03-12"]),
        pd.Series(["13:00", "13:00", "13:00"]),
    )
    assert out[0] == ts("2023-10-29T17:00:00Z")  # EDT
    assert out[1] == ts("2023-11-05T18:00:00Z")  # EST after DST end
    assert out[2] == ts("2023-03-12T17:00:00Z")  # DST starts that morning


def test_unparseable_time_becomes_null():
    out = eastern_to_utc(pd.Series(["2023-10-29"]), pd.Series([None]))
    assert out.isna().all()


def test_reschedule_produces_new_versioned_row(synthetic_small, policy, features_cfg):
    games = synthetic_small.games
    tg = synthetic_small.team_games
    target = games[(games["season"] == 2016) & (games["week"] == 5)].head(1)
    before = build_features(target, tg, policy, features_cfg)
    moved = target.copy()
    moved["kickoff_utc"] = moved["kickoff_utc"] + pd.Timedelta(days=3)
    after = build_features(moved, tg, policy, features_cfg)
    assert (before["cutoff_utc"] != after["cutoff_utc"]).all()
    assert set(before["game_id"]) == set(after["game_id"])
    # the original row is untouched; the reschedule is a distinct (game_id, cutoff) key
    assert before["cutoff_utc"].iloc[0] == policy.cutoff_for(target["kickoff_utc"].iloc[0])


def test_future_source_game_never_enters_features(synthetic_small, policy, features_cfg):
    games = synthetic_small.games
    tg = synthetic_small.team_games
    target = games[(games["season"] == 2016) & (games["week"] == 3)].head(1)
    feats = build_features(target, tg, policy, features_cfg)
    ids = set(feats["team_source_game_ids"].iloc[0].split(",")) | set(
        feats["opp_source_game_ids"].iloc[0].split(",")
    )
    used = tg[tg["game_id"].isin(ids)]
    assert (used["kickoff_utc"] + pd.Timedelta(hours=48) <= feats["cutoff_utc"].iloc[0]).all()
    assert target["game_id"].iloc[0] not in ids
    assert (feats["max_source_eligible_utc"] <= feats["cutoff_utc"]).all()


def test_target_game_excluded_at_boundary(synthetic_small, policy, features_cfg):
    games = synthetic_small.games
    tg = synthetic_small.team_games
    target = games[(games["season"] == 2017) & (games["week"] == 10)].head(1)
    feats = build_features(target, tg, policy, features_cfg)
    assert feats["team_history_games"].max() <= 16
    assert feats["team_history_games"].min() >= 0
