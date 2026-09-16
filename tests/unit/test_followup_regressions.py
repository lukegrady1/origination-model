"""Regression tests for the follow-up review findings F1 (stale caches) and F2 (unknown times)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from nfl_origination.config import load_config
from nfl_origination.data.download import SourceManifest
from nfl_origination.data.normalize import NormalizedData
from nfl_origination.errors import ModelValidationError
from nfl_origination.experiment import prepare_dataset
from nfl_origination.features.aggregate import aggregate_team_games
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.builder import (
    build_features,
    require_forecastable,
    usable_rows,
)
from nfl_origination.fixtures import load_mini_raw_fixture
from nfl_origination.schemas import FEATURE_COLUMNS_FULL, FEATURE_VERSION
from tests.conftest import ts

ROOT = Path(__file__).resolve().parents[2]
EARLY = ts("2009-01-01T00:00:00Z")
LATE = ts("2030-01-01T00:00:00Z")
NAT = pd.Series([pd.NaT], dtype="datetime64[ns, UTC]").iloc[0]


# ---------------------------------------------------------------------------------------- F1
def _real_source_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "v1.yaml").read_text())
    raw["run"] = {"kind": "development", "label": "f1", "artifacts_dir": str(tmp_path / "a")}
    raw["data"].update(
        {
            "seasons": [2023, 2023],
            "cache_dir": str(tmp_path / "raw"),
            "normalized_dir": str(tmp_path / "normalized"),
            "features_dir": str(tmp_path / "features"),
        }
    )
    raw["model"]["selected_alpha"] = None
    raw["model"]["ablation_selected_alpha"] = None
    raw["evaluation"] = {
        "development_seasons": [2023],
        "confirmation_season": 2024,
        "holdout_season": 2025,
        "bootstrap_replicates": 10,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def mini_source(monkeypatch):
    """Route the real-source dataset path through the committed mini fixture, no network."""
    games, plays, results = load_mini_raw_fixture()
    games = games.copy()
    games["first_observed_utc"] = EARLY
    plays = plays.copy()
    plays["first_observed_utc"] = EARLY
    import nfl_origination.experiment as exp

    manifest = SourceManifest(created_at_utc="x", offline=True, cache_dir="c", entries=[])
    monkeypatch.setattr(exp, "ingest", lambda seasons, cache_dir, offline, refresh=False: manifest)
    monkeypatch.setattr(
        exp,
        "normalize_sources",
        lambda m, seasons, game_type, completed_game_lag_hours: NormalizedData(
            games=games,
            results=results,
            untimed_reference=pd.DataFrame({"game_id": games["game_id"]}),
            plays=plays,
            exclusions=pd.DataFrame(columns=["game_id", "season", "week", "reason", "detail"]),
            source_hashes={"pbp/2023": "h"},
            observed_at={},
        ),
    )
    return games, plays, results


def test_f1_stale_cache_is_invalidated_and_rebuilt(tmp_path, mini_source):
    cfg = load_config(_real_source_config(tmp_path))
    ds = prepare_dataset(cfg, offline=True)  # first load builds the cache
    cache_files = list(Path(cfg.data.features_dir).glob("features_*.parquet"))
    assert len(cache_files) == 1
    path = cache_files[0]
    fresh = pd.read_parquet(path)
    assert (
        "unobserved_inputs" in fresh.columns and (fresh["feature_version"] == FEATURE_VERSION).all()
    )
    # seed an old-format cache under the same key: same sources, same config
    old = fresh.drop(columns=["unobserved_inputs", "prior_games_hash"])
    old["feature_version"] = "1"
    old.to_parquet(path, index=False)
    ds2 = prepare_dataset(cfg, offline=True)  # normal loading path, not build_features
    rebuilt = pd.read_parquet(path)
    assert "unobserved_inputs" in rebuilt.columns
    assert (rebuilt["feature_version"] == FEATURE_VERSION).all()
    assert "unobserved_inputs" in ds2.features.columns
    stale = list(Path(cfg.data.features_dir).glob("features_*.stale-*.parquet"))
    assert len(stale) == 1  # the invalid cache was moved aside, not deleted or reused
    assert (path.with_suffix(".invalidated.json")).exists()
    pd.testing.assert_frame_equal(
        ds.features[FEATURE_COLUMNS_FULL].reset_index(drop=True),
        ds2.features[FEATURE_COLUMNS_FULL].reset_index(drop=True),
    )


def test_f1_cache_from_other_data_mode_is_rejected(tmp_path, mini_source):
    cfg = load_config(_real_source_config(tmp_path))
    prepare_dataset(cfg, offline=True)
    path = next(Path(cfg.data.features_dir).glob("features_*.parquet"))
    wrong = pd.read_parquet(path)
    wrong["data_mode"] = "recorded_asof"
    wrong.to_parquet(path, index=False)
    ds = prepare_dataset(cfg, offline=True)
    assert set(ds.features["data_mode"]) == {"historical_reconstruction"}
    assert len(list(Path(cfg.data.features_dir).glob("*.stale-*.parquet"))) == 1


def test_f1_missing_safety_fields_cannot_pass_the_boundary(synthetic_small, policy, features_cfg):
    games = synthetic_small.games[synthetic_small.games["season"] == 2016].head(2)
    feats = build_features(games, synthetic_small.team_games, policy, features_cfg)
    assert usable_rows(feats).all()
    for col in ("unobserved_inputs", "insufficient_warmup"):
        with pytest.raises(ModelValidationError, match=col):
            usable_rows(feats.drop(columns=[col]))
        with pytest.raises(ModelValidationError, match=col):
            require_forecastable(feats.drop(columns=[col]))
    stale = feats.copy()
    stale["feature_version"] = "1"
    with pytest.raises(ModelValidationError, match="feature_version"):
        usable_rows(stale)


# ---------------------------------------------------------------------------------------- F2
def _mini_with(schedule_at, pbp_at):
    games, plays, results = load_mini_raw_fixture()
    games = games.copy()
    plays = plays.copy()
    games["first_observed_utc"] = pd.Series(
        [schedule_at] * len(games), dtype="datetime64[ns, UTC]"
    ).to_numpy()
    plays["first_observed_utc"] = pd.Series(
        [pbp_at] * len(plays), dtype="datetime64[ns, UTC]"
    ).to_numpy()
    return games, plays, results


@pytest.mark.parametrize(
    "schedule_at,pbp_at,expect_eligible",
    [
        (EARLY, EARLY, True),
        (EARLY, NAT, False),  # PBP observation unknown
        (NAT, EARLY, False),  # schedule observation unknown
        (NAT, NAT, False),
    ],
)
def test_f2_unknown_observation_time_is_ineligible(schedule_at, pbp_at, expect_eligible):
    games, plays, results = _mini_with(schedule_at, pbp_at)
    tg = aggregate_team_games(games, plays, results)
    assert str(tg["first_observed_utc"].dtype) == "datetime64[ns, UTC]"
    policy = AsOfPolicy(mode="recorded_asof")
    mask = policy.eligible_mask(
        tg["kickoff_utc"], ts("2024-06-01T00:00:00Z"), tg["first_observed_utc"]
    )
    assert mask.all() == expect_eligible and mask.any() == expect_eligible
    if not expect_eligible:
        assert tg["first_observed_utc"].isna().all()


def test_f2_missing_pbp_timestamp_column_is_unknown_not_known():
    games, plays, results = load_mini_raw_fixture()
    games = games.copy()
    games["first_observed_utc"] = EARLY
    tg = aggregate_team_games(games, plays, results)  # plays carry no observation column
    assert tg["first_observed_utc"].isna().all()
    assert (tg["schedule_first_observed_utc"] == EARLY).all()


def test_f2_unparseable_or_naive_timestamps_fail_deliberately():
    games, plays, results = load_mini_raw_fixture()
    games = games.copy()
    plays = plays.copy()
    games["first_observed_utc"] = EARLY
    plays["first_observed_utc"] = "not a timestamp"
    with pytest.raises(ModelValidationError, match="unparseable"):
        aggregate_team_games(games, plays, results)
    plays["first_observed_utc"] = pd.Timestamp("2009-01-01")  # naive
    with pytest.raises(ModelValidationError, match="naive"):
        aggregate_team_games(games, plays, results)
    policy = AsOfPolicy(mode="recorded_asof")
    with pytest.raises(ModelValidationError):
        policy.eligible_mask(
            pd.Series([ts("2020-01-01T00:00:00Z")]), ts("2021-01-01T00:00:00Z"), pd.Series(["x"])
        )


def test_f2_unknown_pbp_cannot_influence_an_accepted_forecast(synthetic_small, features_cfg):
    """Aggregation-to-feature path: rows with unknown PBP observation are not used at all."""
    policy = AsOfPolicy(mode="recorded_asof")
    games = synthetic_small.games.copy()
    games["first_observed_utc"] = EARLY
    tg = synthetic_small.team_games.copy()
    tg["schedule_first_observed_utc"] = EARLY
    tg["first_observed_utc"] = EARLY
    target = games[(games["season"] == 2014) & (games["week"] == 6)].head(1)
    base = build_features(target, tg, policy, features_cfg)
    require_forecastable(base)
    team = target["home_team"].iloc[0]
    # make that team's 2014 history "PBP observation unknown": combined time NaT
    unknown = tg.copy()
    sel = (unknown["team_id"] == team) & (unknown["season"] == 2014)
    unknown.loc[sel, "first_observed_utc"] = pd.NaT
    without = build_features(target, unknown, policy, features_cfg)
    require_forecastable(without)  # still forecastable from older observed history
    unknown_ids = set(unknown.loc[sel, "game_id"])
    used = set(without["team_source_game_ids"].iloc[0].split(","))
    assert used and not (used & unknown_ids)  # unknown-observation rows are never source games
    assert set(base["team_source_game_ids"].iloc[0].split(",")) & unknown_ids
    # mutating the unknown-observation rows must not change the accepted forecast
    mutated = unknown.copy()
    mutated.loc[sel, ["off_epa_sum", "points_for"]] += 50.0
    again = build_features(target, mutated, policy, features_cfg)
    pd.testing.assert_frame_equal(
        without[FEATURE_COLUMNS_FULL].reset_index(drop=True),
        again[FEATURE_COLUMNS_FULL].reset_index(drop=True),
    )
    assert not tg.loc[sel].empty
