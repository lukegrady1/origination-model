"""Feature isolation, leakage sentinels, fold grouping, perspectives, bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_origination.config import DistributionConfig
from nfl_origination.errors import ModelValidationError
from nfl_origination.evaluation.splits import (
    assert_grouped_split,
    residual_seasons,
    training_seasons,
)
from nfl_origination.features.builder import build_features, build_labels
from nfl_origination.models.bundle import (
    ModelBundle,
    build_bundle,
    fit_score_model,
    predict_game,
    predict_location,
)
from nfl_origination.models.distribution import fit_residual_params
from nfl_origination.models.ridge import FitSpec, Preprocessor, RidgeScoreModel
from nfl_origination.schemas import (
    FEATURE_COLUMNS_EPA_FREE,
    FEATURE_COLUMNS_FULL,
    assert_feature_allowlist,
    assert_no_market_columns,
)


def _features(ds, policy, cfg, seasons):
    games = ds.games[ds.games["season"].isin(seasons)]
    return build_features(games, ds.team_games, policy, cfg)


def test_market_columns_cannot_enter_features(synthetic_small, policy, features_cfg):
    tg = synthetic_small.team_games.copy()
    tg["spread_line"] = -3.0
    with pytest.raises(ModelValidationError):
        build_features(synthetic_small.games.head(4), tg, policy, features_cfg)
    with pytest.raises(ModelValidationError):
        assert_no_market_columns(["team_off_epa_per_play", "market_total"], "test")


def test_unapproved_column_fails_allowlist():
    with pytest.raises(ModelValidationError):
        assert_feature_allowlist([*FEATURE_COLUMNS_FULL, "home_moneyline"], "full")
    with pytest.raises(ModelValidationError):
        assert_feature_allowlist(FEATURE_COLUMNS_FULL[:-1], "full")
    assert "team_off_epa_per_play" not in FEATURE_COLUMNS_EPA_FREE
    assert "team_off_explosive_rate" in FEATURE_COLUMNS_EPA_FREE


def test_leakage_sentinel_future_mutation_is_invisible(synthetic_small, policy, features_cfg):
    ds = synthetic_small
    games = ds.games[(ds.games["season"] == 2016) & (ds.games["week"] == 6)].head(2)
    base = build_features(games, ds.team_games, policy, features_cfg)
    cutoff = base["cutoff_utc"].min()
    tg = ds.team_games.copy()
    future = tg["kickoff_utc"] + pd.Timedelta(hours=48) > cutoff  # target game and everything after
    tg.loc[future, ["off_epa_sum", "points_for", "points_against", "def_epa_sum"]] *= 5.0
    mutated = build_features(games, tg, policy, features_cfg)
    cols = FEATURE_COLUMNS_FULL
    pd.testing.assert_frame_equal(
        base[cols].reset_index(drop=True), mutated[cols].reset_index(drop=True)
    )
    # mutating legitimately eligible history must change features
    tg2 = ds.team_games.copy()
    eligible = ~future
    tg2.loc[eligible, "off_epa_sum"] += 50.0
    changed = build_features(games, tg2, policy, features_cfg)
    assert not np.allclose(base["team_off_epa_per_play"], changed["team_off_epa_per_play"])


def test_labels_are_separate_from_features(synthetic_small, policy, features_cfg):
    ds = synthetic_small
    feats = _features(ds, policy, features_cfg, [2016])
    assert "points" not in feats.columns and "home_score" not in feats.columns
    labels = build_labels(ds.games, ds.results, policy)
    assert (
        labels["label_available_utc"]
        == labels.merge(ds.games, on="game_id")["kickoff_utc"] + pd.Timedelta(hours=48)
    ).all()


def test_fold_grouping_and_scaler_isolation(synthetic_small, policy, features_cfg):
    ds = synthetic_small
    feats = _features(ds, policy, features_cfg, [2014, 2015, 2016])
    labels = build_labels(ds.games, ds.results, policy)
    train = feats[feats["season"] < 2016]
    evaluate = feats[feats["season"] == 2016]
    assert_grouped_split(train, evaluate)
    with pytest.raises(ModelValidationError):
        assert_grouped_split(train, pd.concat([evaluate, train.head(2)]))
    with pytest.raises(ModelValidationError):
        assert_grouped_split(train.iloc[1:], evaluate)  # one perspective row split away
    model, used = fit_score_model(train, labels, FitSpec("ridge_score", "full", 10.0))
    pre = model.preprocessor
    train_means = used[pre.features].astype(float).mean().to_numpy()
    assert np.allclose(pre.means[: len(pre.features)], train_means)
    all_means = feats[pre.features].astype(float).mean().to_numpy()
    assert not np.allclose(pre.means[: len(pre.features)], all_means)
    assert training_seasons(2018, 2012) == [2012, 2013, 2014, 2015, 2016, 2017]
    assert residual_seasons(2018, 2016, 5) == [2016, 2017]
    assert residual_seasons(2024, 2016, 5) == [2019, 2020, 2021, 2022, 2023]


def test_perspective_transforms(synthetic_small, policy, features_cfg):
    ds = synthetic_small
    games = ds.games[(ds.games["season"] == 2016) & (ds.games["week"] == 5)]
    neutral = games[games["neutral_site"]]
    assert len(neutral) == 1
    feats = build_features(games, ds.team_games, policy, features_cfg)
    n = feats[feats["game_id"] == neutral["game_id"].iloc[0]]
    assert (n["venue_advantage"] == 0.0).all()
    g = games[~games["neutral_site"]].head(1)
    f = build_features(g, ds.team_games, policy, features_cfg).set_index("perspective")
    assert f.loc["home", "venue_advantage"] == 1.0 and f.loc["away", "venue_advantage"] == -1.0
    assert f.loc["home", "rest_difference"] == -f.loc["away", "rest_difference"]
    swapped = g.copy()
    swapped[["home_team", "away_team"]] = g[["away_team", "home_team"]].to_numpy()
    fs = build_features(swapped, ds.team_games, policy, features_cfg).set_index("perspective")
    assert fs.loc["home", "team_off_epa_per_play"] == f.loc["away", "team_off_epa_per_play"]
    assert fs.loc["home", "rest_difference"] == f.loc["away", "rest_difference"]


def test_null_policy_and_indicators():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, len(FEATURE_COLUMNS_FULL))), columns=FEATURE_COLUMNS_FULL)
    X.loc[:1, "team_points_for"] = np.nan
    pre = Preprocessor(FEATURE_COLUMNS_FULL).fit(X, 0.05)
    assert pre.indicator_features == ["team_points_for"]
    out = pre.transform(X)
    assert out.shape[1] == len(FEATURE_COLUMNS_FULL) + 1
    X.loc[:10, "team_points_for"] = np.nan  # 22% unexpected nulls
    with pytest.raises(ModelValidationError):
        Preprocessor(FEATURE_COLUMNS_FULL).fit(X, 0.05)
    X["team_points_for"] = np.nan
    with pytest.raises(ModelValidationError):
        Preprocessor(FEATURE_COLUMNS_FULL).fit(X, 1.0)


def test_bundle_round_trip_and_predict_game(synthetic_small, policy, features_cfg, tmp_path):
    ds = synthetic_small
    feats = _features(ds, policy, features_cfg, [2014, 2015, 2016, 2017])
    labels = build_labels(ds.games, ds.results, policy)
    spec = FitSpec("ridge_score", "full", 100.0)
    train = feats[feats["season"] < 2017]
    model, used = fit_score_model(train, labels, spec)
    res = np.column_stack(
        [np.random.default_rng(1).normal(0, 9, 600), np.random.default_rng(2).normal(0, 9, 600)]
    )
    params = fit_residual_params(
        res,
        seasons=[2015, 2016],
        game_ids_hash="h",
        min_games=500,
        diagonal_shrinkage=0.1,
        floor=1e-6,
    )
    bundle = build_bundle(
        model,
        spec,
        used,
        params,
        ["g"],
        forecast_season=2017,
        data_mode="historical_reconstruction",
        config_hash="c",
    )
    path = bundle.save(tmp_path / "bundle.json")
    loaded = ModelBundle.load(path)
    restored = loaded.score_model()
    rows = feats[feats["season"] == 2017].head(2)
    assert np.allclose(predict_location(model, rows), predict_location(restored, rows))
    dist = predict_game(loaded, rows, DistributionConfig())
    assert abs(dist.pmf.sum() - 1) < 1e-8
    bad = loaded.model_copy(update={"feature_version": "0"})
    with pytest.raises(ModelValidationError):
        predict_game(bad, rows, DistributionConfig())
    assert isinstance(restored, RidgeScoreModel)
    assert loaded.training_cutoff_utc is not None and loaded.residual_game_ids == ["g"]
