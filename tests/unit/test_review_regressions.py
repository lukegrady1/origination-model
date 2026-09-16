"""Regression tests for the September 2026 implementation review findings R1–R6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from nfl_origination.config import DistributionConfig, FeaturesConfig, load_config
from nfl_origination.errors import InvalidInputError, MissingDataError, ModelValidationError
from nfl_origination.experiment import (
    Dataset,
    chronological_fits,
    fit_residuals_for_season,
    policy_from_config,
    predict_slate,
    run_backtest,
)
from nfl_origination.features.aggregate import aggregate_team_games
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.builder import build_features, build_labels, require_forecastable
from nfl_origination.fixtures import load_mini_raw_fixture
from nfl_origination.market.asof import QuotePolicy
from nfl_origination.market.compare import compare_predictions, load_distribution
from nfl_origination.market.settle import _line_clv, paper_backtest
from nfl_origination.models.bundle import (
    ModelBundle,
    build_bundle,
    check_bundle_compatible,
    fit_score_model,
    predict_game,
)
from nfl_origination.models.distribution import fit_residual_params
from nfl_origination.models.ridge import FitSpec
from nfl_origination.pricing.markets import MarketSpec, settlement_probabilities
from nfl_origination.protocol import (
    FrozenProtocol,
    code_digest,
    freeze_protocol,
    load_protocol,
    protocol_path,
    verify_protocol,
)
from nfl_origination.schemas import FEATURE_COLUMNS_FULL
from tests.conftest import ts

ROOT = Path(__file__).resolve().parents[2]
LATE = ts("2030-01-01T00:00:00Z")
EARLY = ts("2009-01-01T00:00:00Z")


def _observed(ds, games_at: pd.Timestamp, team_games_at: pd.Timestamp):
    games = ds.games.copy()
    games["first_observed_utc"] = games_at
    tg = ds.team_games.copy()
    tg["first_observed_utc"] = team_games_at
    tg["schedule_first_observed_utc"] = games_at
    return games, tg


# ---------------------------------------------------------------------------------------- R1
def test_r1_late_observed_history_is_never_forecastable(synthetic_small, features_cfg):
    policy = AsOfPolicy(mode="recorded_asof")
    games, tg = _observed(synthetic_small, LATE, LATE)
    target = games[(games["season"] == 2012) & (games["week"] == 3)].head(1)
    feats = build_features(target, tg, policy, features_cfg)
    assert feats["insufficient_warmup"].all()  # no eligible prior rows at all
    assert feats["unobserved_inputs"].all()  # the game's own schedule row is unobserved
    with pytest.raises(MissingDataError):
        require_forecastable(feats)
    # mutating unavailable prior-season data must not move any feature value
    tg2 = tg.copy()
    tg2.loc[tg2["season"] < 2012, "points_for"] += 100
    feats2 = build_features(target, tg2, policy, features_cfg)
    pd.testing.assert_frame_equal(feats[FEATURE_COLUMNS_FULL], feats2[FEATURE_COLUMNS_FULL])


def test_r1_priors_use_only_observed_prior_rows(synthetic_small, features_cfg):
    policy = AsOfPolicy(mode="recorded_asof")
    games, tg = _observed(synthetic_small, EARLY, EARLY)
    target = games[(games["season"] == 2014) & (games["week"] == 3)].head(1)
    cutoff = policy.cutoff_for(pd.Timestamp(target["kickoff_utc"].iloc[0]))
    base = build_features(target, tg, policy, features_cfg)
    assert not base["insufficient_warmup"].any() and not base["unobserved_inputs"].any()
    # make one prior season observed only after the cutoff: it must drop out of the prior
    tg_late = tg.copy()
    tg_late.loc[tg_late["season"] == 2013, "first_observed_utc"] = cutoff + pd.Timedelta(days=1)
    late = build_features(target, tg_late, policy, features_cfg)
    assert late["prior_seasons_used"].iloc[0] == "2012"
    assert base["prior_seasons_used"].iloc[0] == "2012,2013"
    assert late["prior_games_hash"].iloc[0] != base["prior_games_hash"].iloc[0]
    assert (late["max_observed_utc"] <= late["cutoff_utc"]).all()
    # provenance includes prior rows: their observation time is reflected
    tg_obs = tg.copy()
    tg_obs.loc[tg_obs["season"] == 2013, "first_observed_utc"] = cutoff - pd.Timedelta(hours=1)
    obs = build_features(target, tg_obs, policy, features_cfg)
    assert obs["max_observed_utc"].iloc[0] == cutoff - pd.Timedelta(hours=1)


def test_r1_late_pbp_with_early_schedule_is_ineligible():
    games, plays, results = load_mini_raw_fixture()
    games = games.copy()
    games["first_observed_utc"] = EARLY
    plays = plays.copy()
    plays["first_observed_utc"] = LATE
    tg = aggregate_team_games(games, plays, results)
    assert (tg["schedule_first_observed_utc"] == EARLY).all()
    assert (tg["pbp_first_observed_utc"] == LATE).all()
    assert (tg["first_observed_utc"] == LATE).all()  # later of the two sources
    policy = AsOfPolicy(mode="recorded_asof")
    mask = policy.eligible_mask(
        tg["kickoff_utc"], ts("2024-06-01T00:00:00Z"), tg["first_observed_utc"]
    )
    assert not mask.any()


def test_r1_late_observed_labels_never_enter_fitting(synthetic_small, features_cfg):
    policy = AsOfPolicy(mode="recorded_asof")
    games, tg = _observed(synthetic_small, EARLY, EARLY)
    labels = build_labels(games, synthetic_small.results, policy)
    feats = build_features(
        games[games["season"].isin([2012, 2013, 2014])], tg, policy, features_cfg
    )
    ds = Dataset(
        games, synthetic_small.results, tg, feats, labels, pd.DataFrame(), {}, None, True, policy
    )
    fits = chronological_fits(ds, FitSpec("league_baseline", "full", None), [2014], 2012)
    assert fits[2014].n_dropped_unavailable_labels == 0
    late_games = games.copy()
    late_games["first_observed_utc"] = LATE
    late_labels = build_labels(late_games, synthetic_small.results, policy)
    assert (late_labels["label_available_utc"] == LATE).all()
    ds_late = Dataset(
        games,
        synthetic_small.results,
        tg,
        feats,
        late_labels,
        pd.DataFrame(),
        {},
        None,
        True,
        policy,
    )
    with pytest.raises(MissingDataError):
        chronological_fits(ds_late, FitSpec("league_baseline", "full", None), [2014], 2012)
    # partially late labels are dropped, counted, and excluded from the residual pool
    mixed = late_labels.copy()
    early_ids = set(games[games["season"] == 2012]["game_id"])
    mixed.loc[mixed["game_id"].isin(early_ids), "label_available_utc"] = EARLY
    ds_mixed = Dataset(
        games, synthetic_small.results, tg, feats, mixed, pd.DataFrame(), {}, None, True, policy
    )
    fits = chronological_fits(
        ds_mixed, FitSpec("league_baseline", "full", None), [2013, 2014], 2012
    )
    assert fits[2014].n_dropped_unavailable_labels > 0
    assert set(fits[2014].train["season"]) == {2012}
    with pytest.raises(ModelValidationError):  # 2013 residuals unavailable -> < 500 games
        fit_residuals_for_season(
            _cfg_with(residual_first_season=2013, min_residual_games=500), fits, 2014
        )


def _cfg_with(**model_overrides):
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["model"].update(model_overrides)
    raw["run"]["artifacts_dir"] = "/tmp/unused"
    tmp = Path("/tmp/_cfg_r1.yaml")
    tmp.write_text(yaml.safe_dump(raw))
    return load_config(tmp)


def test_r1_recorded_asof_backtest_fails_clearly_on_unobserved_data(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["run"] = {
        "kind": "development",
        "label": "r1",
        "artifacts_dir": str(tmp_path / "a"),
        "reports_dir": str(tmp_path / "r"),
    }
    raw["data"]["mode"] = "recorded_asof"
    raw["market"]["enabled"] = False
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    with pytest.raises(MissingDataError):
        run_backtest(cfg, offline=True)  # synthetic rows carry no observation times


# ---------------------------------------------------------------------------------------- R2
def _dists(normal_dist):
    b0 = normal_dist
    shifted = np.roll(normal_dist.margin_pmf, 6)  # a different distribution for M1
    shifted = shifted / shifted.sum()
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "model_id": "B0",
                "max_score": b0.max_score,
                "margin_pmf": b0.margin_pmf.tolist(),
                "total_pmf": b0.total_pmf.tolist(),
            },
            {
                "game_id": "g",
                "model_id": "M1",
                "max_score": b0.max_score,
                "margin_pmf": shifted.tolist(),
                "total_pmf": b0.total_pmf.tolist(),
            },
        ]
    )


def test_r2_distribution_lookup_is_keyed_by_model(normal_dist):
    dists = _dists(normal_dist)
    m1 = load_distribution(dists, "g", "M1")
    b0 = load_distribution(dists, "g", "B0")
    spec = MarketSpec.from_line("spread", "home", -3.5)
    assert settlement_probabilities(m1, spec).p_win != settlement_probabilities(b0, spec).p_win
    reordered = load_distribution(dists.iloc[::-1].reset_index(drop=True), "g", "M1")
    assert np.array_equal(reordered.margin_pmf, m1.margin_pmf)
    assert load_distribution(dists, "g", "nope") is None
    with pytest.raises(ModelValidationError):
        load_distribution(pd.concat([dists, dists]), "g", "M1")
    with pytest.raises(ModelValidationError):
        load_distribution(dists.drop(columns=["model_id"]), "g", "M1")


def test_r2_comparison_matches_selected_models_own_pricing(normal_dist):
    dists = _dists(normal_dist)
    m1 = load_distribution(dists, "g", "M1")
    pred = pd.DataFrame(
        [
            {
                "game_id": "g",
                "cutoff_utc": ts("2020-09-12T17:00:00Z"),
                "model_id": "M1",
                "mean_margin": m1.mean_margin,
                "mean_total": m1.mean_total,
                "fair_home_handicap": -3.5,
                "fair_total": 44.5,
                "p_home_win_given_no_tie": 0.6,
            }
        ]
    )
    odds = pd.DataFrame(
        {
            "quote_id": ["a", "b"],
            "provider_event_id": ["e", "e"],
            "game_id": ["g", "g"],
            "bookmaker": ["book", "book"],
            "market": ["spread", "spread"],
            "selection": ["home", "away"],
            "line": [-3.5, 3.5],
            "decimal_odds": [1.91, 1.91],
            "snapshot_at_utc": [ts("2020-09-12T16:50:00Z")] * 2,
            "bookmaker_updated_at_utc": [ts("2020-09-12T16:49:00Z")] * 2,
            "kickoff_at_snapshot_utc": [ts("2020-09-13T17:00:00Z")] * 2,
            "settlement_rule": ["two_way_tie_void"] * 2,
        }
    )
    table, summary = compare_predictions(pred, dists, odds, QuotePolicy("book"))
    direct = settlement_probabilities(
        m1, MarketSpec.from_line("spread", "home", -3.5)
    ).p_win_given_no_push
    assert table["model_spread_home_given_no_push"].iloc[0] == pytest.approx(direct)
    assert summary["n_with_market"] == 1


# ---------------------------------------------------------------------------------------- R3
def _proto_cfg(tmp_path: Path, label: str = "holdout"):
    raw = yaml.safe_load((ROOT / "configs" / "holdout.yaml").read_text())
    raw["run"]["artifacts_dir"] = str(tmp_path / "artifacts")
    raw["run"]["label"] = label
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_r3_code_digest_scope(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "models").mkdir(parents=True)
    (pkg / "dashboard").mkdir()
    (pkg / "evaluation").mkdir()
    (pkg / "models" / "ridge.py").write_text("x = 1\n")
    (pkg / "dashboard" / "app.py").write_text("ui = 1\n")
    (pkg / "evaluation" / "report.py").write_text("r = 1\n")
    (pkg / "cli.py").write_text("c = 1\n")
    base = code_digest(pkg)
    (pkg / "dashboard" / "app.py").write_text("ui = 2\n")
    (pkg / "evaluation" / "report.py").write_text("r = 2\n")
    (pkg / "cli.py").write_text("c = 2\n")
    assert code_digest(pkg) == base  # presentation code is out of scope
    (pkg / "models" / "ridge.py").write_text("x = 2\n")
    assert code_digest(pkg) != base  # a numerical implementation change is detected


def test_r3_protocol_verifies_code_and_lockfile(tmp_path, monkeypatch):
    cfg = _proto_cfg(tmp_path)
    frozen = freeze_protocol(
        cfg, {"pbp/2025": "abc"}, "ex", development_run_id=None, confirmation_run_id=None
    )
    assert frozen.code_digest and frozen.lock_digest
    verify_protocol(cfg, {"pbp/2025": "abc"}, "ex")
    import nfl_origination.protocol as proto

    monkeypatch.setattr(proto, "code_digest", lambda package_dir=None: "changed")
    with pytest.raises(InvalidInputError, match="code_digest"):
        verify_protocol(cfg, {"pbp/2025": "abc"}, "ex")
    monkeypatch.undo()
    monkeypatch.setattr(proto, "lock_digest", lambda lock_path=None: "changed-lock")
    with pytest.raises(InvalidInputError, match="lock_digest"):
        verify_protocol(cfg, {"pbp/2025": "abc"}, "ex")


def test_r3_legacy_protocol_without_code_digest_is_refused(tmp_path):
    cfg = _proto_cfg(tmp_path)
    frozen = freeze_protocol(
        cfg, {"pbp/2025": "abc"}, "ex", development_run_id=None, confirmation_run_id=None
    )
    legacy = frozen.model_dump()
    legacy["code_digest"] = None
    legacy["lock_digest"] = None
    path = protocol_path(cfg.run.artifacts_dir, cfg.run.label)
    path.write_text(FrozenProtocol.model_validate(legacy).model_dump_json())
    with pytest.raises(InvalidInputError, match="predates code/lockfile enforcement"):
        verify_protocol(cfg, {"pbp/2025": "abc"}, "ex")
    # a new labeled protocol lives in its own file and does not touch the legacy one
    cfg2 = _proto_cfg(tmp_path, label="v1_1_repair")
    freeze_protocol(
        cfg2, {"pbp/2025": "abc"}, "ex", development_run_id=None, confirmation_run_id=None
    )
    assert protocol_path(cfg2.run.artifacts_dir, "v1_1_repair").exists()
    assert load_protocol(cfg.run.artifacts_dir, "holdout").code_digest is None


# ---------------------------------------------------------------------------------------- R4
def _bundle(synthetic_small, policy, features_cfg):
    games = synthetic_small.games[synthetic_small.games["season"].isin([2014, 2015, 2016, 2017])]
    feats = build_features(games, synthetic_small.team_games, policy, features_cfg)
    labels = build_labels(synthetic_small.games, synthetic_small.results, policy)
    spec = FitSpec("ridge_score", "full", 100.0)
    model, used = fit_score_model(feats[feats["season"] < 2017], labels, spec)
    rng = np.random.default_rng(0)
    params = fit_residual_params(
        rng.normal(0, 9, (600, 2)),
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
        policy=policy,
        features_cfg=features_cfg,
        config_hash="c",
    )
    return bundle, feats[feats["season"] == 2017].head(2)


def test_r4_bundle_contract_rejects_changed_feature_definitions(
    synthetic_small, policy, features_cfg
):
    bundle, rows = _bundle(synthetic_small, policy, features_cfg)
    check_bundle_compatible(bundle, policy, features_cfg)
    predict_game(bundle, rows, DistributionConfig(), policy=policy, features_cfg=features_cfg)
    with pytest.raises(ModelValidationError, match="history_games"):
        check_bundle_compatible(bundle, policy, FeaturesConfig(history_games=8))
    with pytest.raises(ModelValidationError, match="shrinkage_plays"):
        check_bundle_compatible(bundle, policy, FeaturesConfig(shrinkage_plays=50))
    with pytest.raises(ModelValidationError, match="data_mode"):
        check_bundle_compatible(bundle, AsOfPolicy(mode="recorded_asof"), features_cfg)
    legacy = bundle.model_copy(update={"feature_contract": None, "contract_hash": None})
    with pytest.raises(ModelValidationError, match="no feature contract"):
        check_bundle_compatible(legacy, policy, features_cfg)


def test_r4_feature_rows_from_another_data_mode_are_rejected(synthetic_small, policy, features_cfg):
    bundle, rows = _bundle(synthetic_small, policy, features_cfg)
    relabeled = rows.copy()
    relabeled["data_mode"] = "recorded_asof"
    with pytest.raises(ModelValidationError, match="provenance"):
        predict_game(bundle, relabeled, DistributionConfig())
    unobserved = rows.copy()
    unobserved["unobserved_inputs"] = True
    with pytest.raises(MissingDataError):
        predict_game(bundle, unobserved, DistributionConfig())


def test_r4_output_directory_changes_stay_compatible(
    synthetic_small, policy, features_cfg, tmp_path
):
    bundle, _rows = _bundle(synthetic_small, policy, features_cfg)
    path = bundle.save(tmp_path / "elsewhere" / "b.json")
    check_bundle_compatible(ModelBundle.load(path), policy, features_cfg)


# ---------------------------------------------------------------------------------------- R5
def _slate_cfg(tmp_path: Path):
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["run"] = {
        "kind": "forecast",
        "label": "r5",
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
    }
    raw["data"]["seasons"] = [2010, 2018]
    raw["model"]["selected_alpha"] = 100
    raw["model"]["ablation_selected_alpha"] = None
    raw["evaluation"]["ablation"] = False
    raw["market"]["enabled"] = False
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


@pytest.fixture
def slate_setup(tmp_path, synthetic_small, features_cfg):
    cfg = _slate_cfg(tmp_path)
    policy = policy_from_config(cfg)
    ds = synthetic_small
    bundle, _ = _bundle(ds, policy, features_cfg)
    bundle = bundle.model_copy(update={"forecast_season": 2018})
    bundle_dir = cfg.run.artifacts_dir / "models" / "forecast_2018"
    bundle.save(bundle_dir / f"{bundle.model_id}.json")

    def loader(config, seasons, offline):
        return ds.games, ds.team_games, {"synthetic": "x"}

    return cfg, ds, loader


def test_r5_live_forecast_uses_generation_time_and_excludes_started_games(slate_setup, monkeypatch):
    cfg, ds, loader = slate_setup
    week5 = ds.games[(ds.games["season"] == 2018) & (ds.games["week"] == 5)]
    fake_now = pd.Timestamp(week5["kickoff_utc"].min()) + pd.Timedelta(
        hours=1
    )  # first game started
    import nfl_origination.experiment as exp

    monkeypatch.setattr(exp, "utc_now", lambda: fake_now.to_pydatetime())
    manifest, slate = predict_slate(cfg, 2018, 5, as_of=None, offline=True, loader=loader)
    assert len(slate) == len(week5) - 1
    assert (slate["cutoff_utc"] == fake_now).all()
    assert (slate["forecast_policy"] == "custom_horizon").all()
    assert manifest.forecast_policy == "custom_horizon"
    assert (slate["model_created_utc"].notna()).all()
    assert not slate["data_quality_flags"].str.contains("cutoff_in_future").any()


def test_r5_standard_horizon_only_as_reconstruction(slate_setup, monkeypatch):
    cfg, ds, loader = slate_setup
    week5 = ds.games[(ds.games["season"] == 2018) & (ds.games["week"] == 5)]
    import nfl_origination.experiment as exp

    early = pd.Timestamp(week5["kickoff_utc"].min()) - pd.Timedelta(days=3)
    monkeypatch.setattr(exp, "utc_now", lambda: early.to_pydatetime())
    with pytest.raises(InvalidInputError, match="standard cutoff"):
        predict_slate(
            cfg, 2018, 5, as_of=None, offline=True, reconstruct_standard_horizon=True, loader=loader
        )
    late = pd.Timestamp(week5["kickoff_utc"].max()) + pd.Timedelta(days=2)
    monkeypatch.setattr(exp, "utc_now", lambda: late.to_pydatetime())
    manifest, slate = predict_slate(
        cfg, 2018, 5, as_of=None, offline=True, reconstruct_standard_horizon=True, loader=loader
    )
    assert len(slate) == len(week5)  # completed games are allowed in an explicit reconstruction
    assert (slate["forecast_policy"] == "kickoff_minus_24h").all()
    assert any("historical reconstruction" in n for n in manifest.notes)
    with pytest.raises(InvalidInputError, match="future"):
        predict_slate(cfg, 2018, 5, as_of=late + pd.Timedelta(days=1), offline=True, loader=loader)


def test_r5_quotes_before_model_availability_are_excluded(normal_dist):
    dists = _dists(normal_dist)
    pred = pd.DataFrame(
        [
            {
                "game_id": "g",
                "cutoff_utc": ts("2020-09-12T17:00:00Z"),
                "model_id": "M1",
                "mean_margin": 1.0,
                "mean_total": 44.0,
                "fair_home_handicap": -1.0,
                "fair_total": 44.0,
                "p_home_win_given_no_tie": 0.55,
                "model_created_utc": ts("2020-09-12T18:00:00Z"),
            }
        ]
    )
    odds = pd.DataFrame(
        {
            "quote_id": ["a", "b"],
            "provider_event_id": ["e", "e"],
            "game_id": ["g", "g"],
            "bookmaker": ["book", "book"],
            "market": ["spread", "spread"],
            "selection": ["home", "away"],
            "line": [-1.0, 1.0],
            "decimal_odds": [1.91, 1.91],
            "snapshot_at_utc": [ts("2020-09-12T16:50:00Z")] * 2,
            "bookmaker_updated_at_utc": [ts("2020-09-12T16:49:00Z")] * 2,
            "kickoff_at_snapshot_utc": [ts("2020-09-13T17:00:00Z")] * 2,
            "settlement_rule": ["two_way_tie_void"] * 2,
        }
    )
    table, summary = compare_predictions(pred, dists, odds, QuotePolicy("book"))
    assert summary["n_with_market"] == 0 and summary["n_decisions_before_model_availability"] == 1
    assert table["exclusion"].iloc[0] == "decision_time_before_model_availability"


# ---------------------------------------------------------------------------------------- R6
@pytest.mark.parametrize(
    "selection,bet,close,expected",
    [
        ("home", -3.5, -3.0, -0.5),  # home favorite: line moved against the bet
        ("home", -3.0, -3.5, 0.5),  # home favorite: line moved in favour
        ("away", 3.5, 3.0, 0.5),  # away underdog: closed shorter -> favorable
        ("away", 3.0, 3.5, -0.5),  # away underdog: closed longer -> unfavorable
        ("home", 2.5, 3.0, -0.5),  # home underdog
        ("away", -2.5, -3.0, 0.5),  # away favorite
    ],
)
def test_r6_spread_line_clv_is_selected_side_difference(selection, bet, close, expected):
    spec = MarketSpec.from_line("spread", selection, bet)
    assert _line_clv(spec, bet, close) == pytest.approx(expected)
    assert _line_clv(MarketSpec.from_line("total", "over", 44.5), 44.5, 45.0) == pytest.approx(0.5)
    assert _line_clv(MarketSpec.from_line("total", "under", 44.5), 44.5, 45.0) == pytest.approx(
        -0.5
    )


def test_r6_paper_backtest_reports_away_clv_correctly(normal_dist):
    """Full paper-backtest path: an away +3.5 bet closing at +3 must show +0.5 line CLV."""
    from nfl_origination.config import MarketConfig

    shifted = np.roll(
        normal_dist.margin_pmf, -12
    )  # away-leaning distribution -> away bet has value
    shifted = shifted / shifted.sum()
    dists = pd.DataFrame(
        [
            {
                "game_id": "g",
                "model_id": "M1",
                "max_score": normal_dist.max_score,
                "margin_pmf": shifted.tolist(),
                "total_pmf": normal_dist.total_pmf.tolist(),
            }
        ]
    )
    kickoff = ts("2020-09-13T17:00:00Z")
    cutoff = kickoff - pd.Timedelta(hours=24)
    pred = pd.DataFrame(
        [
            {
                "game_id": "g",
                "cutoff_utc": cutoff,
                "kickoff_utc": kickoff,
                "model_id": "M1",
                "season": 2020,
                "week": 1,
            }
        ]
    )
    base = {
        "provider_event_id": "e",
        "game_id": "g",
        "bookmaker": "book",
        "settlement_rule": "two_way_tie_void",
        "kickoff_at_snapshot_utc": kickoff,
    }
    rows = [
        base
        | {
            "quote_id": "q1",
            "market": "spread",
            "selection": "home",
            "line": -3.5,
            "decimal_odds": 1.91,
            "snapshot_at_utc": cutoff - pd.Timedelta(minutes=10),
            "bookmaker_updated_at_utc": cutoff - pd.Timedelta(minutes=11),
        },
        base
        | {
            "quote_id": "q2",
            "market": "spread",
            "selection": "away",
            "line": 3.5,
            "decimal_odds": 1.91,
            "snapshot_at_utc": cutoff - pd.Timedelta(minutes=10),
            "bookmaker_updated_at_utc": cutoff - pd.Timedelta(minutes=11),
        },
        base
        | {
            "quote_id": "q3",
            "market": "spread",
            "selection": "home",
            "line": -3.0,
            "decimal_odds": 1.91,
            "snapshot_at_utc": kickoff - pd.Timedelta(minutes=10),
            "bookmaker_updated_at_utc": kickoff - pd.Timedelta(minutes=11),
        },
        base
        | {
            "quote_id": "q4",
            "market": "spread",
            "selection": "away",
            "line": 3.0,
            "decimal_odds": 1.91,
            "snapshot_at_utc": kickoff - pd.Timedelta(minutes=10),
            "bookmaker_updated_at_utc": kickoff - pd.Timedelta(minutes=11),
        },
    ]
    odds = pd.DataFrame(rows)
    results = pd.DataFrame({"game_id": ["g"], "home_score": [20], "away_score": [24]})
    cfg = MarketConfig(enabled=True, bookmaker="book", min_ev=0.0)
    bets, summary = paper_backtest(
        "run", pred, dists, odds, results, cfg, seed=1, bootstrap_replicates=10
    )
    assert len(bets) == 1 and bets["selection"].iloc[0] == "away"
    assert bets["line_clv"].iloc[0] == pytest.approx(0.5)
    assert bets["settlement"].iloc[0] == "win"
    assert summary["line_clv_mean"] == pytest.approx(0.5)
