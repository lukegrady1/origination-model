"""Experiment orchestration: dataset preparation, chronological backtests, fits, forecasts.

This module wires the layers together under the frozen protocol (spec sections 9, 11, 15).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ExperimentConfig
from nfl_origination.data.download import SourceManifest, ingest
from nfl_origination.data.normalize import normalize_sources
from nfl_origination.data.storage import write_parquet
from nfl_origination.data.validate import ValidationReport, validate_normalized
from nfl_origination.errors import InvalidInputError, MissingDataError, ModelValidationError
from nfl_origination.evaluation.bootstrap import block_bootstrap_differences
from nfl_origination.evaluation.metrics import (
    aggregate,
    per_game_scores,
    reliability_table,
    slice_metrics,
)
from nfl_origination.evaluation.splits import (
    assert_grouped_split,
    assert_labels_available,
    residual_seasons,
    training_seasons,
)
from nfl_origination.features.aggregate import aggregate_team_games
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.builder import (
    build_features,
    build_labels,
    features_hash,
    require_availability_fields,
    require_forecastable,
    usable_rows,
)
from nfl_origination.models.bundle import (
    ModelBundle,
    ScoreModel,
    align_labels,
    build_bundle,
    check_bundle_compatible,
    fit_score_model,
    game_ids_hash,
    game_locations,
    predict_location,
)
from nfl_origination.models.distribution import (
    ResidualParams,
    ScoreDistribution,
    fit_residual_params,
    predict_distribution,
)
from nfl_origination.models.ridge import FitSpec
from nfl_origination.pricing.fair_lines import fair_home_handicap, fair_moneyline, fair_total
from nfl_origination.protocol import record_holdout_run, verify_protocol
from nfl_origination.provenance import (
    RunManifest,
    RunRegistry,
    dependency_versions,
    environment_note,
    git_info,
    hash_frame,
    hash_json,
    iso_utc,
    new_run_id,
    sha256_file,
    utc_now,
    write_json,
)
from nfl_origination.schemas import (
    FEATURE_VERSION,
    FEATURES_SCHEMA,
    PREDICTIONS_SCHEMA,
    SCHEMA_VERSION,
    assert_no_market_columns,
    validate_frame,
)
from nfl_origination.synthetic import SYNTHETIC_LABEL, generate_synthetic_dataset

KEY_MARGINS = (0, 3, -3, 7, -7)


@dataclass
class Dataset:
    games: pd.DataFrame
    results: pd.DataFrame
    team_games: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    exclusions: pd.DataFrame
    source_hashes: dict[str, str]
    coverage: ValidationReport | None
    synthetic: bool
    policy: AsOfPolicy
    observed_at: dict[str, str] = field(default_factory=dict)

    @property
    def exclusions_hash(self) -> str:
        return hash_frame(self.exclusions) if len(self.exclusions) else hash_frame(pd.DataFrame())


def policy_from_config(config: ExperimentConfig) -> AsOfPolicy:
    return AsOfPolicy(
        mode=config.data.mode,
        completed_game_lag_hours=config.data.completed_game_lag_hours,
        cutoff_hours_before_kickoff=config.forecast.cutoff_hours_before_kickoff,
    )


def _feature_cache_key(config: ExperimentConfig, source_hashes: dict[str, str]) -> str:
    payload = {
        "sources": source_hashes,
        "policy": policy_from_config(config).__dict__,
        "features": config.features.model_dump(),
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seasons": config.data.seasons,
    }
    return hash_json(payload)[:16]


def prepare_dataset(
    config: ExperimentConfig,
    *,
    offline: bool,
    seasons: list[int] | None = None,
    rebuild: bool = False,
    manifest: SourceManifest | None = None,
) -> Dataset:
    """Load or build everything the models need. Real data comes from the cache only."""
    policy = policy_from_config(config)
    seasons = seasons or config.data.season_list
    if config.data.source == "synthetic":
        syn = generate_synthetic_dataset(
            seed=config.seed, seasons=config.data.seasons, lag_hours=policy.completed_game_lag_hours
        )
        games, results, team_games = syn.games, syn.results, syn.team_games
        feats = build_features(games, team_games, policy, config.features)
        labels = build_labels(games, results, policy)
        return Dataset(
            games,
            results,
            team_games,
            feats,
            labels,
            pd.DataFrame(columns=["game_id", "season", "week", "reason", "detail"]),
            {"synthetic": hash_frame(games)},
            None,
            True,
            policy,
        )
    manifest = manifest or ingest(seasons, config.data.cache_dir, offline=offline, refresh=False)
    data = normalize_sources(
        manifest,
        seasons,
        game_type=config.data.game_type,
        completed_game_lag_hours=policy.completed_game_lag_hours,
    )
    coverage = validate_normalized(data, seasons)
    team_games = aggregate_team_games(data.games, data.plays, data.results)
    labels = build_labels(data.games, data.results, policy)
    exclusions = coverage.exclusions_frame()
    norm_dir = config.data.normalized_dir
    write_parquet(data.games, norm_dir / "games.parquet")
    write_parquet(data.results, norm_dir / "results.parquet")
    write_parquet(team_games, norm_dir / "team_games.parquet")
    write_parquet(data.untimed_reference, norm_dir / "untimed_reference.parquet")
    write_parquet(labels, norm_dir / "labels.parquet")
    exclusions.to_csv(norm_dir / "exclusions.csv", index=False)
    write_json(norm_dir / "coverage_report.json", coverage.model_dump())
    key = _feature_cache_key(config, manifest.file_hashes())
    feat_path = config.data.features_dir / f"features_{key}.parquet"
    completed = data.games[data.games["status"] == "final"]
    cached = (
        _load_valid_feature_cache(feat_path, policy) if feat_path.exists() and not rebuild else None
    )
    if cached is not None:
        feats = cached
    else:
        feats = build_features(completed, team_games, policy, config.features)
        write_parquet(feats, feat_path)
        write_json(
            config.data.features_dir / f"features_{key}.json",
            {"sources": manifest.file_hashes(), "policy": policy.__dict__, "rows": len(feats)},
        )
    return Dataset(
        data.games,
        data.results,
        team_games,
        feats,
        labels,
        exclusions,
        manifest.file_hashes(),
        coverage,
        False,
        policy,
        data.observed_at,
    )


def _load_valid_feature_cache(path: Path, policy: AsOfPolicy) -> pd.DataFrame | None:
    """Accept a cached feature file only if it satisfies the current contract; else rebuild.

    A cache is never trusted on file name alone: it must validate against the current features
    schema, carry the current feature version, and match the data mode. Invalid caches are
    renamed aside (never silently reused) and the features are rebuilt.
    """
    try:
        cached = pd.read_parquet(path)
        validate_frame(cached, FEATURES_SCHEMA)
        require_availability_fields(cached, "feature cache")
        modes = set(cached["data_mode"].astype(str)) if len(cached) else {policy.mode}
        if modes != {policy.mode}:
            raise ModelValidationError(f"cached data mode {sorted(modes)} != {policy.mode}")
    except (ModelValidationError, KeyError, ValueError, OSError) as exc:
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        stale = path.with_name(f"{path.stem}.stale-{stamp}.parquet")
        path.replace(stale)
        write_json(
            path.with_suffix(".invalidated.json"),
            {"reason": str(exc), "moved_to": str(stale), "feature_version": FEATURE_VERSION},
        )
        return None
    return cached


# ----------------------------------------------------------------------------------------------
# Chronological fitting
# ----------------------------------------------------------------------------------------------


@dataclass
class SeasonFit:
    season: int
    model: ScoreModel
    train: pd.DataFrame
    locations: pd.DataFrame  # game_id, mu_home_score, mu_away_score, actual_home, actual_away
    fit_time_utc: pd.Timestamp
    n_dropped_unavailable_labels: int = 0


def _labeled_rows(dataset: Dataset) -> pd.DataFrame:
    labeled = align_labels(dataset.features, dataset.labels)
    return labeled[usable_rows(labeled)].reset_index(drop=True)


def _actuals(dataset: Dataset) -> pd.DataFrame:
    actual = dataset.results[["game_id", "home_score", "away_score"]].rename(
        columns={"home_score": "actual_home", "away_score": "actual_away"}
    )
    avail = dataset.labels[dataset.labels["perspective"] == "home"][
        ["game_id", "label_available_utc"]
    ]
    return actual.merge(avail, on="game_id", how="left")


def _available_training_rows(
    train: pd.DataFrame, fit_time: pd.Timestamp
) -> tuple[pd.DataFrame, int]:
    """Keep rows whose labels were available at the fit time; report how many were dropped."""
    ok = train["label_available_utc"] <= fit_time
    dropped = int((~ok).sum())
    kept = train[ok]
    # never split a game's two perspective rows
    counts = kept.groupby("game_id")["perspective"].nunique()
    whole = counts[counts == 2].index
    kept = kept[kept["game_id"].isin(whole)]
    return kept, dropped + int(len(train) - len(kept) - dropped)


def chronological_fits(
    dataset: Dataset,
    spec: FitSpec,
    seasons: list[int],
    train_start: int,
    *,
    fit_time_override: pd.Timestamp | None = None,
) -> dict[int, SeasonFit]:
    """Fit once per season on all prior seasons from ``train_start``; predict that season.

    The fit time is the season's first cutoff (or an explicit override); only labels available
    by then may enter fitting, in every data mode.
    """
    labeled = _labeled_rows(dataset)
    actuals = _actuals(dataset)
    out: dict[int, SeasonFit] = {}
    for season in seasons:
        train = labeled[labeled["season"].isin(training_seasons(season, train_start))]
        evaluate = labeled[labeled["season"] == season]
        if evaluate.empty:
            continue
        fit_time = (
            fit_time_override
            if fit_time_override is not None
            else pd.Timestamp(evaluate["cutoff_utc"].min())
        )
        train, dropped = _available_training_rows(train, fit_time)
        if train.empty:
            raise MissingDataError(
                f"season {season}: no training labels available at fit time {fit_time} "
                f"({dropped} rows dropped as unavailable under {dataset.policy.mode})"
            )
        assert_grouped_split(train, evaluate)
        assert_labels_available(train, fit_time)
        model, used = fit_score_model(train, dataset.labels, spec)
        mu = predict_location(model, evaluate)
        loc = game_locations(evaluate, mu).merge(actuals, on="game_id", how="left")
        out[season] = SeasonFit(season, model, used, loc, fit_time, dropped)
    return out


def residual_pool(
    fits: dict[int, SeasonFit],
    seasons: list[int],
    *,
    available_by: pd.Timestamp | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Residual pairs from prior out-of-fold seasons with labels available by ``available_by``."""
    frames = [fits[s].locations for s in seasons if s in fits]
    if not frames:
        return np.zeros((0, 2)), []
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["actual_home"].notna() & pool["actual_away"].notna()]
    if available_by is not None and "label_available_utc" in pool.columns:
        pool = pool[pool["label_available_utc"] <= available_by]
    res = np.column_stack(
        [
            pool["actual_home"].to_numpy(float) - pool["mu_home_score"].to_numpy(float),
            pool["actual_away"].to_numpy(float) - pool["mu_away_score"].to_numpy(float),
        ]
    )
    return res, pool["game_id"].astype(str).tolist()


def fit_residuals_for_season(
    config: ExperimentConfig, fits: dict[int, SeasonFit], season: int
) -> tuple[ResidualParams, list[str]]:
    seasons = residual_seasons(
        season, config.model.residual_first_season, config.model.residual_window_seasons
    )
    residuals, ids = residual_pool(fits, seasons, available_by=fits[season].fit_time_utc)
    params = fit_residual_params(
        residuals,
        seasons=[s for s in seasons if s in fits],
        game_ids_hash=game_ids_hash(ids),
        min_games=config.model.min_residual_games,
        diagonal_shrinkage=config.model.covariance_diagonal_shrinkage,
        floor=config.model.covariance_floor,
    )
    return params, ids


# ----------------------------------------------------------------------------------------------
# Pricing a game into a prediction row
# ----------------------------------------------------------------------------------------------


def prediction_row(
    dist: ScoreDistribution,
    *,
    run_id: str,
    game: pd.Series,
    cutoff: pd.Timestamp,
    model_id: str,
    data_mode: str,
    forecast_policy: str,
    config_hash: str,
    features_hash_value: str,
    quality_flags: list[str],
    actual: tuple[int, int] | None = None,
) -> dict[str, Any]:
    ml = fair_moneyline(dist)
    hc = fair_home_handicap(dist)
    tt = fair_total(dist)
    warnings = list(dist.warnings) + list(ml.flags)
    row: dict[str, Any] = {
        "run_id": run_id,
        "game_id": str(game["game_id"]),
        "cutoff_utc": cutoff,
        "season": int(game["season"]),
        "week": int(game["week"]),
        "home_team": str(game["home_team"]),
        "away_team": str(game["away_team"]),
        "kickoff_utc": pd.Timestamp(game["kickoff_utc"]),
        "model_id": model_id,
        "data_mode": data_mode,
        "forecast_policy": forecast_policy,
        "mu_home_score": dist.mu_home,
        "mu_away_score": dist.mu_away,
        "dist_mean_home_score": dist.mean_home,
        "dist_mean_away_score": dist.mean_away,
        "mean_margin": dist.mean_margin,
        "mean_total": dist.mean_total,
        "p_home_win": ml.p_home,
        "p_tie": ml.p_tie,
        "p_away_win": ml.p_away,
        "p_home_win_given_no_tie": np.nan if ml.q_home is None else ml.q_home,
        "fair_decimal_home": np.nan if ml.decimal_home is None else ml.decimal_home,
        "fair_decimal_away": np.nan if ml.decimal_away is None else ml.decimal_away,
        "fair_american_home": np.nan if ml.american_home is None else ml.american_home,
        "fair_american_away": np.nan if ml.american_away is None else ml.american_away,
        "mean_home_handicap": -dist.mean_margin,
        "fair_home_handicap": hc.line,
        "fair_home_handicap_p_win": hc.probs.p_win,
        "fair_home_handicap_p_push": hc.probs.p_push,
        "fair_home_handicap_p_loss": hc.probs.p_loss,
        "fair_total": tt.line,
        "fair_total_p_over": tt.probs.p_win,
        "fair_total_p_push": tt.probs.p_push,
        "fair_total_p_under": tt.probs.p_loss,
        "negative_latent_mass": dist.negative_latent_mass,
        "omitted_upper_mass": dist.omitted_upper_mass,
        "max_score_support": dist.max_score,
        "clipped_negative_cells": dist.clipped_negative_cells,
        "warnings": ";".join(warnings),
        "data_quality_flags": ";".join(quality_flags),
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "config_hash": config_hash,
        "features_hash": features_hash_value,
        "neutral_site": bool(game["neutral_site"]),
    }
    for level, tag in ((0.5, 50), (0.8, 80), (0.95, 95)):
        lo, hi = dist.margin_interval(level)
        row[f"margin_q{tag}_lo"], row[f"margin_q{tag}_hi"] = lo, hi
        lo, hi = dist.total_interval(level)
        row[f"total_q{tag}_lo"], row[f"total_q{tag}_hi"] = lo, hi
    for m in KEY_MARGINS:
        row[f"p_margin_{m}"] = dist.margin_probability(m)
    row["p_total_even"] = float(dist.total_pmf[0::2].sum())
    if actual is not None:
        row["actual_home"], row["actual_away"] = int(actual[0]), int(actual[1])
        row["crps_margin"] = dist.crps_margin(actual[0] - actual[1])
        row["crps_total"] = dist.crps_total(actual[0] + actual[1])
    return row


def _finalize_predictions(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(PREDICTIONS_SCHEMA.required))
    df = pd.DataFrame(rows)
    for c in ("cutoff_utc", "kickoff_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in [k for k, v in PREDICTIONS_SCHEMA.required.items() if v == "int"]:
        df[c] = df[c].astype(np.int64)
    for c in [k for k, v in PREDICTIONS_SCHEMA.required.items() if v == "str"]:
        df[c] = df[c].astype(str)
    validate_frame(df, PREDICTIONS_SCHEMA)
    return df


def _quality_flags(feature_rows: pd.DataFrame) -> list[str]:
    flags = []
    if feature_rows["team_cold_start"].any():
        flags.append("cold_start")
    if feature_rows["team_rest_missing"].any():
        flags.append("rest_missing")
    if "games_without_pbp" in feature_rows and feature_rows["games_without_pbp"].sum() > 0:
        flags.append("history_without_pbp")
    if (feature_rows["team_history_games"] < 16).any():
        flags.append("short_history")
    return flags


# ----------------------------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------------------------


def model_variants(config: ExperimentConfig, *, candidates: bool) -> list[FitSpec]:
    seed = config.seed
    specs = [FitSpec("league_baseline", "full", None, seed)]
    if candidates:
        for a in config.model.alpha_candidates:
            specs.append(FitSpec("ridge_score", "full", float(a), seed))
        if config.evaluation.ablation:
            for a in config.model.alpha_candidates:
                specs.append(FitSpec("ridge_score", "epa_free", float(a), seed))
    else:
        if config.model.selected_alpha is None:
            raise InvalidInputError("model.selected_alpha must be resolved for this run kind")
        specs.append(FitSpec("ridge_score", "full", float(config.model.selected_alpha), seed))
        if config.evaluation.ablation and config.model.ablation_selected_alpha is not None:
            specs.append(
                FitSpec(
                    "ridge_score", "epa_free", float(config.model.ablation_selected_alpha), seed
                )
            )
    return specs


def select_alpha(
    scored_by_model: dict[str, pd.DataFrame],
    specs: list[FitSpec],
    feature_set: str,
    dev_seasons: list[int],
    tie_tolerance: float,
) -> dict[str, Any]:
    """Pick alpha by pooled game-level score MAE over development seasons; ties -> larger alpha."""
    table: list[dict[str, Any]] = []
    for spec in specs:
        if spec.family != "ridge_score" or spec.feature_set != feature_set:
            continue
        scored = scored_by_model.get(spec.model_id)
        if scored is None:
            continue
        dev = scored[scored["season"].isin(dev_seasons)]
        per_season = {int(s): float(p["score_mae"].mean()) for s, p in dev.groupby("season")}
        table.append(
            {
                "model_id": spec.model_id,
                "alpha": spec.alpha,
                "pooled_score_mae": float(dev["score_mae"].mean()),
                "n_games": len(dev),
                "per_season_score_mae": per_season,
            }
        )
    if not table:
        raise ModelValidationError(f"no candidates scored for feature set {feature_set}")
    best_mae = min(float(t["pooled_score_mae"]) for t in table)
    within = [t for t in table if float(t["pooled_score_mae"]) <= best_mae + tie_tolerance]
    chosen = max(within, key=lambda t: float(t["alpha"]))
    return {
        "feature_set": feature_set,
        "rule": "min pooled game-level score MAE over development seasons; "
        f"ties within {tie_tolerance} points resolved toward the larger alpha",
        "development_seasons": dev_seasons,
        "candidates": table,
        "selected_alpha": chosen["alpha"],
        "selected_model_id": chosen["model_id"],
    }


def _score_season(
    config: ExperimentConfig,
    dataset: Dataset,
    fits: dict[int, SeasonFit],
    spec: FitSpec,
    season: int,
    run_id: str,
    run_dir: Path,
    feats_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Price every completed game of ``season`` with the frozen season model + residuals."""
    if season not in fits:
        return [], [], {"season": season, "status": "no_games"}
    try:
        residual, ids = fit_residuals_for_season(config, fits, season)
    except ModelValidationError as exc:
        return [], [], {"season": season, "status": f"insufficient_residual_history: {exc}"}
    fit = fits[season]
    bundle = build_bundle(
        fit.model,
        spec,
        fit.train,
        residual,
        ids,
        forecast_season=season,
        policy=dataset.policy,
        features_cfg=config.features,
        config_hash=config.config_hash(),
        notes=[f"fit_time_utc={iso_utc(fit.fit_time_utc.to_pydatetime())}"],
    )
    bundle.save(run_dir / "bundles" / f"{spec.model_id}_{season}.json")
    games = dataset.games.set_index("game_id")
    feats = dataset.features[dataset.features["season"] == season]
    rows, dists = [], []
    policy = dataset.policy
    for loc in fit.locations.itertuples(index=False):
        game = games.loc[loc.game_id]
        game = game.copy()
        game["game_id"] = loc.game_id
        cutoff = pd.Timestamp(policy.cutoff_for(pd.Timestamp(game["kickoff_utc"])))
        dist = predict_distribution(
            np.array([loc.mu_home_score, loc.mu_away_score]), residual, config.distribution
        )
        frows = feats[feats["game_id"] == loc.game_id]
        rows.append(
            prediction_row(
                dist,
                run_id=run_id,
                game=game,
                cutoff=cutoff,
                model_id=spec.model_id,
                data_mode=config.data.mode,
                forecast_policy=policy.forecast_policy_label,
                config_hash=config.config_hash(),
                features_hash_value=feats_hash,
                quality_flags=_quality_flags(frows),
                actual=(int(loc.actual_home), int(loc.actual_away)),
            )
        )
        dists.append(
            {
                "game_id": loc.game_id,
                "model_id": spec.model_id,
                "max_score": dist.max_score,
                "margin_pmf": dist.margin_pmf.tolist(),
                "total_pmf": dist.total_pmf.tolist(),
            }
        )
    info = {
        "season": season,
        "status": "scored",
        "n_games": len(rows),
        "residual": residual.to_dict(),
        "training_seasons": bundle.training_seasons,
        "training_games": bundle.training_games,
        "training_cutoff_utc": bundle.training_cutoff_utc,
    }
    return rows, dists, info


def _model_report(scored: pd.DataFrame, sparse: int) -> dict[str, Any]:
    if scored.empty:
        return {"pooled": {"n_games": 0}}
    rep: dict[str, Any] = {"pooled": aggregate(scored), "slices": slice_metrics(scored)}
    non_tie = scored[scored["is_tie"] == 0]
    rep["reliability_home_win_3way"] = reliability_table(
        scored["p_home_win"].to_numpy(), scored["home_win"].to_numpy(), sparse_threshold=sparse
    )
    rep["reliability_home_win_given_no_tie"] = reliability_table(
        non_tie["p_home_win_given_no_tie"].to_numpy(),
        non_tie["home_win"].to_numpy(),
        sparse_threshold=sparse,
    )
    rep["key_numbers"] = {
        **{
            f"margin_{m}": {
                "predicted": float(scored[f"p_margin_{m}"].mean()),
                "observed": float(scored[f"obs_margin_{m}"].mean()),
            }
            for m in KEY_MARGINS
        },
        "total_even": {
            "predicted": float(scored["p_total_even"].mean()),
            "observed": float(scored["obs_total_even"].mean()),
        },
        "n_games": len(scored),
    }
    return rep


def run_backtest(
    config: ExperimentConfig, *, offline: bool = False, rerun_reason: str | None = None
) -> RunManifest:
    started = time.time()
    kind = config.run.kind
    if kind not in ("development", "confirmation", "holdout", "demo"):
        raise InvalidInputError(f"backtest does not support run kind {kind!r}")
    dataset = prepare_dataset(config, offline=offline)
    registry = RunRegistry(config.run.artifacts_dir)
    frozen = None
    if kind == "holdout":
        frozen = verify_protocol(config, dataset.source_hashes, dataset.exclusions_hash)
        if frozen.holdout_runs and not rerun_reason:
            raise InvalidInputError(
                f"holdout already run {len(frozen.holdout_runs)} time(s); pass --rerun-reason to "
                "log a labeled rerun (e.g. a bug correction)"
            )
    run_id = new_run_id(kind)
    run_dir = registry.create_run_dir(run_id)
    try:
        return _run_backtest_into(
            config, dataset, registry, frozen, run_id, run_dir, started, rerun_reason=rerun_reason
        )
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _run_backtest_into(
    config: ExperimentConfig,
    dataset: Dataset,
    registry: RunRegistry,
    frozen: Any,
    run_id: str,
    run_dir: Path,
    started: float,
    *,
    rerun_reason: str | None,
) -> RunManifest:
    kind = config.run.kind
    feats_hash = features_hash(dataset.features)
    dev_seasons = list(config.evaluation.development_seasons)
    score_seasons = config.score_seasons()
    candidates = kind in ("development", "demo")
    specs = model_variants(config, candidates=candidates)
    all_seasons = list(range(config.model.residual_first_season, max(score_seasons) + 1))
    notes: list[str] = []
    if dataset.synthetic:
        notes.append(SYNTHETIC_LABEL)

    fits: dict[str, dict[int, SeasonFit]] = {}
    for spec in specs:
        fits[spec.model_id] = chronological_fits(
            dataset, spec, all_seasons, config.model.train_start_season
        )

    # Stage 1: seasons scored for every variant (development seasons, or the single season).
    stage1 = dev_seasons if candidates else score_seasons
    preds: dict[str, list[dict[str, Any]]] = {s.model_id: [] for s in specs}
    dists: dict[str, list[dict[str, Any]]] = {s.model_id: [] for s in specs}
    season_info: dict[str, list[dict[str, Any]]] = {s.model_id: [] for s in specs}
    for spec in specs:
        for season in stage1:
            rows, d, info = _score_season(
                config, dataset, fits[spec.model_id], spec, season, run_id, run_dir, feats_hash
            )
            preds[spec.model_id] += rows
            dists[spec.model_id] += d
            season_info[spec.model_id].append(info)

    scored: dict[str, pd.DataFrame] = {}
    for spec in specs:
        df = _finalize_predictions(preds[spec.model_id])
        scored[spec.model_id] = per_game_scores(df) if len(df) else df

    decision: dict[str, Any] | None = None
    selected_ids = [s.model_id for s in specs]
    if candidates:
        decision = {
            "full": select_alpha(
                scored, specs, "full", dev_seasons, config.model.alpha_tie_tolerance
            )
        }
        if config.evaluation.ablation:
            decision["epa_free"] = select_alpha(
                scored, specs, "epa_free", dev_seasons, config.model.alpha_tie_tolerance
            )
        decision["frozen_before"] = {
            "confirmation_season": config.evaluation.confirmation_season,
            "holdout_season": config.evaluation.holdout_season,
        }
        write_json(run_dir / "decision_record.json", decision)
        selected_ids = ["B0_league_baseline", decision["full"]["selected_model_id"]]
        if "epa_free" in decision:
            selected_ids.append(decision["epa_free"]["selected_model_id"])
        if kind == "demo":  # continue to confirmation and holdout with the selected alphas only
            later = [s for s in score_seasons if s not in dev_seasons]
            for spec in specs:
                if spec.model_id not in selected_ids:
                    continue
                for season in later:
                    rows, d, info = _score_season(
                        config,
                        dataset,
                        fits[spec.model_id],
                        spec,
                        season,
                        run_id,
                        run_dir,
                        feats_hash,
                    )
                    preds[spec.model_id] += rows
                    dists[spec.model_id] += d
                    season_info[spec.model_id].append(info)
                df = _finalize_predictions(preds[spec.model_id])
                scored[spec.model_id] = per_game_scores(df) if len(df) else df

    primary_id = (
        next(m for m in selected_ids if m.startswith("M1_ridge_alpha"))
        if any(m.startswith("M1_ridge_alpha") for m in selected_ids)
        else selected_ids[-1]
    )
    baseline_id = "B0_league_baseline"

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "run_kind": kind,
        "data_mode": config.data.mode,
        "forecast_policy": dataset.policy.forecast_policy_label,
        "synthetic": dataset.synthetic,
        "score_seasons": score_seasons,
        "primary_model_id": primary_id,
        "baseline_model_id": baseline_id,
        "selected_model_ids": selected_ids,
        "models": {},
        "comparisons": {},
        "seasons": season_info,
        "exclusions": {"n": len(dataset.exclusions), "hash": dataset.exclusions_hash},
        "coverage": dataset.coverage.model_dump() if dataset.coverage else None,
    }
    for model_id, df in scored.items():
        metrics["models"][model_id] = _model_report(df, config.evaluation.sparse_bin_threshold)
        metrics["models"][model_id]["n_games"] = len(df)
    base = scored.get(baseline_id)
    for model_id, df in scored.items():
        if model_id == baseline_id or base is None or df.empty or base.empty:
            continue
        metrics["comparisons"][f"{model_id}_vs_{baseline_id}"] = block_bootstrap_differences(
            df, base, replicates=config.evaluation.bootstrap_replicates, seed=config.seed
        )
    if decision is not None:
        metrics["decision_record"] = decision

    output_hashes: dict[str, str] = {}
    for model_id, df in scored.items():
        path = run_dir / f"predictions_{model_id}.parquet"
        write_parquet(df, path)
        output_hashes[path.name] = sha256_file(path)
        if dists[model_id]:
            dpath = run_dir / f"distributions_{model_id}.parquet"
            write_parquet(pd.DataFrame(dists[model_id]), dpath)
    primary_path = run_dir / "predictions.parquet"
    write_parquet(scored[primary_id], primary_path)
    output_hashes["predictions.parquet"] = sha256_file(primary_path)
    write_json(run_dir / "metrics.json", metrics)
    output_hashes["metrics.json"] = sha256_file(run_dir / "metrics.json")
    dataset.exclusions.to_csv(run_dir / "exclusions.csv", index=False)
    write_json(run_dir / "resolved_config.json", config.resolved_dict())

    holdout_count = None
    if kind == "holdout":
        holdout_count = record_holdout_run(
            config.run.artifacts_dir, run_id, rerun_reason=rerun_reason, label=config.run.label
        )
        if holdout_count > 1:
            notes.append(f"HOLDOUT RERUN #{holdout_count}: {rerun_reason}")
    git = git_info()
    primary_seasons = [i for i in season_info[primary_id] if i.get("status") == "scored"]
    manifest = RunManifest(
        run_id=run_id,
        run_kind=kind,
        label=config.run.label,
        created_at_utc=iso_utc(utc_now()) or "",
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        platform=environment_note(),
        dependency_versions=dependency_versions(),
        seed=config.seed,
        config=config.resolved_dict(),
        config_hash=config.config_hash(),
        source_file_hashes=dataset.source_hashes,
        schema_version=SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        model_id=primary_id,
        training_cutoff_utc=primary_seasons[-1]["training_cutoff_utc"] if primary_seasons else None,
        forecast_policy=dataset.policy.forecast_policy_label,
        data_mode=config.data.mode,
        evaluation_mode=kind,
        output_hashes=output_hashes,
        protocol_checksum=frozen.checksum if frozen else None,
        notes=notes,
        runtime_seconds=round(time.time() - started, 2),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump())
    registry.record(manifest)
    return manifest


# ----------------------------------------------------------------------------------------------
# Forecast bundle fit and slate prediction
# ----------------------------------------------------------------------------------------------


def forecast_bundle_dir(config: ExperimentConfig, forecast_season: int) -> Path:
    return config.run.artifacts_dir / "models" / f"forecast_{forecast_season}"


def fit_forecast_bundles(
    config: ExperimentConfig, through_season: int, *, offline: bool
) -> tuple[RunManifest, dict[str, Path]]:
    """Fit B0 and M1 on eligible completed seasons through ``through_season`` for the next one.

    The fit time is the wall-clock time of the fit; only labels available by then are used.
    """
    if config.model.selected_alpha is None:
        raise InvalidInputError("fit requires model.selected_alpha from the development decision")
    started = time.time()
    seasons = list(range(config.data.seasons[0], through_season + 1))
    dataset = prepare_dataset(config, offline=offline, seasons=seasons)
    forecast_season = through_season + 1
    specs = model_variants(config, candidates=False)
    out_dir = forecast_bundle_dir(config, forecast_season)
    paths: dict[str, Path] = {}
    fit_time = pd.Timestamp(utc_now())
    labeled = _labeled_rows(dataset)
    train_all = labeled[
        labeled["season"].isin(training_seasons(forecast_season, config.model.train_start_season))
    ]
    train_all, dropped = _available_training_rows(train_all, fit_time)
    if train_all.empty:
        raise MissingDataError("no training labels available at fit time")
    res_seasons = residual_seasons(
        forecast_season, config.model.residual_first_season, config.model.residual_window_seasons
    )
    for spec in specs:
        fits = chronological_fits(dataset, spec, res_seasons, config.model.train_start_season)
        residuals, ids = residual_pool(fits, res_seasons, available_by=fit_time)
        params = fit_residual_params(
            residuals,
            seasons=[s for s in res_seasons if s in fits],
            game_ids_hash=game_ids_hash(ids),
            min_games=config.model.min_residual_games,
            diagonal_shrinkage=config.model.covariance_diagonal_shrinkage,
            floor=config.model.covariance_floor,
        )
        model, used = fit_score_model(train_all, dataset.labels, spec)
        bundle = build_bundle(
            model,
            spec,
            used,
            params,
            ids,
            forecast_season=forecast_season,
            policy=dataset.policy,
            features_cfg=config.features,
            config_hash=config.config_hash(),
            notes=[
                f"fit through season {through_season} for {forecast_season} forecasting",
                f"fit_time_utc={iso_utc(fit_time.to_pydatetime())}",
                f"training_rows_dropped_unavailable={dropped}",
            ],
        )
        paths[spec.model_id] = bundle.save(out_dir / f"{spec.model_id}.json")
    registry = RunRegistry(config.run.artifacts_dir)
    run_id = new_run_id("fit")
    run_dir = registry.create_run_dir(run_id)
    git = git_info()
    manifest = RunManifest(
        run_id=run_id,
        run_kind="fit",
        label=f"forecast_{forecast_season}",
        created_at_utc=iso_utc(utc_now()) or "",
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        platform=environment_note(),
        dependency_versions=dependency_versions(),
        seed=config.seed,
        config=config.resolved_dict(),
        config_hash=config.config_hash(),
        source_file_hashes=dataset.source_hashes,
        schema_version=SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        model_id=next(s.model_id for s in specs if s.family == "ridge_score"),
        training_cutoff_utc=ModelBundle.load(next(iter(paths.values()))).training_cutoff_utc,
        forecast_policy=dataset.policy.forecast_policy_label,
        data_mode=config.data.mode,
        evaluation_mode="fit",
        output_hashes={k: sha256_file(v) for k, v in paths.items()},
        notes=[
            f"bundles saved under {out_dir}",
            f"fit_time_utc={iso_utc(fit_time.to_pydatetime())}",
        ],
        runtime_seconds=round(time.time() - started, 2),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump())
    registry.record(manifest)
    return manifest, paths


SlateLoader = Callable[
    [ExperimentConfig, list[int], bool], tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]
]


def load_slate_inputs(
    config: ExperimentConfig, seasons: list[int], offline: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Games and completed team-games from the cache for a live/as-of slate."""
    manifest_src = ingest(seasons, config.data.cache_dir, offline=offline, refresh=False)
    policy = policy_from_config(config)
    data = normalize_sources(
        manifest_src,
        seasons,
        game_type=config.data.game_type,
        completed_game_lag_hours=policy.completed_game_lag_hours,
    )
    team_games = aggregate_team_games(data.games, data.plays, data.results)
    return data.games, team_games, manifest_src.file_hashes()


def predict_slate(
    config: ExperimentConfig,
    season: int,
    week: int | None,
    *,
    as_of: pd.Timestamp | None,
    offline: bool,
    reconstruct_standard_horizon: bool = False,
    loader: SlateLoader | None = None,
) -> tuple[RunManifest, pd.DataFrame]:
    """Price games with the saved forecast bundle; never trains on the fly.

    Three explicit horizons:

    - live (default): information time is the generation time; only games that have not kicked
      off are priced; labeled ``custom_horizon``.
    - ``as_of``: an explicit research time in the past; games kicking off after it are priced.
    - ``reconstruct_standard_horizon``: each game's own ``kickoff − 24h`` cutoff, allowed only
      once that cutoff has passed; a historical reconstruction, never a live forecast.
    """
    started = time.time()
    now = pd.Timestamp(utc_now())
    if as_of is not None and reconstruct_standard_horizon:
        raise InvalidInputError("use either --as-of or --reconstruct-standard-horizon, not both")
    if as_of is not None and as_of > now:
        raise InvalidInputError(
            f"--as-of {as_of} is in the future; information time cannot exceed now"
        )
    bundle_dir = forecast_bundle_dir(config, season)
    if not bundle_dir.exists():
        raise MissingDataError(
            f"no forecast bundle for season {season} under {bundle_dir}; "
            f"run `fit --through-season {season - 1}`"
        )
    bundles = {p.stem: ModelBundle.load(p) for p in sorted(bundle_dir.glob("*.json"))}
    primary = next((b for b in bundles.values() if b.family == "ridge_score"), None)
    if primary is None:
        raise MissingDataError("forecast bundle directory has no ridge_score bundle")
    policy = policy_from_config(config)
    for bundle in bundles.values():
        check_bundle_compatible(bundle, policy, config.features)
    seasons = list(range(config.data.seasons[0], season + 1))
    games, team_games, source_hashes = (loader or load_slate_inputs)(config, seasons, offline)
    targets = games[games["season"] == season]
    if week is not None:
        targets = targets[targets["week"] == week]
    if reconstruct_standard_horizon:
        horizon = "standard_reconstruction"
        cutoffs = policy.cutoff_for(targets["kickoff_utc"])
        pending = targets[cutoffs > now]
        if len(pending):
            raise InvalidInputError(
                f"{len(pending)} games have not reached their standard cutoff yet; use the live "
                "forecast (no --as-of) or wait"
            )
        cutoff_override = None
    else:
        horizon = "as_of" if as_of is not None else "live"
        cutoff_override = as_of if as_of is not None else now
        targets = targets[targets["kickoff_utc"] > cutoff_override]
    registry = RunRegistry(config.run.artifacts_dir)
    run_id = new_run_id("forecast")
    run_dir = registry.create_run_dir(run_id)
    try:
        return _predict_slate_into(
            config,
            season,
            week,
            horizon=horizon,
            cutoff_override=cutoff_override,
            started=started,
            bundles=bundles,
            primary=primary,
            source_hashes=source_hashes,
            policy=policy,
            team_games=team_games,
            targets=targets,
            now=now,
            registry=registry,
            run_id=run_id,
            run_dir=run_dir,
        )
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def _predict_slate_into(
    config: ExperimentConfig,
    season: int,
    week: int | None,
    *,
    horizon: str,
    cutoff_override: pd.Timestamp | None,
    started: float,
    bundles: dict[str, ModelBundle],
    primary: ModelBundle,
    source_hashes: dict[str, str],
    policy: AsOfPolicy,
    team_games: pd.DataFrame,
    targets: pd.DataFrame,
    now: pd.Timestamp,
    registry: RunRegistry,
    run_id: str,
    run_dir: Path,
) -> tuple[RunManifest, pd.DataFrame]:
    notes: list[str] = [f"horizon={horizon}"]
    if horizon == "standard_reconstruction":
        notes.append(
            "historical reconstruction of the standard kickoff-24h horizon; not a live forecast"
        )
    training_modes = sorted({b.data_mode for b in bundles.values()})
    notes.append(
        f"training_data_mode={','.join(training_modes)}; forecast_input_mode={policy.mode}"
    )
    rows: list[dict[str, Any]] = []
    dists: list[dict[str, Any]] = []
    if targets.empty:
        notes.append("empty slate: no games match the requested season/week/horizon filter")
        feats = pd.DataFrame()
    else:
        feats = build_features(
            targets, team_games, policy, config.features, cutoff_override_utc=cutoff_override
        )
        require_forecastable(feats)
        assert_no_market_columns(list(feats.columns), "predict_slate")
        feats_hash = features_hash(feats)
        for game in targets.sort_values(["kickoff_utc", "game_id"]).itertuples(index=False):
            g = pd.Series(game._asdict())
            gf = feats[feats["game_id"] == game.game_id]
            cutoff = pd.Timestamp(gf["cutoff_utc"].iloc[0])
            standard_cutoff = pd.Timestamp(policy.cutoff_for(pd.Timestamp(game.kickoff_utc)))
            if cutoff > now:
                raise ModelValidationError("internal error: forecast cutoff after generation time")
            forecast_policy = (
                policy.forecast_policy_label if cutoff == standard_cutoff else "custom_horizon"
            )
            flags = _quality_flags(gf)
            if policy.mode == "recorded_asof":
                observed = gf["max_observed_utc"].max()
                if pd.notna(observed) and observed > cutoff:
                    raise MissingDataError(
                        "recorded_asof: source data was first observed after the cutoff; "
                        "refusing to backdate"
                    )
            for bundle in bundles.values():
                created = pd.Timestamp(bundle.created_at_utc)
                if policy.mode == "recorded_asof" and created > cutoff:
                    raise MissingDataError(
                        f"recorded_asof: bundle {bundle.model_id} was created at {created}, "
                        f"after the decision time {cutoff}"
                    )
                model = bundle.score_model(config.seed)
                mu = predict_location(model, gf)
                loc = game_locations(gf, mu).iloc[0]
                dist = predict_distribution(
                    np.array([loc["mu_home_score"], loc["mu_away_score"]]),
                    bundle.residual_params(),
                    config.distribution,
                )
                row = prediction_row(
                    dist,
                    run_id=run_id,
                    game=g,
                    cutoff=cutoff,
                    model_id=bundle.model_id,
                    data_mode=policy.mode,
                    forecast_policy=forecast_policy,
                    config_hash=config.config_hash(),
                    features_hash_value=feats_hash,
                    quality_flags=flags,
                )
                row["model_created_utc"] = created
                row["training_data_mode"] = bundle.data_mode
                row["horizon"] = horizon
                rows.append(row)
                dists.append(
                    {
                        "game_id": game.game_id,
                        "model_id": bundle.model_id,
                        "max_score": dist.max_score,
                        "margin_pmf": dist.margin_pmf.tolist(),
                        "total_pmf": dist.total_pmf.tolist(),
                    }
                )
        write_parquet(feats, run_dir / "features.parquet")
    by_model: dict[str, pd.DataFrame] = {}
    for model_id in bundles:
        frame = _finalize_predictions([r for r in rows if r["model_id"] == model_id])
        if len(frame):
            frame["model_created_utc"] = pd.to_datetime(frame["model_created_utc"], utc=True)
            frame["information_time_utc"] = frame["cutoff_utc"]
        by_model[model_id] = frame
        write_parquet(frame, run_dir / f"predictions_{model_id}.parquet")
    slate = by_model.get(primary.model_id, _finalize_predictions([]))
    write_parquet(slate, run_dir / "predictions.parquet")
    if dists:
        dist_frame = pd.DataFrame(dists)
        write_parquet(dist_frame, run_dir / "distributions.parquet")
        for model_id in bundles:
            part = dist_frame[dist_frame["model_id"] == model_id]
            if len(part):
                write_parquet(part, run_dir / f"distributions_{model_id}.parquet")
    slate.to_csv(run_dir / "slate.csv", index=False)
    git = git_info()
    if horizon == "standard_reconstruction":
        manifest_policy = policy.forecast_policy_label
    else:
        manifest_policy = "custom_horizon"
    manifest = RunManifest(
        run_id=run_id,
        run_kind="forecast",
        label=f"{season}_week{week if week is not None else 'all'}",
        created_at_utc=iso_utc(utc_now()) or "",
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        platform=environment_note(),
        dependency_versions=dependency_versions(),
        seed=config.seed,
        config=config.resolved_dict(),
        config_hash=config.config_hash(),
        source_file_hashes=source_hashes,
        schema_version=SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        model_id=primary.model_id,
        training_cutoff_utc=primary.training_cutoff_utc,
        forecast_policy=manifest_policy,
        data_mode=policy.mode,
        evaluation_mode="forecast",
        output_hashes={"predictions.parquet": sha256_file(run_dir / "predictions.parquet")},
        notes=notes
        + (
            [f"information_time_utc={iso_utc(cutoff_override.to_pydatetime())}"]
            if cutoff_override is not None
            else []
        ),
        runtime_seconds=round(time.time() - started, 2),
    )
    write_json(run_dir / "manifest.json", manifest.model_dump())
    registry.record(manifest)
    return manifest, slate
