"""Typer CLI: entry point ``nfl-origination`` (spec section 15)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer

from nfl_origination.config import (
    ExperimentConfig,
    load_config,
    parse_overrides,
    parse_season_range,
)
from nfl_origination.errors import InvalidInputError, MissingDataError, OriginationError

app = typer.Typer(
    name="nfl-origination",
    help="Reproducible NFL pregame price origination research pipeline (V1).",
    add_completion=False,
    no_args_is_help=True,
)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config path")]
OfflineOpt = Annotated[bool, typer.Option("--offline", help="Prohibit all network access")]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override config key, e.g. --set model.selected_alpha=100"),
]


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=1, default=str))


def _load(config: Path, overrides: list[str] | None) -> ExperimentConfig:
    return _run(load_config, config, parse_overrides(overrides or []))


def _run(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except OriginationError as exc:
        typer.secho(f"error [{exc.__class__.__name__}]: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(code=exc.exit_code) from exc


@app.command()
def ingest(
    seasons: Annotated[str, typer.Option(help="Inclusive range, e.g. 2010:2025")] = "2010:2025",
    config: ConfigOpt = Path("configs/v1.yaml"),
    offline: OfflineOpt = False,
) -> None:
    """Download/refresh nflverse schedule and play-by-play assets into the immutable cache."""
    from nfl_origination.data.download import ingest as do_ingest
    from nfl_origination.data.download import manifest_summary
    from nfl_origination.provenance import write_json

    cfg = _load(config, None)
    season_list = parse_season_range(seasons)
    manifest = _run(do_ingest, season_list, cfg.data.cache_dir, offline)
    out = cfg.data.cache_dir / "source_manifest.json"
    write_json(out, manifest.model_dump())
    _echo_json({**manifest_summary(manifest), "manifest": str(out)})


@app.command("validate-data")
def validate_data(
    seasons: Annotated[str, typer.Option(help="Inclusive range, e.g. 2010:2025")] = "2010:2025",
    config: ConfigOpt = Path("configs/v1.yaml"),
    offline: OfflineOpt = True,
) -> None:
    """Audit season coverage and write the exclusion table (no modeling)."""
    from nfl_origination.experiment import prepare_dataset

    cfg = _load(config, None)
    dataset = _run(prepare_dataset, cfg, offline=offline, seasons=parse_season_range(seasons))
    assert dataset.coverage is not None
    _echo_json(
        {
            "seasons": [s.model_dump() for s in dataset.coverage.seasons],
            "all_complete": dataset.coverage.all_complete,
            "notes": dataset.coverage.notes,
            "exclusions": len(dataset.exclusions),
            "written": str(cfg.data.normalized_dir),
        }
    )
    if not dataset.coverage.all_complete:
        typer.secho(
            "coverage incomplete: see coverage_report.json and exclusions.csv",
            fg=typer.colors.YELLOW,
        )


@app.command("build-features")
def build_features_cmd(
    config: ConfigOpt = Path("configs/v1.yaml"),
    offline: OfflineOpt = True,
    rebuild: Annotated[bool, typer.Option(help="Ignore the feature cache")] = False,
    set_: SetOpt = None,
) -> None:
    """Build as-of historical features and the generated feature dictionary."""
    from nfl_origination.experiment import prepare_dataset
    from nfl_origination.features.builder import feature_dictionary

    cfg = _load(config, set_)
    dataset = _run(prepare_dataset, cfg, offline=offline, rebuild=rebuild)
    dict_path = Path("docs/feature_dictionary_generated.md")
    dict_path.parent.mkdir(exist_ok=True)
    rows = feature_dictionary()
    header = "| " + " | ".join(rows[0].keys()) + " |\n|" + "|".join(["---"] * len(rows[0])) + "|\n"
    body = "".join("| " + " | ".join(str(v) for v in r.values()) + " |\n" for r in rows)
    dict_path.write_text("# Generated feature dictionary\n\n" + header + body)
    _echo_json(
        {
            "feature_rows": len(dataset.features),
            "games": int(dataset.features["game_id"].nunique()),
            "insufficient_warmup_rows": int(dataset.features["insufficient_warmup"].sum()),
            "feature_dictionary": str(dict_path),
            "features_dir": str(cfg.data.features_dir),
        }
    )


@app.command()
def backtest(
    config: ConfigOpt = Path("configs/development.yaml"),
    offline: OfflineOpt = True,
    rerun_reason: Annotated[str | None, typer.Option(help="Required to rerun a holdout")] = None,
    set_: SetOpt = None,
) -> None:
    """Run the chronological backtest for the config's run kind."""
    from nfl_origination.experiment import run_backtest

    cfg = _load(config, set_)
    manifest = _run(run_backtest, cfg, offline=offline, rerun_reason=rerun_reason)
    run_dir = cfg.run.artifacts_dir / "runs" / manifest.run_id
    metrics = json.loads((run_dir / "metrics.json").read_text())
    summary = {
        "run_id": manifest.run_id,
        "run_kind": manifest.run_kind,
        "primary_model_id": metrics["primary_model_id"],
        "runtime_seconds": manifest.runtime_seconds,
        "run_dir": str(run_dir),
    }
    if "decision_record" in metrics:
        summary["selected_alpha"] = metrics["decision_record"]["full"]["selected_alpha"]
        if "epa_free" in metrics["decision_record"]:
            summary["ablation_selected_alpha"] = metrics["decision_record"]["epa_free"][
                "selected_alpha"
            ]
    for mid in metrics["selected_model_ids"]:
        pooled = metrics["models"][mid]["pooled"]
        summary[mid] = {
            k: pooled.get(k)
            for k in ("n_games", "score_mae", "margin_mae", "logloss_3way", "crps_margin")
        }
    _echo_json(summary)


@app.command("freeze-protocol")
def freeze_protocol_cmd(
    config: ConfigOpt = Path("configs/holdout.yaml"),
    development_run: Annotated[
        str | None, typer.Option(help="Run ID or latest-development")
    ] = "latest-development",
    confirmation_run: Annotated[
        str | None, typer.Option(help="Run ID or latest-confirmation")
    ] = "latest-confirmation",
    offline: OfflineOpt = True,
    set_: SetOpt = None,
) -> None:
    """Write the checksummed frozen protocol that gates the holdout backtest."""
    from nfl_origination.experiment import prepare_dataset
    from nfl_origination.protocol import freeze_protocol
    from nfl_origination.provenance import RunRegistry

    cfg = _load(config, set_)
    dataset = _run(prepare_dataset, cfg, offline=offline)
    registry = RunRegistry(cfg.run.artifacts_dir)

    def resolve(ref: str | None) -> str | None:
        if ref is None:
            return None
        try:
            return registry.resolve(ref)
        except InvalidInputError:
            return None

    frozen = _run(
        freeze_protocol,
        cfg,
        dataset.source_hashes,
        dataset.exclusions_hash,
        development_run_id=resolve(development_run),
        confirmation_run_id=resolve(confirmation_run),
    )
    _echo_json(
        {
            "label": frozen.label,
            "checksum": frozen.checksum,
            "code_digest": frozen.code_digest,
            "lock_digest": frozen.lock_digest,
            "frozen_at_utc": frozen.frozen_at_utc,
            "selected_alpha": frozen.selected_alpha,
            "holdout_season": frozen.holdout_season,
        }
    )


@app.command()
def report(
    run: Annotated[str, typer.Option(help="Run ID or latest-<kind>")] = "latest-holdout",
    config: ConfigOpt = Path("configs/v1.yaml"),
) -> None:
    """Build the Markdown report, charts, metrics JSON and prediction exports for a run."""
    from nfl_origination.evaluation.report import build_report
    from nfl_origination.provenance import RunRegistry

    cfg = _load(config, None)
    registry = RunRegistry(cfg.run.artifacts_dir)
    run_id = _run(registry.resolve, run)
    typer.echo(f"resolved {run} -> {run_id}")
    path = _run(build_report, registry.run_dir(run_id), cfg.run.reports_dir)
    _echo_json({"run_id": run_id, "report": str(path)})


@app.command()
def fit(
    through_season: Annotated[int, typer.Option(help="Last completed season to train on")],
    config: ConfigOpt = Path("configs/v1.yaml"),
    offline: OfflineOpt = True,
    set_: SetOpt = None,
) -> None:
    """Fit the separate next-season forecasting bundles (B0 and M1; V2 adds the challenger)."""
    from nfl_origination.experiment import fit_forecast_bundles

    cfg = _load(config, set_)
    if cfg.v2 is not None:
        from nfl_origination.prospective.clock import SystemClock
        from nfl_origination.prospective.runner import (
            fit_challenger_bundle,
            fit_prospective_bundles,
            load_decision,
        )

        clock = SystemClock()
        paths = _run(fit_prospective_bundles, cfg, clock=clock, through_season=through_season)
        decision = _run(load_decision, cfg)
        champion = paths[cfg.v2.models.champion_model_id]
        chal = _run(
            fit_challenger_bundle,
            cfg,
            base_bundle_path=champion,
            through_season=through_season,
            clock=clock,
            decision=decision,
        )
        _echo_json(
            {
                "bundles": {k: str(v) for k, v in paths.items()},
                "challenger": None if chal is None else str(chal),
                "decision": None if decision is None else decision.get("model_decision"),
            }
        )
        return
    manifest, paths = _run(fit_forecast_bundles, cfg, through_season, offline=offline)
    _echo_json({"run_id": manifest.run_id, "bundles": {k: str(v) for k, v in paths.items()}})


@app.command()
def predict(
    season: Annotated[int, typer.Option(help="NFL season to forecast")],
    week: Annotated[int | None, typer.Option(help="Week; omit for all remaining games")] = None,
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Explicit past research time, ISO-8601 UTC")
    ] = None,
    reconstruct_standard_horizon: Annotated[
        bool,
        typer.Option(
            "--reconstruct-standard-horizon",
            help="Historically reconstruct each game's kickoff-24h cutoff (must have passed)",
        ),
    ] = False,
    config: ConfigOpt = Path("configs/v1.yaml"),
    offline: OfflineOpt = True,
    set_: SetOpt = None,
) -> None:
    """Price upcoming games from the saved bundle at each cutoff or an explicit as-of time."""
    from nfl_origination.experiment import predict_slate

    cfg = _load(config, set_)
    as_of_ts = None
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is None:
            raise typer.BadParameter("--as-of must include a UTC offset, e.g. 2026-09-16T12:00:00Z")
        as_of_ts = as_of_ts.tz_convert("UTC")
    manifest, slate = _run(
        predict_slate,
        cfg,
        season,
        week,
        as_of=as_of_ts,
        offline=offline,
        reconstruct_standard_horizon=reconstruct_standard_horizon,
    )
    run_dir = cfg.run.artifacts_dir / "runs" / manifest.run_id
    if slate.empty:
        typer.echo("empty slate: no games matched the request (valid empty artifact written)")
    else:
        cols = [
            "game_id",
            "kickoff_utc",
            "cutoff_utc",
            "fair_home_handicap",
            "fair_total",
            "fair_american_home",
            "fair_american_away",
            "p_tie",
            "forecast_policy",
            "data_quality_flags",
        ]
        typer.echo(slate[cols].to_string(index=False))
    _echo_json(
        {
            "run_id": manifest.run_id,
            "games": len(slate),
            "forecast_policy": manifest.forecast_policy,
            "run_dir": str(run_dir),
            "notes": manifest.notes,
        }
    )


@app.command("import-odds")
def import_odds_cmd(
    path: Annotated[Path, typer.Option(help="Canonical odds CSV")],
    config: ConfigOpt = Path("configs/v1.yaml"),
    synthetic: Annotated[bool, typer.Option(help="Label the file as synthetic")] = False,
    out: Annotated[Path | None, typer.Option(help="Output parquet path")] = None,
) -> None:
    """Import and validate a user-supplied odds CSV (never requires a provider subscription)."""
    from nfl_origination.market.import_csv import import_odds
    from nfl_origination.provenance import write_json

    cfg = _load(config, None)
    games_path = cfg.data.normalized_dir / "games.parquet"
    games = pd.read_parquet(games_path) if games_path.exists() else None
    out_path = out or Path("data/user/odds.parquet")
    synthetic = synthetic or cfg.market.synthetic
    _odds, rep = _run(import_odds, path, games=games, synthetic=synthetic, output_path=out_path)
    write_json(out_path.with_suffix(".import_report.json"), rep.model_dump())
    _echo_json({**rep.summary(), "output": str(out_path), "synthetic": synthetic})


def _load_odds(cfg: ExperimentConfig) -> pd.DataFrame:
    path = Path("data/user/odds.parquet")
    if cfg.run.artifacts_dir != Path("artifacts"):
        alt = cfg.run.artifacts_dir / "odds.parquet"
        if alt.exists():
            path = alt
    if not path.exists():
        raise MissingDataError(
            "no imported odds; run import-odds first (optional market data absent)"
        )
    return pd.read_parquet(path)


@app.command()
def compare(
    run: Annotated[str, typer.Option(help="Run ID or latest-<kind>")] = "latest-forecast",
    config: ConfigOpt = Path("configs/v1.yaml"),
) -> None:
    """Compare saved predictions with contemporaneous quotes selected at each cutoff."""
    from nfl_origination.market.asof import QuotePolicy
    from nfl_origination.market.compare import UNAVAILABLE, compare_predictions
    from nfl_origination.provenance import RunRegistry, write_json

    cfg = _load(config, None)
    registry = RunRegistry(cfg.run.artifacts_dir)
    run_id = _run(registry.resolve, run)
    run_dir = registry.run_dir(run_id)
    manifest = registry.load_manifest(run_id)
    try:
        odds = _load_odds(cfg)
    except MissingDataError as exc:
        typer.echo(f"{UNAVAILABLE} ({exc})")
        write_json(run_dir / "market_comparison.json", {"status": UNAVAILABLE})
        return
    if cfg.market.bookmaker is None:
        typer.echo(f"{UNAVAILABLE} (market.bookmaker not configured)")
        return
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    preds = preds[preds["model_id"] == manifest.model_id] if "model_id" in preds else preds
    dist_path = run_dir / f"distributions_{manifest.model_id}.parquet"
    if not dist_path.exists():
        dist_path = run_dir / "distributions.parquet"  # lookup is still keyed by model_id
    dists = pd.read_parquet(dist_path) if dist_path.exists() else None
    policy = QuotePolicy(
        cfg.market.bookmaker,
        cfg.market.max_quote_age_minutes,
        True,
        cfg.market.moneyline_settlement_rule,
    )
    table, summary = _run(compare_predictions, preds, dists, odds, policy)
    table.to_parquet(run_dir / "market_comparison.parquet", index=False)
    write_json(run_dir / "market_comparison.json", summary)
    _echo_json(summary)


@app.command("paper-backtest")
def paper_backtest_cmd(
    run: Annotated[str, typer.Option(help="Run ID or latest-<kind>")] = "latest-holdout",
    config: ConfigOpt = Path("configs/v1.yaml"),
) -> None:
    """Paper-only settlement of eligible quotes against saved predictions (frozen policy)."""
    from nfl_origination.market.settle import UNAVAILABLE, paper_backtest
    from nfl_origination.provenance import RunRegistry, write_json

    cfg = _load(config, None)
    registry = RunRegistry(cfg.run.artifacts_dir)
    run_id = _run(registry.resolve, run)
    run_dir = registry.run_dir(run_id)
    manifest = registry.load_manifest(run_id)
    try:
        odds = _load_odds(cfg)
    except MissingDataError as exc:
        typer.echo(f"{UNAVAILABLE} ({exc})")
        write_json(run_dir / "paper_backtest.json", {"status": UNAVAILABLE})
        return
    if cfg.market.bookmaker is None:
        typer.echo(f"{UNAVAILABLE} (market.bookmaker not configured)")
        return
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    dist_path = run_dir / f"distributions_{manifest.model_id}.parquet"
    if not dist_path.exists():
        dist_path = run_dir / "distributions.parquet"
    dists = pd.read_parquet(dist_path)
    results_path = cfg.data.normalized_dir / "results.parquet"
    if "actual_home" in preds.columns:
        results = preds[["game_id", "actual_home", "actual_away"]].rename(
            columns={"actual_home": "home_score", "actual_away": "away_score"}
        )
    elif results_path.exists():
        results = pd.read_parquet(results_path)
    else:
        results = pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    bets, summary = _run(
        paper_backtest,
        run_id,
        preds,
        dists,
        odds,
        results,
        cfg.market,
        seed=cfg.seed,
        bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
    )
    bets.to_parquet(run_dir / "paper_bets.parquet", index=False)
    write_json(run_dir / "paper_backtest.json", summary)
    _echo_json(summary)


@app.command()
def demo(
    config: ConfigOpt = Path("configs/demo.yaml"),
    offline: OfflineOpt = True,
) -> None:
    """Offline review path: synthetic data, full protocol, pricing, odds settlement, report."""
    from nfl_origination.data.storage import write_parquet
    from nfl_origination.evaluation.report import build_report
    from nfl_origination.experiment import run_backtest
    from nfl_origination.features.aggregate import aggregate_team_games
    from nfl_origination.fixtures import load_mini_raw_fixture
    from nfl_origination.market.import_csv import import_odds
    from nfl_origination.market.settle import paper_backtest
    from nfl_origination.provenance import RunRegistry, write_json
    from nfl_origination.synthetic import SYNTHETIC_LABEL

    cfg = _load(config, None)
    if not offline:
        typer.echo("demo always runs offline")
    typer.secho(SYNTHETIC_LABEL, fg=typer.colors.YELLOW)
    # 1. normalization of the miniature raw fixture (schedule + play-by-play)
    sched, plays, results = _run(load_mini_raw_fixture)
    tg = _run(aggregate_team_games, sched, plays, results)
    typer.echo(
        f"[1/5] normalized mini raw fixture: {len(sched)} games, {len(plays)} plays -> "
        f"{len(tg)} team-game rows"
    )
    # 2. backtest with selection, confirmation and holdout on synthetic seasons
    manifest = _run(run_backtest, cfg, offline=True)
    run_dir = cfg.run.artifacts_dir / "runs" / manifest.run_id
    typer.echo(f"[2/5] backtest {manifest.run_id} in {manifest.runtime_seconds}s -> {run_dir}")
    # 3. synthetic odds import and paper settlement
    if cfg.market.enabled and cfg.market.odds_path is not None and cfg.market.bookmaker:
        games = pd.read_parquet(run_dir / "predictions.parquet")[
            ["game_id", "kickoff_utc"]
        ].drop_duplicates()
        odds, rep = _run(
            import_odds,
            cfg.market.odds_path,
            games=games,
            synthetic=True,
            output_path=cfg.run.artifacts_dir / "odds.parquet",
        )
        typer.echo(f"[3/5] odds import: {rep.summary()}")
        preds = pd.read_parquet(run_dir / "predictions.parquet")
        dists = pd.read_parquet(run_dir / f"distributions_{manifest.model_id}.parquet")
        results = preds[["game_id", "actual_home", "actual_away"]].rename(
            columns={"actual_home": "home_score", "actual_away": "away_score"}
        )
        bets, summary = _run(
            paper_backtest,
            manifest.run_id,
            preds,
            dists,
            odds,
            results,
            cfg.market,
            seed=cfg.seed,
            bootstrap_replicates=cfg.evaluation.bootstrap_replicates,
        )
        write_parquet(bets, run_dir / "paper_bets.parquet")
        write_json(run_dir / "paper_backtest.json", summary)
        typer.echo(
            f"[4/5] paper backtest: bets={summary.get('n_bets')} "
            f"roi={summary.get('roi_on_non_void_stake')} (synthetic)"
        )
    else:
        typer.echo("[3/5] market disabled; [4/5] unavailable: no eligible timestamped odds")
    path = _run(build_report, run_dir, cfg.run.reports_dir)
    typer.echo(f"[5/5] report -> {path}")
    registry = RunRegistry(cfg.run.artifacts_dir)
    _echo_json(
        {
            "run_id": manifest.run_id,
            "latest_demo": registry.resolve("latest-demo"),
            "report": str(path),
        }
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()


@app.command("snapshot-inventory")
def snapshot_inventory_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
) -> None:
    """Read-only inventory of source blobs, legacy V1 metadata and V2 observation receipts."""
    from nfl_origination.data.snapshots import snapshot_inventory

    cfg = _load(config, None)
    v2 = _run(cfg.require_v2)
    _echo_json(_run(snapshot_inventory, v2.storage.receipts_cache_dir))


@app.command("migrate-snapshots")
def migrate_snapshots_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    no_baseline: Annotated[
        bool, typer.Option("--no-baseline", help="Skip the 'observed now' baseline receipts")
    ] = False,
) -> None:
    """Additive, idempotent import of legacy V1 cache metadata into immutable receipts."""
    from nfl_origination.data.snapshots import migrate_legacy
    from nfl_origination.provenance import write_json

    cfg = _load(config, None)
    v2 = _run(cfg.require_v2)
    report = _run(migrate_legacy, v2.storage.receipts_cache_dir, baseline=not no_baseline)
    out = (
        v2.storage.artifacts_dir
        / "snapshots"
        / f"migration_{report['at_utc'].replace(':', '')}.json"
    )
    write_json(out, report)
    _echo_json(
        {
            "created": len(report["created"]),
            "skipped_existing": report["skipped_existing"],
            "report": str(out),
            "receipts_by_quality": report["inventory"]["receipts_by_quality"],
        }
    )


@app.command("collect-odds")
def collect_odds_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    offline: OfflineOpt = False,
) -> None:
    """Collect ONE bounded odds snapshot from the configured provider (needs ODDS_API_KEY)."""
    from nfl_origination.prospective.runner import collect_odds_once

    cfg = _load(config, None)
    if offline:
        typer.echo("collect-odds is a network command; --offline makes it a no-op")
        _echo_json({"status": "skipped_offline"})
        return
    summary = _run(collect_odds_once, cfg)
    _echo_json(summary)


ProtocolOpt = Annotated[str | None, typer.Option("--protocol-id", help="Epoch protocol ID")]


@app.command("freeze-prospective")
def freeze_prospective_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    champion: Annotated[Path | None, typer.Option(help="Champion bundle JSON")] = None,
    challenger: Annotated[Path | None, typer.Option(help="Challenger bundle JSON")] = None,
    label: Annotated[str, typer.Option(help="Epoch label")] = "v2",
) -> None:
    """Freeze an immutable prospective epoch (system time) and set the active pointer."""
    from nfl_origination.prospective.clock import SystemClock
    from nfl_origination.prospective.protocol import freeze_epoch
    from nfl_origination.prospective.runner import resolve_default_bundles

    cfg = _load(config, None)
    champ, chal = _run(resolve_default_bundles, cfg, champion, challenger)
    epoch = _run(
        freeze_epoch, cfg, SystemClock(), champion_path=champ, challenger_path=chal, label=label
    )
    _echo_json(
        {
            "protocol_id": epoch.protocol_id,
            "activated_at_utc": epoch.activated_at_utc,
            "champion": epoch.champion.model_id,
            "challenger": None if epoch.challenger is None else epoch.challenger.model_id,
            "code_digest": epoch.code_digest[:16],
        }
    )


@app.command("prospective-tick")
def prospective_tick_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    protocol_id: ProtocolOpt = None,
    offline: OfflineOpt = False,
) -> None:
    """One bounded cycle: read the epoch, collect permitted snapshots, forecast due games, exit."""
    from nfl_origination.prospective.clock import SystemClock
    from nfl_origination.prospective.runner import tick

    cfg = _load(config, None)
    if offline and cfg.v2 is not None and cfg.v2.market.enabled:
        typer.echo("--offline: market collection skipped this tick (replay cannot be prospective)")
    summary = _run(
        tick,
        cfg,
        clock=SystemClock(),
        protocol_id=protocol_id,
        provider=None if not offline else _no_provider(),
    )
    _echo_json(summary.to_dict())


def _no_provider() -> Any:
    class _Offline:
        def collect(self, **kwargs: Any) -> Any:
            raise RuntimeError("offline: no network collection")

    return _Offline()


@app.command("settle-prospective")
def settle_prospective_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    protocol_id: ProtocolOpt = None,
    refresh: Annotated[
        bool, typer.Option(help="Fetch new source observations first (network)")
    ] = False,
) -> None:
    """Append final results/corrections for committed forecasts; forecasts are never modified."""
    from nfl_origination.prospective.clock import SystemClock
    from nfl_origination.prospective.runner import settle

    cfg = _load(config, None)
    _echo_json(_run(settle, cfg, clock=SystemClock(), protocol_id=protocol_id, refresh=refresh))


@app.command("verify-ledger")
def verify_ledger_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    protocol_id: ProtocolOpt = None,
) -> None:
    """Recompute ledger hashes/markers and check the epoch against the current environment."""
    from nfl_origination.prospective.runner import verify

    cfg = _load(config, None)
    report = _run(verify, cfg, protocol_id=protocol_id)
    _echo_json(report)
    if report["status"] != "ok" or report["epoch_status"] != "ok":
        raise typer.Exit(code=4)


@app.command("demo-v2")
def demo_v2_cmd(
    config: ConfigOpt = Path("configs/v2_demo.yaml"),
    offline: OfflineOpt = True,
) -> None:
    """Synthetic end-to-end V2 lifecycle with a controlled clock; no network, no credentials."""
    from nfl_origination.prospective.demo import run_demo_v2
    from nfl_origination.synthetic import SYNTHETIC_LABEL

    cfg = _load(config, None)
    typer.secho(SYNTHETIC_LABEL, fg=typer.colors.YELLOW)
    if not offline:
        typer.echo("demo-v2 always runs offline")
    log = _run(run_demo_v2, cfg)
    _echo_json({k: v for k, v in log.items() if k != "steps"})


@app.command("research-v2")
def research_v2_cmd(
    config: ConfigOpt = Path("configs/v2_research.yaml"),
    offline: OfflineOpt = True,
) -> None:
    """Retrospective distribution experiment on cached data; writes a decision record."""
    from nfl_origination.evaluation.v2 import research_v2

    cfg = _load(config, None)
    if not offline:
        typer.echo("research-v2 reads the cache only; --offline is implied")
    metrics = _run(research_v2, cfg)
    dec = metrics["decision"]
    _echo_json(
        {
            "run_id": metrics["run_id"],
            "run_dir": metrics["run_dir"],
            "selected_candidate": dec["selected_candidate"],
            "model_decision": dec["model_decision"],
            "development": {
                k: {
                    kk: v.get(kk)
                    for kk in ("n_games", "crps_margin", "crps_total", "logloss_3way", "key_gap")
                }
                for k, v in metrics["development_metrics"].items()
            },
            "exclusions": len(metrics["exclusions"]),
        }
    )


@app.command("report-prospective")
def report_prospective_cmd(
    config: ConfigOpt = Path("configs/v2_prospective.yaml"),
    protocol_id: ProtocolOpt = None,
) -> None:
    """Evaluate committed forecasts against outcome versions as of now; write the report."""
    from nfl_origination.evaluation.v2 import prospective_report
    from nfl_origination.prospective.clock import SystemClock

    cfg = _load(config, None)
    report = _run(prospective_report, cfg, clock=SystemClock(), protocol_id=protocol_id)
    _echo_json(
        {
            k: report[k]
            for k in (
                "protocol_id",
                "results_as_of_utc",
                "coverage",
                "model_metrics",
                "review_gate",
                "model_decision",
                "integrity",
                "report_path",
            )
        }
    )
