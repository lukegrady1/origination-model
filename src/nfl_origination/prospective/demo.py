"""Offline V2 lifecycle demo on SYNTHETIC data with a controlled clock (V2 R7 `demo-v2`).

receipts -> bundle/epoch -> forecast/quote commit -> closing receipt -> final outcome ->
settlement -> report -> integrity verify, including one missed game and one missing book.
Nothing here is evidence about real games.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ExperimentConfig
from nfl_origination.data.snapshots import record_observation
from nfl_origination.data.storage import write_parquet
from nfl_origination.prospective.clock import FixedClock, iso
from nfl_origination.prospective.ledger import Ledger
from nfl_origination.prospective.protocol import freeze_epoch
from nfl_origination.prospective.runner import (
    fit_prospective_bundles,
    persist_collected_quotes,
    settle,
    tick,
    verify,
)
from nfl_origination.synthetic import SYNTHETIC_LABEL, generate_synthetic_dataset


def _write_blob(cache: Path, dataset: str, frame: pd.DataFrame, at: pd.Timestamp) -> str:
    blob_dir = cache / dataset / "all"
    blob_dir.mkdir(parents=True, exist_ok=True)
    tmp = blob_dir / f".tmp-{dataset}.parquet"
    write_parquet(frame, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    final = blob_dir / f"{digest}.parquet"
    if not final.exists():
        tmp.rename(final)
    else:
        tmp.unlink()
    receipt = record_observation(
        cache,
        dataset=dataset,
        season=None,
        blob_path=final,
        sha256=digest,
        request_started_at_utc=iso(at),
        observed_at_utc=iso(at),
        source_url="synthetic://generator",
        synthetic=True,
        row_count=len(frame),
        columns=list(frame.columns),
    )
    return receipt.receipt_id


def _synthetic_quotes(
    games: pd.DataFrame,
    results: pd.DataFrame,
    observed: pd.Timestamp,
    *,
    missing_game: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    merged = games.merge(results[["game_id", "home_score", "away_score"]], on="game_id")
    for g in merged.itertuples(index=False):
        if g.game_id == missing_game:
            continue  # the reference book does not quote this game -> market_unavailable
        margin = 0.5 * (g.home_score - g.away_score) + rng.normal(1.0, 6.0)
        spread = float(np.round(-margin * 2) / 2)
        if spread == int(spread):
            spread += 0.5
        total = float(np.round((45 + rng.normal(0, 3)) * 2) / 2)
        p_home = 1 / (1 + np.exp(spread / 7.0))
        kick = pd.Timestamp(g.kickoff_utc)
        base = {
            "provider_event_id": f"syn-{g.game_id}",
            "game_id": g.game_id,
            "bookmaker": "synthetic_book",
            "snapshot_at_utc": iso(observed),
            "bookmaker_updated_at_utc": iso(observed - timedelta(minutes=2)),
            "kickoff_at_snapshot_utc": iso(kick),
            "observed_at_utc": iso(observed),
            "provenance_mode": "synthetic",
            "provider": "synthetic_provider",
            "market_role": "main",
        }
        for market, sel, line, dec, rule in (
            ("spread", "home", spread, 1.9091, "push_refund"),
            ("spread", "away", -spread, 1.9091, "push_refund"),
            ("total", "over", total, 1.9091, "push_refund"),
            ("total", "under", total, 1.9091, "push_refund"),
            ("moneyline", "home", None, 1 / min(0.97, p_home * 1.04), "two_way_tie_void"),
            ("moneyline", "away", None, 1 / min(0.97, (1 - p_home) * 1.04), "two_way_tie_void"),
        ):
            key = f"{g.game_id}|{market}|{sel}|{iso(observed)}"
            rows.append(
                base
                | {
                    "quote_id": "sq-" + hashlib.sha256(key.encode()).hexdigest()[:16],
                    "market": market,
                    "selection": sel,
                    "line": "" if line is None else line,
                    "decimal_odds": round(dec, 4),
                    "settlement_rule": rule,
                }
            )
    return pd.DataFrame(rows)


def run_demo_v2(
    config: ExperimentConfig, *, clock: FixedClock | None = None, reset: bool = True
) -> dict[str, Any]:
    v2 = config.require_v2()
    if not v2.evidence.synthetic or config.data.source != "synthetic":
        raise ValueError(
            "demo-v2 requires a synthetic config (v2.evidence.synthetic and data.source)"
        )
    root = v2.storage.artifacts_dir
    if reset and root.exists():
        shutil.rmtree(root)
    cache = v2.storage.receipts_cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    first, last = config.data.seasons
    syn = generate_synthetic_dataset(
        seed=config.seed, seasons=(first, last), lag_hours=v2.sources.completed_game_lag_hours
    )
    games, results, team_games = syn.games, syn.results, syn.team_games
    season_start = pd.Timestamp(games[games["season"] == last]["kickoff_utc"].min())
    t0 = season_start - pd.Timedelta(days=30)
    clock = clock or FixedClock(t0)
    clock.set(t0)
    log: dict[str, Any] = {"label": SYNTHETIC_LABEL, "t0": iso(t0), "steps": []}

    # 1. receipts: schedule (all seasons), results and team-games only for completed seasons
    prior = results[results["game_id"].isin(games[games["season"] < last]["game_id"])]
    prior_tg = team_games[team_games["season"] < last]
    receipts = {
        "synthetic_games": _write_blob(cache, "synthetic_games", games, clock.now()),
        "synthetic_results": _write_blob(cache, "synthetic_results", prior, clock.now()),
        "synthetic_team_games": _write_blob(cache, "synthetic_team_games", prior_tg, clock.now()),
    }
    log["steps"].append({"step": "receipts", "receipts": receipts})

    # 2. bundles at t0 (training on completed seasons observed now) and epoch freeze
    clock.advance(timedelta(minutes=5))
    paths = fit_prospective_bundles(config, clock=clock, through_season=last - 1)
    challenger_path = None
    for model_id, path in paths.items():
        if model_id.startswith("KN_"):
            challenger_path = path
    clock.advance(timedelta(minutes=5))
    epoch = freeze_epoch(
        config,
        clock,
        champion_path=paths[v2.models.champion_model_id],
        challenger_path=challenger_path,
        label="v2_demo",
    )
    log["steps"].append(
        {
            "step": "epoch",
            "protocol_id": epoch.protocol_id,
            "bundles": {k: str(v) for k, v in paths.items()},
        }
    )

    # 3. week-1 games: forecast two on time, miss one, quote all but one (missing book)
    week1 = games[(games["season"] == last) & (games["week"] == 1)].sort_values("kickoff_utc")
    # ticks target the two earliest kickoff times; the latest kickoff (a distinct time) is
    # deliberately missed; one quoted game lacks the reference book
    kickoffs = sorted(week1["kickoff_utc"].unique())
    due_two = week1[week1["kickoff_utc"].isin(kickoffs[:2])].groupby("kickoff_utc").head(1)
    missed_row = week1[week1["kickoff_utc"] == kickoffs[-1]].iloc[0]
    missed_game = str(missed_row["game_id"])
    missing_book_game = str(week1[week1["kickoff_utc"] == kickoffs[1]].iloc[-1]["game_id"])
    target_hours = v2.horizon.target_hours_before_kickoff
    ticks = []
    for g in due_two.itertuples(index=False):
        cutoff_time = (
            pd.Timestamp(g.kickoff_utc) - pd.Timedelta(hours=target_hours) - pd.Timedelta(minutes=1)
        )
        clock.set(cutoff_time)
        quotes = _synthetic_quotes(
            week1,
            results,
            clock.now() - pd.Timedelta(minutes=10),
            missing_game=missing_book_game,
            seed=config.seed + int(g.week),
        )
        persist_collected_quotes(
            v2.storage.odds_dir, f"synthetic-{iso(clock.now())}", quotes, games, True
        )
        summary = tick(config, clock=clock, protocol_id=epoch.protocol_id)
        ticks.append(summary.to_dict())
        # re-running the same tick must be idempotent
        again = tick(config, clock=clock, protocol_id=epoch.protocol_id)
        assert not again.forecasts_created, "second tick must not create new forecasts"
    # the latest-kickoff game's window is missed: tick after the window closed, before kickoff
    missed_kick = pd.Timestamp(missed_row["kickoff_utc"])
    clock.set(missed_kick - pd.Timedelta(hours=target_hours) + pd.Timedelta(minutes=30))
    missed_summary = tick(config, clock=clock, protocol_id=epoch.protocol_id)
    ticks.append(missed_summary.to_dict())
    log["steps"].append(
        {
            "step": "ticks",
            "ticks": ticks,
            "missed_game": missed_game,
            "missing_book_game": missing_book_game,
        }
    )

    # 4. closing receipt 20 minutes before the first kickoff, then results after the games
    first_kick = pd.Timestamp(due_two.iloc[0]["kickoff_utc"])
    clock.set(first_kick - pd.Timedelta(minutes=20))
    closing = _synthetic_quotes(
        week1, results, clock.now(), missing_game=missing_book_game, seed=config.seed + 99
    )
    persist_collected_quotes(
        v2.storage.odds_dir, f"synthetic-close-{iso(clock.now())}", closing, games, True
    )
    clock.set(pd.Timestamp(week1["kickoff_utc"].max()) + pd.Timedelta(days=2))
    week1_results = results[
        results["game_id"].isin(pd.concat([games[games["season"] < last], week1])["game_id"])
    ]
    _write_blob(cache, "synthetic_results", week1_results, clock.now())
    settlement = settle(config, clock=clock, protocol_id=epoch.protocol_id)
    log["steps"].append({"step": "settle", **settlement})

    # 5. report + integrity
    from nfl_origination.evaluation.v2 import prospective_report

    report = prospective_report(config, clock=clock, protocol_id=epoch.protocol_id)
    integrity = verify(config, protocol_id=epoch.protocol_id)
    log["steps"].append({"step": "verify", **integrity})
    log["report"] = str(report["report_path"])
    log["protocol_id"] = epoch.protocol_id
    ledger = Ledger(root, epoch.protocol_id)
    log["forecasts_committed"] = len(ledger.list_forecasts())
    log["decisions"] = {
        d.status: sum(1 for x in ledger.list_decisions() if x.status == d.status)
        for d in ledger.list_decisions()
    }
    return log
