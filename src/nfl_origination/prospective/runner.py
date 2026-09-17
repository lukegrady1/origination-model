"""One-shot prospective cycle (V2 R3): collect permitted snapshots, forecast due games, exit.

No scheduler is installed. An external scheduler may call ``prospective-tick`` every five
minutes; each call is bounded, idempotent, and reads the frozen active epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ExperimentConfig
from nfl_origination.data.download import SCHEDULE_DATASET, SourceManifest, ingest
from nfl_origination.data.normalize import normalize_sources
from nfl_origination.data.snapshots import manifest_as_of
from nfl_origination.data.storage import write_parquet
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.features.aggregate import aggregate_team_games
from nfl_origination.features.asof import AsOfPolicy
from nfl_origination.features.builder import build_features, require_forecastable
from nfl_origination.market.asof import QuotePolicy, select_quotes
from nfl_origination.market.import_csv import import_odds
from nfl_origination.models.bundle import (
    ModelBundle,
    check_bundle_compatible,
    game_locations,
    predict_location,
)
from nfl_origination.models.distribution import ScoreDistribution, predict_distribution
from nfl_origination.pricing.fair_lines import fair_home_handicap, fair_moneyline, fair_total
from nfl_origination.pricing.markets import MARKET_ORDER, MarketSpec, settlement_probabilities
from nfl_origination.pricing.odds import expected_value
from nfl_origination.prospective.clock import Clock, SystemClock, iso
from nfl_origination.prospective.ledger import (
    DecisionRecord,
    ForecastRecord,
    Ledger,
    forecast_id,
    pmf_sha256,
)
from nfl_origination.prospective.protocol import Epoch, require_verified_epoch
from nfl_origination.protocol import lock_digest
from nfl_origination.provenance import hash_frame

SYNTHETIC_DATASETS = ("synthetic_games", "synthetic_results", "synthetic_team_games")


@dataclass
class SlateInputs:
    games: pd.DataFrame
    results: pd.DataFrame
    team_games: pd.DataFrame
    manifest: SourceManifest
    max_observed_utc: pd.Timestamp | None

    def receipts(self) -> dict[str, str]:
        return {
            f"{e.dataset}/{e.season if e.season is not None else 'all'}": (e.receipt_id or "")
            for e in self.manifest.entries
        }

    def hashes(self) -> dict[str, str]:
        return self.manifest.file_hashes()


def load_prospective_inputs(config: ExperimentConfig, at: pd.Timestamp) -> SlateInputs:
    """Games/results/team-games from the source versions observed at or before ``at``."""
    v2 = config.require_v2()
    cache = v2.storage.receipts_cache_dir
    seasons = config.data.season_list
    if config.data.source == "synthetic":
        manifest = manifest_as_of(
            cache,
            seasons,
            at,
            allow_synthetic=True,
            targets=[(d, None) for d in SYNTHETIC_DATASETS],
        )
        frames = {e.dataset: pd.read_parquet(e.path) for e in manifest.entries}
        games = frames["synthetic_games"].copy()
        results = frames["synthetic_results"].copy()
        tg = frames["synthetic_team_games"].copy()
        sched_obs = pd.Timestamp(manifest.entry("synthetic_games", None).first_observed_at_utc)
        tg_obs = pd.Timestamp(manifest.entry("synthetic_team_games", None).first_observed_at_utc)
        games["first_observed_utc"] = sched_obs
        tg["schedule_first_observed_utc"] = sched_obs
        tg["pbp_first_observed_utc"] = tg_obs
        tg["first_observed_utc"] = max(sched_obs, tg_obs)
        for c in ("kickoff_utc",):
            games[c] = pd.to_datetime(games[c], utc=True)
            tg[c] = pd.to_datetime(tg[c], utc=True)
        observed = max(sched_obs, tg_obs)
        return SlateInputs(games, results, tg, manifest, observed)
    manifest = manifest_as_of(cache, seasons, at)
    data = normalize_sources(
        manifest,
        seasons,
        game_type=config.data.game_type,
        completed_game_lag_hours=v2.sources.completed_game_lag_hours,
    )
    tg = aggregate_team_games(data.games, data.plays, data.results)
    observed = max(pd.Timestamp(e.first_observed_at_utc) for e in manifest.entries)
    return SlateInputs(data.games, data.results, tg, manifest, observed)


def prospective_policy(config: ExperimentConfig) -> AsOfPolicy:
    v2 = config.require_v2()
    return AsOfPolicy(
        mode="recorded_asof",
        completed_game_lag_hours=v2.sources.completed_game_lag_hours,
        cutoff_hours_before_kickoff=v2.horizon.target_hours_before_kickoff,
    )


def predict_joint(
    bundle: ModelBundle, feature_rows: pd.DataFrame, config: ExperimentConfig
) -> tuple[ScoreDistribution, dict[str, Any], tuple[float, float]]:
    """Joint score distribution for one game from any supported bundle family."""
    if bundle.family in ("ridge_score", "league_baseline"):
        model = bundle.score_model(config.seed)
        mu = predict_location(model, feature_rows)
        loc = game_locations(feature_rows, mu).iloc[0]
        location = (float(loc["mu_home_score"]), float(loc["mu_away_score"]))
        dist = predict_distribution(
            np.array(location), bundle.residual_params(), config.distribution
        )
        diag = {
            "base_omitted_upper_mass": dist.omitted_upper_mass,
            "base_negative_latent_mass": dist.negative_latent_mass,
            "max_score": dist.max_score,
        }
        return dist, diag, location
    if bundle.family == "key_number_adjusted":
        from nfl_origination.models.key_number import predict_adjusted

        return predict_adjusted(bundle, feature_rows, config)
    raise ModelValidationError(f"unsupported bundle family {bundle.family!r}")


def _intervals(dist: ScoreDistribution) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for level, tag in ((0.5, "50"), (0.8, "80"), (0.95, "95")):
        out[f"margin_{tag}"] = list(dist.margin_interval(level))
        out[f"total_{tag}"] = list(dist.total_interval(level))
    return out


def _key_probabilities(dist: ScoreDistribution) -> dict[str, float]:
    return {
        "tie": dist.p_tie(),
        "margin_3": dist.margin_probability(3),
        "margin_-3": dist.margin_probability(-3),
        "margin_7": dist.margin_probability(7),
        "margin_-7": dist.margin_probability(-7),
        "total_even": float(dist.total_pmf[0::2].sum()),
    }


@dataclass
class TickSummary:
    protocol_id: str
    now_utc: str
    in_scope_games: int
    due_games: list[str]
    forecasts_created: list[str]
    forecasts_existing: list[str]
    forecast_failures: list[dict[str, str]]
    missed_recorded: list[str]
    decisions: dict[str, int]
    collection: dict[str, Any] | None
    max_source_observed_utc: str | None
    synthetic: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _quotes_store(odds_dir: Path) -> pd.DataFrame:
    files = sorted((Path(odds_dir) / "quotes").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def persist_collected_quotes(
    odds_dir: Path, receipt_id: str, quotes: pd.DataFrame, games: pd.DataFrame, synthetic: bool
) -> int:
    """Run collected quotes through the canonical import and store them per receipt."""
    if quotes.empty:
        return 0
    tmp_csv = Path(odds_dir) / "quotes" / f".{receipt_id}.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_csv(tmp_csv, index=False)
    odds, _rep = import_odds(tmp_csv, games=games[["game_id", "kickoff_utc"]], synthetic=synthetic)
    tmp_csv.unlink()
    write_parquet(odds, Path(odds_dir) / "quotes" / f"{receipt_id}.parquet")
    return len(odds)


def collect_odds_once(
    config: ExperimentConfig, *, clock: Clock | None = None, provider: Any = None
) -> dict[str, Any]:
    """One bounded provider collection; persists a receipt and canonical quotes."""
    v2 = config.require_v2()
    clock = clock or SystemClock()
    if not v2.market.enabled:
        return {"status": "market_disabled"}
    now = clock.now()
    inputs = load_prospective_inputs(config, now)
    if provider is None:
        from nfl_origination.market.providers import CollectionBudget, TheOddsApiAdapter

        provider = TheOddsApiAdapter(
            v2.storage.odds_dir,
            sport_key=v2.collection.sport_key,
            regions=v2.collection.regions,
            timeout_seconds=v2.collection.timeout_seconds,
            max_attempts=v2.collection.max_attempts_per_request,
            budget=CollectionBudget(
                v2.storage.odds_dir,
                v2.collection.request_budget_per_invocation,
                v2.collection.quota_reserve,
            ),
            clock=lambda: clock.now().to_pydatetime(),
        )
    books = tuple(v2.market.bookmakers) or ((v2.market.bookmaker,) if v2.market.bookmaker else ())
    result = provider.collect(
        bookmakers=books,
        markets=tuple(v2.market.main_markets),
        games=inputs.games,
        settlement_rules=v2.market.settlement_rules,
        synthetic=v2.evidence.synthetic,
    )
    stored = persist_collected_quotes(
        v2.storage.odds_dir,
        result.receipt.receipt_id,
        result.quotes,
        inputs.games,
        v2.evidence.synthetic,
    )
    return {
        "status": result.receipt.status,
        "receipt_id": result.receipt.receipt_id,
        "events_seen": result.events_seen,
        "quotes_stored": stored,
        "quarantined": len(result.quarantined),
        "quota": result.receipt.quota,
        "error": result.receipt.error,
    }


def _paper_decision(
    *,
    record: ForecastRecord,
    dist: ScoreDistribution,
    quotes: pd.DataFrame,
    config: ExperimentConfig,
    cutoff: pd.Timestamp,
    clock: Clock,
    synthetic: bool,
) -> DecisionRecord:
    v2 = config.require_v2()
    policy_id = (
        f"ev{v2.paper.min_ev:g}_flat{v2.paper.stake_units:g}"
        f"_max{v2.paper.max_selections_per_game_per_model}"
    )
    base: dict[str, Any] = dict(
        forecast_id=record.forecast_id,
        protocol_id=record.protocol_id,
        game_id=record.game_id,
        role=record.role,
        model_id=record.model_id,
        decision_time_utc=iso(cutoff),
        policy_id=policy_id,
        committed_at_utc=iso(clock.now()),
        synthetic=synthetic,
    )
    if not v2.market.enabled or v2.market.bookmaker is None:
        return DecisionRecord(
            **base,
            status="market_unavailable",
            reason="market_disabled",
            market_probabilities=[],
            selection=None,
            exclusions=[],
        )
    if quotes.empty:
        return DecisionRecord(
            **base,
            status="market_unavailable",
            reason="no_quotes_collected",
            market_probabilities=[],
            selection=None,
            exclusions=[],
        )
    qpolicy = QuotePolicy(
        v2.market.bookmaker, v2.market.max_quote_age_minutes, True, "two_way_tie_void"
    )
    eligible = select_quotes(
        quotes, cutoff, qpolicy, game_id=record.game_id, require_local_observation=True
    )
    if eligible.empty:
        return DecisionRecord(
            **base,
            status="market_unavailable",
            reason="no_eligible_pair_at_cutoff",
            market_probabilities=[],
            selection=None,
            exclusions=eligible.exclusions,
        )
    probs_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for q in eligible.quotes.itertuples(index=False):
        rule = v2.market.settlement_rules.get(q.bookmaker, {}).get(q.market)
        if rule is None:
            probs_rows.append(
                {
                    "quote_id": q.quote_id,
                    "market": q.market,
                    "selection": q.selection,
                    "status": "no_settlement_rule",
                }
            )
            continue
        spec = MarketSpec.from_line(
            q.market, q.selection, None if pd.isna(q.line) else float(q.line)
        )
        sp = settlement_probabilities(dist, spec)
        ev = expected_value(sp.p_win, sp.p_push, sp.p_loss, float(q.decimal_odds))
        row = {
            "quote_id": q.quote_id,
            "pair_id": getattr(q, "pair_id", None),
            "market": q.market,
            "selection": q.selection,
            "line": None if pd.isna(q.line) else float(q.line),
            "decimal_odds": float(q.decimal_odds),
            "p_win": sp.p_win,
            "p_push": sp.p_push,
            "p_loss": sp.p_loss,
            "ev": ev,
            "settlement_rule": rule,
            "snapshot_at_utc": iso(pd.Timestamp(q.snapshot_at_utc)),
        }
        probs_rows.append(row)
        if ev >= v2.paper.min_ev:
            candidates.append(row)
    if not candidates:
        return DecisionRecord(
            **base,
            status="no_bet",
            reason=f"no selection with EV >= {v2.paper.min_ev}",
            market_probabilities=probs_rows,
            selection=None,
            exclusions=eligible.exclusions,
        )
    order = {m: i for i, m in enumerate(v2.paper.market_order)}
    candidates.sort(
        key=lambda r: (
            -r["ev"],
            order.get(r["market"], MARKET_ORDER.get(r["market"], 9)),
            r["selection"],
        )
    )
    chosen = {**candidates[0], "stake": v2.paper.stake_units}
    return DecisionRecord(
        **base,
        status="bet",
        reason="highest_ev_under_frozen_policy",
        market_probabilities=probs_rows,
        selection=chosen,
        exclusions=eligible.exclusions,
    )


def tick(
    config: ExperimentConfig,
    *,
    clock: Clock | None = None,
    protocol_id: str | None = None,
    provider: Any = None,
) -> TickSummary:
    """Read the active epoch, collect permitted snapshots, forecast due games, write, exit."""
    v2 = config.require_v2()
    clock = clock or SystemClock()
    epoch: Epoch = require_verified_epoch(config, protocol_id)
    ledger = Ledger(v2.storage.artifacts_dir, epoch.protocol_id)
    synthetic = bool(epoch.synthetic)
    now = clock.now()
    inputs = load_prospective_inputs(config, now)
    games = inputs.games
    scope_seasons = (
        set(epoch.scope["seasons"]) if isinstance(epoch.scope.get("seasons"), list) else set()
    )
    if len(scope_seasons) == 2:
        lo, hi = sorted(scope_seasons)
        scope_seasons = set(range(int(lo), int(hi) + 1))
    in_scope = games[games["season"].isin(scope_seasons)] if scope_seasons else games
    if "game_type" in in_scope.columns:
        in_scope = in_scope[in_scope["game_type"] == epoch.scope.get("game_type", "REG")]
    target_hours = float(epoch.horizon["target_hours_before_kickoff"])
    window = pd.Timedelta(minutes=float(epoch.horizon["window_minutes"]))
    horizon_id = str(epoch.horizon["horizon_policy_id"])
    target_cutoffs = in_scope["kickoff_utc"] - pd.Timedelta(hours=target_hours)
    final_ids = set(inputs.results["game_id"]) if len(inputs.results) else set()
    not_started = (in_scope["kickoff_utc"] > now) & ~in_scope["game_id"].isin(final_ids)
    due = in_scope[not_started & ((target_cutoffs - now).abs() <= window)]
    missed_mask = (target_cutoffs + window < now) & (in_scope["kickoff_utc"] > now)
    missed = in_scope[missed_mask]
    closing_soon = in_scope[
        (in_scope["kickoff_utc"] > now)
        & (
            in_scope["kickoff_utc"]
            <= now
            + pd.Timedelta(minutes=float(epoch.market_policy["closing_proxy"]["window_minutes"]))
        )
    ]

    collection: dict[str, Any] | None = None
    if v2.market.enabled and (len(due) or len(closing_soon)):
        if provider is None and synthetic:
            collection = {"status": "no_provider_in_synthetic_mode"}
        else:
            try:
                collection = collect_odds_once(config, clock=clock, provider=provider)
                if collection.get("status") != "ok":
                    ledger.append_event({"event": "collection_failure", **collection}, clock)
            except Exception as exc:
                collection = {"status": "error", "error": str(exc)[:200]}
                ledger.append_event({"event": "collection_failure", **collection}, clock)
    quotes = _quotes_store(v2.storage.odds_dir) if v2.market.enabled else pd.DataFrame()

    recorded_missed: list[str] = []
    existing_missed = {
        e.get("game_id") for e in ledger.events() if e.get("event") == "missed_forecast_window"
    }
    for g in missed.itertuples(index=False):
        fid = forecast_id(epoch.protocol_id, str(g.game_id), horizon_id, epoch.champion.bundle_hash)
        if ledger.committed_forecast(fid) is None and g.game_id not in existing_missed:
            ledger.append_event(
                {
                    "event": "missed_forecast_window",
                    "game_id": str(g.game_id),
                    "kickoff_utc": iso(pd.Timestamp(g.kickoff_utc)),
                    "target_cutoff_utc": iso(
                        pd.Timestamp(g.kickoff_utc) - pd.Timedelta(hours=target_hours)
                    ),
                },
                clock,
            )
            recorded_missed.append(str(g.game_id))

    created: list[str] = []
    existing: list[str] = []
    failures: list[dict[str, str]] = []
    decisions = {
        "bet": 0,
        "no_bet": 0,
        "market_unavailable": 0,
        "model_unavailable": 0,
        "existing": 0,
    }
    policy = prospective_policy(config)
    refs = [epoch.champion] + ([epoch.challenger] if epoch.challenger else [])
    bundles = {r.role: ModelBundle.load(Path(r.bundle_path)) for r in refs}
    for b in bundles.values():
        check_bundle_compatible(b, policy, config.features)
    for g in due.sort_values(["kickoff_utc", "game_id"]).itertuples(index=False):
        game_id = str(g.game_id)
        kickoff = pd.Timestamp(g.kickoff_utc)
        target_cutoff = kickoff - pd.Timedelta(hours=target_hours)
        cutoff = clock.now()  # real current time after input collection, frozen for this game
        target = in_scope[in_scope["game_id"] == game_id]
        try:
            feats = build_features(
                target, inputs.team_games, policy, config.features, cutoff_override_utc=cutoff
            )
            require_forecastable(feats)
            if inputs.max_observed_utc is not None and inputs.max_observed_utc > cutoff:
                raise MissingDataError(
                    "a required source receipt was observed after the information cutoff"
                )
        except (MissingDataError, ModelValidationError) as exc:
            failures.append({"game_id": game_id, "role": "all", "reason": str(exc)[:200]})
            ledger.append_event(
                {"event": "forecast_failure", "game_id": game_id, "reason": str(exc)[:200]}, clock
            )
            continue
        feats_hash = hash_frame(feats)
        for ref in refs:
            bundle = bundles[ref.role]
            fid = forecast_id(epoch.protocol_id, game_id, horizon_id, ref.bundle_hash)
            if ledger.committed_forecast(fid) is not None:
                existing.append(fid)
                decisions["existing"] += 1
                continue
            status, reason = "valid", None
            if pd.Timestamp(bundle.created_at_utc) > cutoff:
                status, reason = "invalid_timing", "bundle_created_after_cutoff"
            try:
                dist, diag, location = predict_joint(bundle, feats, config)
            except (MissingDataError, ModelValidationError) as exc:
                failures.append({"game_id": game_id, "role": ref.role, "reason": str(exc)[:200]})
                ledger.append_event(
                    {
                        "event": "forecast_failure",
                        "game_id": game_id,
                        "role": ref.role,
                        "reason": str(exc)[:200],
                    },
                    clock,
                )
                continue
            ml = fair_moneyline(dist)
            hc = fair_home_handicap(dist)
            tt = fair_total(dist)
            record = ForecastRecord(
                forecast_id=fid,
                protocol_id=epoch.protocol_id,
                game_id=game_id,
                season=int(g.season),
                week=int(g.week),
                home_team=str(g.home_team),
                away_team=str(g.away_team),
                role=ref.role,
                model_id=bundle.model_id,
                model_family=bundle.family,
                bundle_hash=ref.bundle_hash,
                horizon_policy_id=horizon_id,
                kickoff_at_forecast_utc=iso(kickoff),
                target_cutoff_utc=iso(target_cutoff),
                information_cutoff_utc=iso(cutoff),
                created_at_utc=iso(clock.now()),
                source_receipts=inputs.receipts(),
                source_hashes=inputs.hashes(),
                max_source_observed_utc=None
                if inputs.max_observed_utc is None
                else iso(inputs.max_observed_utc),
                feature_rows_hash=feats_hash,
                feature_rows=_feature_rows_payload(feats),
                mu_home_score=location[0],
                mu_away_score=location[1],
                dist_mean_home_score=dist.mean_home,
                dist_mean_away_score=dist.mean_away,
                mean_margin=dist.mean_margin,
                mean_total=dist.mean_total,
                p_home_win=ml.p_home,
                p_tie=ml.p_tie,
                p_away_win=ml.p_away,
                fair_home_handicap=hc.line,
                fair_total=tt.line,
                fair_decimal_home=ml.decimal_home,
                fair_decimal_away=ml.decimal_away,
                intervals=_intervals(dist),
                key_probabilities=_key_probabilities(dist),
                max_score=dist.max_score,
                margin_pmf=dist.margin_pmf.tolist(),
                total_pmf=dist.total_pmf.tolist(),
                pmf_sha256=pmf_sha256(dist.pmf),
                tail_diagnostics=diag,
                code_digest=epoch.code_digest,
                config_hash=epoch.config_hash,
                lock_digest=lock_digest(),
                training_evidence_mode=ref.training_evidence_mode,
                forecast_evidence_mode=ref.forecast_evidence_mode,
                synthetic=synthetic,
                quality_flags=_flags(feats),
                status=status,
                status_reason=reason,
            )
            _rec, _manifest, was_created = ledger.commit_forecast(
                record,
                dist.pmf,
                clock,
                kickoff=kickoff,
                window_minutes=float(epoch.horizon["window_minutes"]),
                target_hours=target_hours,
            )
            (created if was_created else existing).append(fid)
            decision = _paper_decision(
                record=record,
                dist=dist,
                quotes=quotes,
                config=config,
                cutoff=cutoff,
                clock=clock,
                synthetic=synthetic,
            )
            _d, d_created = ledger.commit_decision(decision)
            decisions[decision.status if d_created else "existing"] += 1
    summary = TickSummary(
        protocol_id=epoch.protocol_id,
        now_utc=iso(now),
        in_scope_games=len(in_scope),
        due_games=[str(x) for x in due["game_id"]],
        forecasts_created=created,
        forecasts_existing=existing,
        forecast_failures=failures,
        missed_recorded=recorded_missed,
        decisions=decisions,
        collection=collection,
        max_source_observed_utc=None
        if inputs.max_observed_utc is None
        else iso(inputs.max_observed_utc),
        synthetic=synthetic,
    )
    ledger.append_event(
        {"event": "tick", **{k: v for k, v in summary.to_dict().items() if k != "notes"}}, clock
    )
    return summary


def _feature_rows_payload(feats: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for rec in feats.to_dict("records"):
        clean: dict[str, Any] = {}
        for k, v in rec.items():
            if isinstance(v, pd.Timestamp):
                clean[k] = iso(v) if pd.notna(v) else None
            elif isinstance(v, (np.floating, float)):
                clean[k] = None if pd.isna(v) else float(v)
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        rows.append(clean)
    return rows


def _flags(feats: pd.DataFrame) -> list[str]:
    flags = []
    if feats["team_cold_start"].any():
        flags.append("cold_start")
    if feats["team_rest_missing"].any():
        flags.append("rest_missing")
    if (feats["team_history_games"] < 16).any():
        flags.append("short_history")
    return flags


def settle(
    config: ExperimentConfig,
    *,
    clock: Clock | None = None,
    protocol_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Append result versions for committed forecasts from the newest observed schedule.

    ``refresh`` explicitly fetches new source observations (network); otherwise settlement uses
    the cache. New observations are distinguished from cached settlement in the summary.
    """
    v2 = config.require_v2()
    clock = clock or SystemClock()
    epoch = require_verified_epoch(config, protocol_id)
    ledger = Ledger(v2.storage.artifacts_dir, epoch.protocol_id)
    refreshed = None
    if refresh and config.data.source != "synthetic":
        fresh = ingest(
            config.data.season_list, v2.storage.receipts_cache_dir, offline=False, refresh=True
        )
        refreshed = {e.dataset + "/" + str(e.season): e.receipt_id for e in fresh.entries}
    now = clock.now()
    inputs = load_prospective_inputs(config, now)
    sched_entry = next(
        (e for e in inputs.manifest.entries if e.dataset in (SCHEDULE_DATASET, "synthetic_games")),
        None,
    )
    results = inputs.results.set_index("game_id") if len(inputs.results) else pd.DataFrame()
    games = inputs.games.set_index("game_id")
    appended: list[str] = []
    reschedules: list[str] = []
    pending: list[str] = []
    for record, _man in ledger.list_forecasts():
        gid = record.game_id
        if gid not in games.index:
            pending.append(gid)
            continue
        kickoff_now = pd.Timestamp(games.loc[gid, "kickoff_utc"])
        status = "final" if gid in results.index else "pending"
        note = None
        if iso(kickoff_now) != record.kickoff_at_forecast_utc:
            status = "rescheduled_review_required"
            note = (
                f"kickoff changed from {record.kickoff_at_forecast_utc} to {iso(kickoff_now)} "
                "after commitment"
            )
            existing = {
                e.get("game_id") for e in ledger.events() if e.get("event") == "reschedule_observed"
            }
            if gid not in existing:
                ledger.append_event(
                    {
                        "event": "reschedule_observed",
                        "game_id": gid,
                        "old_kickoff_utc": record.kickoff_at_forecast_utc,
                        "new_kickoff_utc": iso(kickoff_now),
                    },
                    clock,
                )
                reschedules.append(gid)
        if status == "pending":
            pending.append(gid)
            continue
        hs = int(results.loc[gid, "home_score"]) if gid in results.index else None
        as_ = int(results.loc[gid, "away_score"]) if gid in results.index else None
        _out, was_new = ledger.append_outcome(
            gid,
            status=status,
            home_score=hs,
            away_score=as_,
            kickoff_utc=iso(kickoff_now),
            source_receipt_id=None if sched_entry is None else sched_entry.receipt_id,
            source_hash=None if sched_entry is None else sched_entry.sha256,
            observed_at_utc=(sched_entry.first_observed_at_utc if sched_entry else iso(now)),
            clock=clock,
            note=note,
        )
        if was_new:
            appended.append(gid)
    return {
        "protocol_id": epoch.protocol_id,
        "as_of_utc": iso(now),
        "new_result_versions": appended,
        "reschedules_observed": reschedules,
        "pending": sorted(set(pending)),
        "refreshed_sources": refreshed,
        "source_observation": None if sched_entry is None else sched_entry.first_observed_at_utc,
    }


def verify(config: ExperimentConfig, *, protocol_id: str | None = None) -> dict[str, Any]:
    v2 = config.require_v2()
    from nfl_origination.prospective.protocol import load_epoch, verify_epoch

    epoch = load_epoch(v2.storage.artifacts_dir, protocol_id)
    problems = verify_epoch(epoch, config)
    ledger = Ledger(v2.storage.artifacts_dir, epoch.protocol_id)
    report = ledger.verify()
    report["epoch_problems"] = problems
    report["epoch_status"] = "ok" if not problems else "epoch_mismatch"
    return report


def resolve_default_bundles(
    config: ExperimentConfig, champion: Path | None, challenger: Path | None
) -> tuple[Path, Path | None]:
    """Default champion/challenger bundle paths for the next season under the V2 artifacts root."""
    v2 = config.require_v2()
    season = config.data.seasons[1] + 1
    root = v2.storage.artifacts_dir / "models" / f"forecast_{season}"
    champ = champion or (root / f"{v2.models.champion_model_id}.json")
    if not champ.exists():
        raise MissingDataError(
            f"champion bundle not found: {champ}; run `fit` with this config first"
        )
    chal = challenger
    if chal is None and v2.models.challenger_model_id:
        cand = root / f"{v2.models.challenger_model_id}.json"
        chal = cand if cand.exists() else None
    if chal is not None and not chal.exists():
        raise MissingDataError(f"challenger bundle not found: {chal}")
    return champ, chal


def fit_prospective_bundles(
    config: ExperimentConfig,
    *,
    clock: Clock | None = None,
    through_season: int | None = None,
    extra_specs: list[Any] | None = None,
) -> dict[str, Path]:
    """Fit V2-era bundles from the source versions observed now.

    Training uses retrospective reconstruction of completed seasons (the practical startup
    constraint of V2 section 5.3); the bundle declares that explicitly together with the strict
    observed-availability contract for live inputs. The fit time is the clock time.
    """
    from nfl_origination.experiment import fit_forecast_bundles

    v2 = config.require_v2()
    clock = clock or SystemClock()
    now = clock.now()
    through = through_season if through_season is not None else config.data.seasons[1]
    seasons = list(range(config.data.seasons[0], through + 1))
    cache = v2.storage.receipts_cache_dir
    if config.data.source == "synthetic":
        manifest = manifest_as_of(
            cache,
            seasons,
            now,
            allow_synthetic=True,
            targets=[(d, None) for d in SYNTHETIC_DATASETS],
        )
    else:
        manifest = manifest_as_of(cache, seasons, now)
    training_config = config.model_copy(
        update={"data": config.data.model_copy(update={"mode": "historical_reconstruction"})}
    )
    execution_contract = {
        "training_evidence_mode": "retrospective_reconstruction",
        "forecast_evidence_mode": v2.evidence.mode,
        "fit_time_utc": iso(now),
        "source_receipts": {
            f"{e.dataset}/{e.season if e.season is not None else 'all'}": e.receipt_id
            for e in manifest.entries
        },
        "note": "training inputs are completed seasons as observed at fit time; live forecast "
        "inputs must satisfy strict observed availability",
    }
    out_dir = v2.storage.artifacts_dir / "models" / f"forecast_{through + 1}"
    _manifest, paths = fit_forecast_bundles(
        training_config,
        through,
        offline=True,
        fit_time=now,
        source_manifest=manifest,
        out_dir=out_dir,
        execution_contract=execution_contract,
        extra_specs=extra_specs,
    )
    return paths
