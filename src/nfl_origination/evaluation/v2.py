"""V2 reports: prospective evaluation from the ledger (research selection lives alongside)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ExperimentConfig
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.market.asof import QuotePolicy, closing_proxy
from nfl_origination.market.benchmark import (
    line_clv,
    paper_summary,
    price_clv,
    score_aligned,
    settle_paper,
)
from nfl_origination.models.distribution import MarginalScoreDistribution, discrete_crps
from nfl_origination.pricing.markets import MarketSpec
from nfl_origination.pricing.odds import no_vig_pair
from nfl_origination.prospective.clock import Clock, SystemClock, iso
from nfl_origination.prospective.ledger import Ledger, summarize_ledger
from nfl_origination.prospective.protocol import load_epoch
from nfl_origination.prospective.runner import _quotes_store, load_prospective_inputs
from nfl_origination.provenance import hash_json, write_json

EPS = 1e-12


def _model_scores(rec: Any, outcome: Any) -> dict[str, float]:
    hs, as_ = int(outcome.home_score), int(outcome.away_score)
    margin, total = hs - as_, hs + as_
    dist = MarginalScoreDistribution(
        np.asarray(rec.margin_pmf), np.asarray(rec.total_pmf), rec.max_score
    )
    y = 1.0 if margin > 0 else 0.0
    p3 = {1.0: rec.p_home_win, 0.0: rec.p_away_win}.get(y, rec.p_tie) if margin != 0 else rec.p_tie
    return {
        "score_mae": 0.5
        * (abs(hs - rec.dist_mean_home_score) + abs(as_ - rec.dist_mean_away_score)),
        "margin_mae": abs(margin - rec.mean_margin),
        "total_mae": abs(total - rec.mean_total),
        "logloss_3way": float(-np.log(max(p3, EPS))),
        "crps_margin": discrete_crps(dist.margin_pmf, dist.margins, margin),
        "crps_total": discrete_crps(dist.total_pmf, dist.totals, total),
        "is_tie": float(margin == 0),
        "p_tie": rec.p_tie,
    }


def prospective_report(
    config: ExperimentConfig,
    *,
    clock: Clock | None = None,
    protocol_id: str | None = None,
    results_as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Combine committed forecasts, decisions and outcome versions as of a results timestamp."""
    v2 = config.require_v2()
    clock = clock or SystemClock()
    now = clock.now()
    as_of = results_as_of or now
    epoch = load_epoch(v2.storage.artifacts_dir, protocol_id)
    ledger = Ledger(v2.storage.artifacts_dir, epoch.protocol_id)
    integrity = ledger.verify()
    excluded = {p["forecast_id"] for p in integrity["problems"]}
    forecasts = [(r, m) for r, m in ledger.list_forecasts() if r.forecast_id not in excluded]
    decisions = {d.forecast_id: d for d in ledger.list_decisions()}
    events = ledger.events()
    try:
        inputs = load_prospective_inputs(config, now)
        scheduled = inputs.games[
            (inputs.games["kickoff_utc"] > pd.Timestamp(epoch.activated_at_utc))
            & inputs.games["season"].isin(set(config.data.season_list))
        ]
        n_scheduled = len(scheduled)
    except Exception:
        n_scheduled = -1
    quotes = _quotes_store(v2.storage.odds_dir) if v2.market.enabled else pd.DataFrame()

    per_role: dict[str, list[dict[str, Any]]] = {}
    market_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    pending = 0
    review = 0
    for rec, man in forecasts:
        outcome = ledger.latest_outcome(rec.game_id, as_of=as_of)
        row: dict[str, Any] = {
            "forecast_id": rec.forecast_id,
            "game_id": rec.game_id,
            "season": rec.season,
            "week": rec.week,
            "role": rec.role,
            "model_id": rec.model_id,
            "eligible": man.eligible_for_scoring,
            "eligibility_reason": man.eligibility_reason,
            "actual_horizon_hours": man.actual_horizon_hours,
            "outcome_status": None if outcome is None else outcome.status,
        }
        d = decisions.get(rec.forecast_id)
        settled = outcome is not None and outcome.status == "final"
        if outcome is not None and outcome.status == "rescheduled_review_required":
            review += 1
        if not settled:
            pending += int(outcome is None or outcome.status == "pending")
        if settled and man.eligible_for_scoring:
            row.update(_model_scores(rec, outcome))
            if d is not None and d.status in ("bet", "no_bet"):
                _append_market_rows(market_rows, rec, d, outcome)
        if d is not None:
            settled_row = settle_paper(
                d.model_dump(), None if outcome is None else outcome.model_dump()
            )
            clv = _clv(d, quotes, rec, v2) if d.status == "bet" else {}
            paper_rows.append(
                {
                    **row,
                    **settled_row,
                    "decision_status": d.status,
                    "decision_reason": d.reason,
                    "kickoff_utc": rec.kickoff_at_forecast_utc,
                    **clv,
                }
            )
        per_role.setdefault(rec.role, []).append(row)

    metrics: dict[str, Any] = {}
    frames = {role: pd.DataFrame(rows) for role, rows in per_role.items()}
    for role, df in frames.items():
        scored = (
            df[df.get("score_mae", pd.Series(dtype=float)).notna()]
            if "score_mae" in df
            else df.iloc[0:0]
        )
        metrics[role] = {
            "committed": len(df),
            "eligible": int(df["eligible"].sum()) if len(df) else 0,
            "settled_eligible": len(scored),
            **(
                {
                    k: float(scored[k].mean())
                    for k in (
                        "score_mae",
                        "margin_mae",
                        "total_mae",
                        "logloss_3way",
                        "crps_margin",
                        "crps_total",
                    )
                }
                if len(scored)
                else {}
            ),
            "observed_ties": int(scored["is_tie"].sum()) if len(scored) else 0,
            "predicted_tie_rate": float(scored["p_tie"].mean()) if len(scored) else None,
        }
    paired = None
    if "champion" in frames and "challenger" in frames:
        a = frames["champion"]
        b = frames["challenger"]
        if "crps_margin" in a and "crps_margin" in b:
            common = a.merge(b, on="game_id", suffixes=("_c", "_x"))
            common = common[common["crps_margin_c"].notna() & common["crps_margin_x"].notna()]
            paired = {
                "n_games": len(common),
                **(
                    {
                        f"challenger_minus_champion_{k}": float(
                            (common[f"{k}_x"] - common[f"{k}_c"]).mean()
                        )
                        for k in ("crps_margin", "crps_total", "logloss_3way")
                    }
                    if len(common)
                    else {}
                ),
            }
    market_table = pd.DataFrame(market_rows)
    market_scores = score_aligned(
        market_table,
        replicates=v2.evaluation.bootstrap_replicates,
        seed=v2.evaluation.seed,
        min_blocks=v2.evaluation.min_blocks,
        eps=v2.evaluation.log_loss_epsilon,
    )
    paper_df = pd.DataFrame(paper_rows)
    paper: dict[str, Any] = {}
    if len(paper_df):
        for role, part in paper_df.groupby("role"):
            summary = paper_summary(
                part,
                replicates=v2.evaluation.bootstrap_replicates,
                seed=v2.evaluation.seed,
                min_blocks=v2.evaluation.min_blocks,
            )
            for kind in ("line_clv_spread", "line_clv_total", "price_clv"):
                vals = part[kind].dropna() if kind in part else pd.Series(dtype=float)
                summary[kind] = {
                    "n": len(vals),
                    "mean": float(vals.mean()) if len(vals) else None,
                }
            paper[str(role)] = summary
    counts = summarize_ledger(ledger, as_of=as_of)
    coverage = {
        "scheduled_in_scope_after_activation": n_scheduled,
        "forecast_attempts": sum(
            1 for e in events if e.get("event") == "tick" and e.get("due_games")
        ),
        "committed_forecasts": counts["committed"],
        "eligible_forecasts": counts["eligible"],
        "ineligible_forecasts": counts["ineligible"],
        "missed_forecast_window": counts["missed_forecast_window"],
        "collection_failures": counts["collection_failures"],
        "matched_markets": len(market_table),
        "settled_matched": int((market_table["outcome"] != "refund").sum())
        if len(market_table)
        else 0,
        "pending": pending,
        "rescheduled_review_required": review,
        "integrity_excluded": len(excluded),
    }
    eligible_settled = sum(m.get("settled_eligible", 0) for m in metrics.values())
    blocks = len(
        {
            (r["season"], r["week"])
            for rows in per_role.values()
            for r in rows
            if r.get("score_mae") is not None
        }
    )
    gate = {
        "planned_review_games": v2.evaluation.planned_review_games,
        "planned_review_blocks": v2.evaluation.planned_review_blocks,
        "settled_eligible_games": eligible_settled,
        "season_week_blocks": blocks,
        "status": "review_due"
        if eligible_settled >= v2.evaluation.planned_review_games
        and blocks >= v2.evaluation.planned_review_blocks
        else "pending_evidence",
    }
    report = {
        "protocol_id": epoch.protocol_id,
        "label": epoch.label,
        "synthetic": epoch.synthetic,
        "evidence_mode": epoch.evidence_mode,
        "results_as_of_utc": iso(as_of),
        "generated_at_utc": iso(now),
        "activated_at_utc": epoch.activated_at_utc,
        "champion": epoch.champion.model_id,
        "challenger": None if epoch.challenger is None else epoch.challenger.model_id,
        "coverage": coverage,
        "model_metrics": metrics,
        "paired_champion_challenger": paired,
        "market_benchmark": market_scores,
        "paper": paper,
        "review_gate": gate,
        "integrity": integrity["status"],
        "model_decision": "evidence_insufficient"
        if gate["status"] != "review_due"
        else "review_required",
        "note": "prospective local recording; sample sizes shown; no ROI or edge claim",
    }
    out_dir = ledger.root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = iso(as_of).replace(":", "")
    json_path = out_dir / f"prospective_{stamp}.json"
    write_json(json_path, report)
    md_path = out_dir / f"prospective_{stamp}.md"
    md_path.write_text(_markdown(report))
    if len(paper_df):
        paper_df.to_csv(out_dir / f"paper_ledger_{stamp}.csv", index=False)
    report["report_path"] = str(md_path)
    report["json_path"] = str(json_path)
    return report


def _append_market_rows(rows: list[dict[str, Any]], rec: Any, d: Any, outcome: Any) -> None:
    hs, as_ = int(outcome.home_score), int(outcome.away_score)
    margin, total = hs - as_, hs + as_
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for q in d.market_probabilities:
        if "p_win" not in q:
            continue
        by_pair.setdefault(str(q.get("pair_id")), []).append(q)
    for pair in by_pair.values():
        if len(pair) != 2:
            continue
        side = next((q for q in pair if q["selection"] in ("home", "over")), None)
        other = next((q for q in pair if q is not side), None)
        if side is None or other is None:
            continue
        q_side, _q_other, _ = no_vig_pair(float(side["decimal_odds"]), float(other["decimal_odds"]))
        denom = 1.0 - float(side["p_push"])
        model_p = float(side["p_win"]) / denom if denom > 0 else None
        if model_p is None:
            continue
        market = side["market"]
        if market == "moneyline":
            outcome_v: Any = "refund" if margin == 0 else float(margin > 0)
        elif market == "spread":
            adj = margin + float(side["line"])
            outcome_v = "refund" if adj == 0 else float(adj > 0)
        else:
            diff = total - float(side["line"])
            outcome_v = "refund" if diff == 0 else float(diff > 0)
        rows.append(
            {
                "game_id": rec.game_id,
                "season": rec.season,
                "week": rec.week,
                "model_id": rec.model_id,
                "role": rec.role,
                "market": market,
                "line": side.get("line"),
                "model_p": model_p,
                "market_q": q_side,
                "outcome": outcome_v,
            }
        )


def _clv(d: Any, quotes: pd.DataFrame, rec: Any, v2: Any) -> dict[str, Any]:
    sel = d.selection
    if sel is None or quotes.empty or v2.market.bookmaker is None:
        return {}
    close = closing_proxy(
        quotes,
        pd.Timestamp(rec.kickoff_at_forecast_utc),
        QuotePolicy(v2.market.bookmaker, v2.market.max_quote_age_minutes),
        game_id=rec.game_id,
        window_minutes=v2.closing_proxy.window_minutes,
        require_local_observation=True,
    )
    out: dict[str, Any] = {"line_clv_spread": None, "line_clv_total": None, "price_clv": None}
    cm = close[close["market"] == sel["market"]] if len(close) else close
    if len(cm) != 2:
        return out
    same = cm[cm["selection"] == sel["selection"]].iloc[0]
    other = cm[cm["selection"] != sel["selection"]].iloc[0]
    spec = MarketSpec.from_line(
        sel["market"],
        sel["selection"],
        sel.get("line"),
        sel.get("settlement_rule", "two_way_tie_void"),
    )
    if sel["market"] in ("spread", "total"):
        value = line_clv(spec, float(sel["line"]), float(same["line"]))
        out["line_clv_spread" if sel["market"] == "spread" else "line_clv_total"] = value
        same_line = abs(float(same["line"]) - float(sel["line"])) < 1e-9
    else:
        same_line = True
    if same_line and same["settlement_rule"] == spec.settlement_rule:
        q_sel, _, _ = no_vig_pair(float(same["decimal_odds"]), float(other["decimal_odds"]))
        out["price_clv"] = price_clv(float(sel["decimal_odds"]), q_sel)
    return out


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Prospective report — epoch `{report['protocol_id']}`", ""]
    badge = "SYNTHETIC" if report["synthetic"] else "PROSPECTIVE LOCAL RECORDING"
    lines.append(
        f"**Evidence: {badge}** · results as of {report['results_as_of_utc']} · "
        f"integrity `{report['integrity']}`"
    )
    lines.append("")
    lines.append("## Coverage")
    lines += [f"- {k}: {v}" for k, v in report["coverage"].items()]
    lines.append("")
    lines.append("## Model metrics (eligible, settled)")
    lines.append("```json\n" + json.dumps(report["model_metrics"], indent=1) + "\n```")
    lines.append("## Market benchmark (matched games, one orientation, refunds excluded)")
    lines.append(
        "```json\n" + json.dumps(report["market_benchmark"], indent=1, default=str)[:6000] + "\n```"
    )
    lines.append("## Paper ledger (frozen policy; not realized profit)")
    lines.append("```json\n" + json.dumps(report["paper"], indent=1, default=str)[:6000] + "\n```")
    lines.append("## Review gate")
    lines.append("```json\n" + json.dumps(report["review_gate"], indent=1) + "\n```")
    lines.append(f"Model decision: **{report['model_decision']}**. {report['note']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Research: nested chronological calibration and candidate selection (V2 R5)
# ---------------------------------------------------------------------------------------------


def _base_distributions(
    config: ExperimentConfig, dataset: Any, spec: Any, seasons: list[int]
) -> dict[int, dict[str, Any]]:
    """Per season: residual params and per-game base distributions (chronological, OOF)."""
    from nfl_origination.experiment import chronological_fits, fit_residuals_for_season
    from nfl_origination.models.distribution import predict_distribution

    fits = chronological_fits(dataset, spec, seasons, config.model.train_start_season)
    out: dict[int, dict[str, Any]] = {}
    for season in seasons:
        if season not in fits:
            continue
        try:
            residual, _ids = fit_residuals_for_season(config, fits, season)
        except Exception as exc:
            out[season] = {"status": f"insufficient_residual_history: {exc}"}
            continue
        games = []
        for loc in fits[season].locations.itertuples(index=False):
            location = np.array([loc.mu_home_score, loc.mu_away_score])
            dist = predict_distribution(location, residual, config.distribution)
            games.append(
                {
                    "game_id": loc.game_id,
                    "location": location,
                    "dist": dist,
                    "actual": (int(loc.actual_home), int(loc.actual_away)),
                }
            )
        out[season] = {
            "status": "ok",
            "residual": residual,
            "residual_ids_hash": residual.game_ids_hash,
            "residual_seasons": residual.seasons,
            "training_seasons": sorted(fits[season].train["season"].unique().tolist()),
            "fit_time_utc": iso(fits[season].fit_time_utc),
            "games": games,
        }
    return out


def calibration_pool(
    base_by_season: dict[int, dict[str, Any]],
    target_season: int,
    *,
    max_seasons: int,
    min_games: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Calibration examples from up to ``max_seasons`` latest seasons before the target.

    Each example's base distribution was built from a score model fit before its own season
    and residuals from earlier seasons (nested provenance recorded); its outcome never entered
    its own base distribution.
    """
    from nfl_origination.models.key_number import calibration_example

    seasons = [
        c
        for c in range(target_season - max_seasons, target_season)
        if base_by_season.get(c, {}).get("status") == "ok"
    ]
    examples = []
    provenance = {}
    for c in seasons:
        entry = base_by_season[c]
        for g in entry["games"]:
            examples.append(calibration_example(g["game_id"], c, g["dist"], *g["actual"]))
        provenance[str(c)] = {
            "training_seasons": entry["training_seasons"],
            "residual_seasons": entry["residual_seasons"],
            "residual_ids_hash": entry["residual_ids_hash"],
            "n_games": len(entry["games"]),
        }
    if len(examples) < min_games:
        raise MissingDataError(
            f"season {target_season}: {len(examples)} calibration games < {min_games} "
            f"(seasons {seasons})"
        )
    ids_hash = hash_json(sorted(e.game_id for e in examples))
    return examples, {
        "seasons": seasons,
        "n_games": len(examples),
        "game_ids_hash": ids_hash,
        "nested": provenance,
    }


def _score_candidate(
    config: ExperimentConfig,
    dataset: Any,
    run_id: str,
    model_id: str,
    entry: dict[str, Any],
    theta: np.ndarray | None,
) -> pd.DataFrame:
    from nfl_origination.evaluation.metrics import per_game_scores
    from nfl_origination.experiment import KEY_MARGINS, _finalize_predictions, prediction_row
    from nfl_origination.models.key_number import adjusted_distribution, base_from_location

    games = dataset.games.set_index("game_id")
    policy = dataset.policy
    rows = []
    for g in entry["games"]:
        base = g["dist"]
        dist = base
        if theta is not None:
            residual = entry["residual"]
            dist = adjusted_distribution(
                base,
                theta,
                config.distribution,
                rebuild=lambda m, loc=g["location"], r=residual: base_from_location(
                    loc, r, config.distribution, m
                ),
            )
        game = games.loc[g["game_id"]].copy()
        game["game_id"] = g["game_id"]
        cutoff = pd.Timestamp(policy.cutoff_for(pd.Timestamp(game["kickoff_utc"])))
        rows.append(
            prediction_row(
                dist,
                run_id=run_id,
                game=game,
                cutoff=cutoff,
                model_id=model_id,
                data_mode=policy.mode,
                forecast_policy=policy.forecast_policy_label,
                config_hash=config.config_hash(),
                features_hash_value="research",
                quality_flags=[],
                actual=g["actual"],
            )
        )
    df = _finalize_predictions(rows)
    scored = per_game_scores(df)
    assert set(KEY_MARGINS) == {0, 3, -3, 7, -7}
    return scored


def key_frequency_gap(scored: pd.DataFrame) -> float:
    """Mean absolute gap between predicted and observed frequencies over M=0, |M|=3, |M|=7."""
    gaps = []
    for cols in (("p_margin_0",), ("p_margin_3", "p_margin_-3"), ("p_margin_7", "p_margin_-7")):
        pred = sum(scored[c].mean() for c in cols)
        obs = sum(scored[c.replace("p_", "obs_", 1)].mean() for c in cols)
        gaps.append(abs(float(pred) - float(obs)))
    return float(np.mean(gaps))


def research_v2(config: ExperimentConfig, *, clock: Clock | None = None) -> dict[str, Any]:
    """Bounded distribution experiment on cached history; writes a decision record."""
    from nfl_origination.evaluation.bootstrap import block_bootstrap_differences
    from nfl_origination.evaluation.metrics import aggregate
    from nfl_origination.experiment import prepare_dataset
    from nfl_origination.models.key_number import fit_theta, identity_fit
    from nfl_origination.models.ridge import FitSpec
    from nfl_origination.provenance import RunRegistry, git_info, new_run_id

    v2 = config.require_v2()
    ch = v2.challenger
    clock = clock or SystemClock()
    dataset = prepare_dataset(config, offline=True)
    dev = list(ch.development_seasons)
    checks = list(ch.retrospective_check_seasons)
    target_seasons = sorted(set(dev) | set(checks))
    all_seasons = list(range(config.model.residual_first_season, max(target_seasons) + 1))
    v1_spec = FitSpec("ridge_score", "full", float(v2.models.score_alpha), config.seed)
    b0_spec = FitSpec("league_baseline", "full", None, config.seed)
    base_v1 = _base_distributions(config, dataset, v1_spec, all_seasons)
    base_b0 = _base_distributions(config, dataset, b0_spec, all_seasons)
    registry = RunRegistry(v2.storage.artifacts_dir / "research")
    run_id = new_run_id("v2research")
    run_dir = registry.create_run_dir(run_id)
    candidates = ["identity", *[f"lambda{lam:g}" for lam in ch.lambdas]]
    scored: dict[str, list[pd.DataFrame]] = {"V1": [], "B0": [], **{c: [] for c in candidates}}
    fits_log: dict[str, Any] = {}
    exclusions: list[dict[str, Any]] = []
    for s in target_seasons:
        entry = base_v1.get(s)
        if not entry or entry.get("status") != "ok":
            exclusions.append({"season": s, "reason": entry.get("status") if entry else "no_fit"})
            continue
        scored["V1"].append(
            _score_candidate(config, dataset, run_id, "V1_reconstructed", entry, None)
        )
        if base_b0.get(s, {}).get("status") == "ok":
            scored["B0"].append(
                _score_candidate(config, dataset, run_id, "B0_reconstructed", base_b0[s], None)
            )
        try:
            examples, pool = calibration_pool(
                base_v1,
                s,
                max_seasons=ch.max_calibration_seasons,
                min_games=ch.min_calibration_games,
            )
        except MissingDataError as exc:
            exclusions.append({"season": s, "reason": str(exc)})
            fits_log[str(s)] = {"status": "insufficient_calibration_history"}
            continue
        season_fits: dict[str, Any] = {"pool": pool}
        ident = identity_fit(pool["n_games"], pool["seasons"], pool["game_ids_hash"])
        season_fits["identity"] = ident.to_dict()
        scored["identity"].append(
            _score_candidate(config, dataset, run_id, "KN_identity", entry, ident.theta)
        )
        for lam in ch.lambdas:
            name = f"lambda{lam:g}"
            try:
                fit = fit_theta(
                    examples,
                    lam,
                    bound=ch.bound,
                    seasons=pool["seasons"],
                    game_ids_hash=pool["game_ids_hash"],
                    min_games=ch.min_calibration_games,
                )
            except (MissingDataError, ModelValidationError) as exc:
                season_fits[name] = {"status": "failed", "message": str(exc)}
                exclusions.append({"season": s, "candidate": name, "reason": str(exc)})
                continue
            season_fits[name] = fit.to_dict()
            if fit.status == "failed":
                exclusions.append({"season": s, "candidate": name, "reason": fit.message})
                continue
            try:
                scored[name].append(
                    _score_candidate(config, dataset, run_id, f"KN_{name}", entry, fit.theta)
                )
            except ModelValidationError as exc:
                exclusions.append({"season": s, "candidate": name, "reason": f"numerical: {exc}"})
        fits_log[str(s)] = season_fits
    frames = {
        k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in scored.items()
    }

    def pooled(name: str, seasons: list[int]) -> dict[str, Any]:
        df = frames[name]
        if df.empty:
            return {"n_games": 0}
        part = df[df["season"].isin(seasons)]
        agg: dict[str, Any] = dict(aggregate(part)) if len(part) else {"n_games": 0}
        if len(part):
            agg["key_gap"] = key_frequency_gap(part)
            agg["n_seasons"] = int(part["season"].nunique())
        return agg

    dev_metrics = {name: pooled(name, dev) for name in frames}
    check_metrics = {name: pooled(name, checks) for name in frames}
    # selection: pooled margin CRPS; ties within tolerance -> stronger regularization;
    # identity wins ties
    ranked = []
    for name in candidates:
        m = dev_metrics[name]
        if m.get("n_games", 0) and all(
            dev_metrics[name].get("n_games") == dev_metrics["V1"].get("n_games") for _ in [0]
        ):
            ranked.append((name, m["crps_margin"]))
    decision: dict[str, Any] = {
        "rule": "min pooled margin CRPS on development seasons; ties within 1e-6 favour "
        "stronger regularization; identity wins ties",
        "development_seasons": dev,
        "candidates": {},
    }
    selected = None
    if ranked:
        best = min(r[1] for r in ranked)
        within = [n for n, c in ranked if c <= best + ch.crps_tie_tolerance]
        order = ["identity", *[f"lambda{lam:g}" for lam in sorted(ch.lambdas, reverse=True)]]
        selected = next(n for n in order if n in within)
    v1 = dev_metrics["V1"]
    for name in candidates:
        m = dev_metrics[name]
        screen = {}
        if m.get("n_games") and v1.get("n_games"):
            screen = {
                "margin_crps_strictly_lower": m["crps_margin"] < v1["crps_margin"],
                "total_crps_within_1pct": m["crps_total"]
                <= v1["crps_total"] * (1 + ch.max_worse_fraction),
                "logloss_3way_within_1pct": m["logloss_3way"]
                <= v1["logloss_3way"] * (1 + ch.max_worse_fraction),
                "key_gap_smaller": m["key_gap"] < v1["key_gap"],
                "no_candidate_only_exclusions": not any(
                    e.get("candidate") == name for e in exclusions
                ),
            }
        decision["candidates"][name] = {
            "development": m,
            "screening": screen,
            "passes": bool(screen) and all(screen.values()),
        }
    ready = (
        selected is not None
        and selected != "identity"
        and decision["candidates"][selected]["passes"]
    )
    decision["selected_candidate"] = selected
    decision["model_decision"] = "challenger_ready_for_shadow_collection" if ready else "retain_v1"
    decision["selected_lambda"] = (
        None if not selected or selected == "identity" else float(selected.replace("lambda", ""))
    )
    # paired uncertainty vs reconstructed V1 on identical development games (2000 replicates)
    comparisons: dict[str, Any] = {}
    v1_dev = frames["V1"][frames["V1"]["season"].isin(dev)] if len(frames["V1"]) else pd.DataFrame()
    for name in [*candidates, "B0"]:
        df = frames[name]
        if df.empty or v1_dev.empty:
            continue
        part = df[df["season"].isin(dev)]
        if part.empty:
            continue
        comparisons[f"{name}_minus_V1"] = block_bootstrap_differences(
            part,
            v1_dev,
            replicates=v2.evaluation.bootstrap_replicates,
            seed=v2.evaluation.seed,
            columns=[
                "crps_margin",
                "crps_total",
                "logloss_3way",
                "score_mae",
                "margin_mae",
                "total_mae",
            ],
        )
    checks_report = {name: check_metrics[name] for name in frames}
    git = git_info()
    metrics = {
        "run_id": run_id,
        "kind": "v2_research",
        "evidence": "retrospective_reconstruction (reused research data; 2024-2025 previously "
        "inspected)",
        "development_seasons": dev,
        "retrospective_check_seasons": checks,
        "candidates": candidates,
        "development_metrics": dev_metrics,
        "retrospective_check_metrics": checks_report,
        "paired_vs_V1_development": comparisons,
        "fits": fits_log,
        "exclusions": exclusions,
        "decision": decision,
        "git": git,
        "generated_at_utc": iso(clock.now()),
        "config_hash": config.config_hash(),
        "source_hashes": dataset.source_hashes,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "decision_record.json", decision)
    for name, df in frames.items():
        if len(df):
            df.to_parquet(run_dir / f"scored_{name}.parquet", index=False)
    (run_dir / "report.md").write_text(_research_markdown(metrics))
    from nfl_origination.provenance import RunManifest, dependency_versions, environment_note
    from nfl_origination.schemas import FEATURE_VERSION, SCHEMA_VERSION

    manifest = RunManifest(
        run_id=run_id,
        run_kind="v2_research",
        label=config.run.label,
        created_at_utc=iso(clock.now()),
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
        model_id=f"KN_{selected}" if selected else None,
        training_cutoff_utc=None,
        forecast_policy=dataset.policy.forecast_policy_label,
        data_mode=config.data.mode,
        evaluation_mode="v2_research",
        output_hashes={},
        notes=[decision["model_decision"]],
    )
    write_json(run_dir / "manifest.json", manifest.model_dump())
    registry.record(manifest)
    metrics["run_dir"] = str(run_dir)
    return metrics


def _research_markdown(m: dict[str, Any]) -> str:
    lines = [
        f"# V2 distribution research — `{m['run_id']}`",
        "",
        f"Evidence: **{m['evidence']}**",
        "",
    ]
    lines.append("## Development metrics (pooled, identical games)")
    keys = [
        "n_games",
        "score_mae",
        "margin_mae",
        "total_mae",
        "crps_margin",
        "crps_total",
        "logloss_3way",
        "brier_3way",
        "margin_cover_50",
        "margin_cover_80",
        "margin_cover_95",
        "total_cover_80",
        "predicted_tie_rate",
        "observed_tie_rate",
        "n_ties",
        "pred_margin_0",
        "obs_margin_0",
        "pred_margin_3",
        "obs_margin_3",
        "pred_margin_7",
        "obs_margin_7",
        "pred_total_even",
        "obs_total_even",
        "key_gap",
    ]
    names = list(m["development_metrics"])
    lines.append("| metric | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for k in keys:
        lines.append(
            f"| {k} | "
            + " | ".join(
                str(round(m["development_metrics"][n].get(k), 4))
                if isinstance(m["development_metrics"][n].get(k), int | float)
                else "n/a"
                for n in names
            )
            + " |"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("```json\n" + json.dumps(m["decision"], indent=1, default=str)[:8000] + "\n```")
    lines.append("## Retrospective checks (previously inspected seasons; not for tuning)")
    lines.append(
        "```json\n"
        + json.dumps(
            {
                k: {
                    kk: v.get(kk)
                    for kk in ("n_games", "crps_margin", "crps_total", "logloss_3way", "key_gap")
                }
                for k, v in m["retrospective_check_metrics"].items()
            },
            indent=1,
        )
        + "\n```"
    )
    lines.append("## Paired differences vs reconstructed V1 (development, block bootstrap)")
    lines.append(
        "```json\n"
        + json.dumps(m["paired_vs_V1_development"], indent=1, default=str)[:8000]
        + "\n```"
    )
    lines.append(
        "## Exclusions\n```json\n" + json.dumps(m["exclusions"], indent=1, default=str) + "\n```"
    )
    return "\n".join(lines) + "\n"
