"""V2 reports: prospective evaluation from the ledger (research selection lives alongside)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import ExperimentConfig
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
from nfl_origination.provenance import write_json

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
