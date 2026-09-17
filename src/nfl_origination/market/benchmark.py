"""Aligned market benchmarks, closing value and paper-ledger accounting (V2 R4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.evaluation.bootstrap import block_bootstrap_differences, block_bootstrap_ratio
from nfl_origination.pricing.markets import MarketSpec, settle
from nfl_origination.pricing.odds import no_vig_pair

EPS_DEFAULT = 1e-12


def devig(d1: float, d2: float) -> dict[str, float]:
    """Proportional de-vig: r_i = 1/d_i, q1 = r1/(r1+r2). Inferred, not true, probabilities."""
    q1, q2, over = no_vig_pair(d1, d2)
    return {"q1": q1, "q2": q2, "overround": over, "method": "proportional"}  # type: ignore[dict-item]


def line_clv(spec: MarketSpec, bet_line: float, close_line: float) -> float:
    """Selected-side line improvement, positive favorable: spread bet-close; over close-bet;
    under bet-close."""
    if spec.market == "spread":
        return bet_line - close_line
    return close_line - bet_line if spec.selection == "over" else bet_line - close_line


def price_clv(bet_decimal: float, closing_no_vig_probability: float) -> float:
    """bet_decimal * closing no-vig probability - 1 (same line and rules only)."""
    return bet_decimal * closing_no_vig_probability - 1.0


def binary_scores(
    p: np.ndarray, y: np.ndarray, eps: float = EPS_DEFAULT
) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-observation log loss and Brier; returns (logloss, brier, n_clipped)."""
    clipped = int(((p < eps) | (p > 1 - eps)).sum())
    pc = np.clip(p, eps, 1 - eps)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    br = (p - y) ** 2
    return ll, br, clipped


def aligned_market_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per (game, model, market) with model/market conditional probabilities and outcome.

    Required keys: game_id, season, week, model_id, market, line, model_p (conditional on no
    refund), market_q (no-vig for the same side), outcome in {1, 0, 'refund'}.
    """
    return pd.DataFrame(rows)


def score_aligned(
    table: pd.DataFrame, *, replicates: int, seed: int, min_blocks: int, eps: float = EPS_DEFAULT
) -> dict[str, Any]:
    """Model-minus-market paired scores per market on identical games/lines; one orientation."""
    out: dict[str, Any] = {}
    if table.empty:
        return {"status": "no_matched_markets"}
    for (model_id, market), part in table.groupby(["model_id", "market"]):
        refunds = part[part["outcome"] == "refund"]
        scored = part[part["outcome"] != "refund"].copy()
        y = scored["outcome"].astype(float).to_numpy()
        pm, pk = scored["model_p"].to_numpy(float), scored["market_q"].to_numpy(float)
        ll_m, br_m, clip_m = binary_scores(pm, y, eps)
        ll_k, br_k, clip_k = binary_scores(pk, y, eps)
        scored["model_logloss"], scored["model_brier"] = ll_m, br_m
        scored["market_logloss"], scored["market_brier"] = ll_k, br_k
        entry: dict[str, Any] = {
            "n_scored": len(scored),
            "n_refunds_excluded": len(refunds),
            "unique_games": int(scored["game_id"].nunique()),
            "model_logloss": float(ll_m.mean()) if len(scored) else None,
            "market_logloss": float(ll_k.mean()) if len(scored) else None,
            "model_brier": float(br_m.mean()) if len(scored) else None,
            "market_brier": float(br_k.mean()) if len(scored) else None,
            "clipped_model": clip_m,
            "clipped_market": clip_k,
            "epsilon": eps,
        }
        blocks = scored[["season", "week"]].drop_duplicates()
        entry["n_blocks"] = len(blocks)
        if len(scored) and len(blocks) >= min_blocks:
            a = scored.rename(columns={"model_logloss": "logloss", "model_brier": "brier"})[
                ["game_id", "season", "week", "logloss", "brier"]
            ]
            b = scored.rename(columns={"market_logloss": "logloss", "market_brier": "brier"})[
                ["game_id", "season", "week", "logloss", "brier"]
            ]
            entry["paired_model_minus_market"] = block_bootstrap_differences(
                a, b, replicates=replicates, seed=seed, columns=["logloss", "brier"]
            )
        else:
            entry["paired_model_minus_market"] = "insufficient_blocks"
            if len(scored):
                entry["descriptive_difference"] = {
                    "logloss": float((ll_m - ll_k).mean()),
                    "brier": float((br_m - br_k).mean()),
                }
        out[f"{model_id}:{market}"] = entry
    return out


def settle_paper(decision: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    """Settlement status for one frozen paper selection given the latest outcome version."""
    sel = decision.get("selection")
    if decision.get("status") != "bet" or sel is None:
        return {"status": "no_bet", "profit": 0.0, "stake": 0.0}
    stake = float(sel.get("stake", 1.0))
    if outcome is None or outcome.get("status") == "pending":
        return {"status": "pending", "profit": 0.0, "stake": stake}
    if outcome.get("status") == "rescheduled_review_required":
        return {"status": "review_required", "profit": 0.0, "stake": stake}
    if outcome.get("status") == "canceled" or outcome.get("home_score") is None:
        return {"status": "void", "profit": 0.0, "stake": stake}
    spec = MarketSpec.from_line(
        sel["market"],
        sel["selection"],
        sel.get("line"),
        sel.get("settlement_rule", "two_way_tie_void"),
    )
    res = settle(spec, int(outcome["home_score"]), int(outcome["away_score"]))
    dec = float(sel["decimal_odds"])
    profit = stake * (dec - 1.0) if res == "win" else -stake if res == "loss" else 0.0
    return {"status": res, "profit": profit, "stake": stake}


def paper_summary(
    bets: pd.DataFrame, *, replicates: int, seed: int, min_blocks: int
) -> dict[str, Any]:
    """Fixed-unit profit, settled non-void stake, ROI, drawdown, counts, block-bootstrap ROI."""
    if bets.empty:
        return {"n_decisions": 0, "status": "no_paper_decisions"}
    counts = bets["status"].value_counts().to_dict()
    settled = bets[bets["status"].isin(["win", "loss", "push"])]
    stake = float(settled["stake"].sum())
    profit = float(settled["profit"].sum())
    ordered = settled.sort_values(["kickoff_utc", "game_id"])
    cum = (
        np.concatenate([[0.0], ordered["profit"].cumsum().to_numpy()])
        if len(ordered)
        else np.array([0.0])
    )
    drawdown = float(np.max(np.maximum.accumulate(cum) - cum))
    out: dict[str, Any] = {
        "n_decisions": len(bets),
        "counts": {k: int(v) for k, v in counts.items()},
        "settled_non_void_stake": stake,
        "net_profit_units": profit,
        "roi_on_settled_stake": (profit / stake) if stake else None,
        "max_drawdown_units": drawdown,
        "note": "paper selections frozen before outcomes; theoretical EV and CLV are not "
        "realized profit",
    }
    blocks = settled[["season", "week"]].drop_duplicates() if len(settled) else pd.DataFrame()
    out["n_blocks"] = len(blocks)
    if len(settled) and len(blocks) >= min_blocks:
        block_ids = (settled["season"].astype(str) + "-" + settled["week"].astype(str)).to_numpy()
        out["roi_bootstrap"] = block_bootstrap_ratio(
            settled["profit"].to_numpy(float),
            settled["stake"].to_numpy(float),
            block_ids,
            replicates=replicates,
            seed=seed,
        )
    else:
        out["roi_bootstrap"] = "insufficient_blocks"
    return out
