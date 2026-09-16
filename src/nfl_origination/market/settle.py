"""Paper-only settlement and backtest with a frozen flat-stake policy (spec section 12)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.config import MarketConfig
from nfl_origination.evaluation.bootstrap import block_bootstrap_ratio
from nfl_origination.market.asof import QuotePolicy, closing_proxy, select_quotes
from nfl_origination.market.compare import decision_precedes_model, load_distribution
from nfl_origination.pricing.markets import (
    MARKET_ORDER,
    MarketSpec,
    settle,
    settlement_probabilities,
    settlement_profit,
)
from nfl_origination.pricing.odds import expected_value, no_vig_pair
from nfl_origination.schemas import PAPER_BETS_SCHEMA, validate_frame

UNAVAILABLE = "unavailable: no eligible timestamped odds"


def policy_id(cfg: MarketConfig) -> str:
    return (
        f"flat{cfg.stake_units:g}_ev{cfg.min_ev:g}_max{cfg.max_selections_per_game}"
        f"_{cfg.moneyline_settlement_rule}"
    )


def _line_clv(spec: MarketSpec, bet_line: float, close_line: float) -> float:
    """Selected-side line improvement; positive is favorable to the bettor.

    Spread lines are stored relative to the selected side (home −3.5 / away +3.5), so both spread
    selections use ``bet_handicap − close_handicap``. Totals keep separate over/under formulas.
    """
    if spec.market == "spread":
        return bet_line - close_line
    return close_line - bet_line if spec.selection == "over" else bet_line - close_line


def paper_backtest(
    run_id: str,
    predictions: pd.DataFrame,
    distributions: pd.DataFrame,
    odds: pd.DataFrame,
    results: pd.DataFrame,
    cfg: MarketConfig,
    *,
    seed: int,
    bootstrap_replicates: int = 1000,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every eligible quote at each prediction's cutoff; bet by the frozen policy."""
    if cfg.bookmaker is None:
        raise ValueError("market.bookmaker must be configured for a paper backtest")
    qpolicy = QuotePolicy(
        cfg.bookmaker, cfg.max_quote_age_minutes, True, cfg.moneyline_settlement_rule
    )
    scores = results.set_index("game_id")[["home_score", "away_score"]]
    bets: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    n_with_market = 0
    for pred in predictions.sort_values(["kickoff_utc", "game_id"]).itertuples(index=False):
        gid = str(pred.game_id)
        decision_time = pd.Timestamp(pred.cutoff_utc)
        if decision_precedes_model(pred, decision_time):
            exclusions.append(
                {
                    "game_id": gid,
                    "quote_id": None,
                    "reason": "decision_time_before_model_availability",
                }
            )
            continue
        eligible = select_quotes(odds, decision_time, qpolicy, game_id=gid)
        exclusions.extend({"game_id": gid, **e} for e in eligible.exclusions)
        if eligible.empty:
            continue
        dist = load_distribution(distributions, gid, str(pred.model_id))
        if dist is None:
            exclusions.append({"game_id": gid, "quote_id": None, "reason": "no_saved_distribution"})
            continue
        n_with_market += 1
        candidates = []
        for q in eligible.quotes.itertuples(index=False):
            spec = MarketSpec.from_line(
                q.market, q.selection, None if pd.isna(q.line) else float(q.line), q.settlement_rule
            )
            probs = settlement_probabilities(dist, spec)
            ev = expected_value(probs.p_win, probs.p_push, probs.p_loss, float(q.decimal_odds))
            signal = {
                "game_id": gid,
                "quote_id": q.quote_id,
                "market": q.market,
                "selection": q.selection,
                "line": None if pd.isna(q.line) else float(q.line),
                "decimal_odds": float(q.decimal_odds),
                "p_win": probs.p_win,
                "p_push": probs.p_push,
                "p_loss": probs.p_loss,
                "ev": ev,
                "qualifies": ev >= cfg.min_ev,
                "spec": spec,
            }
            signals.append(signal)
            if signal["qualifies"]:
                candidates.append(signal)
            else:
                exclusions.append(
                    {
                        "game_id": gid,
                        "quote_id": q.quote_id,
                        "reason": f"ev_below_threshold({ev:.4f})",
                    }
                )
        if not candidates:
            continue
        candidates.sort(key=lambda s: (-s["ev"], MARKET_ORDER[s["market"]], s["selection"]))
        chosen = candidates[: cfg.max_selections_per_game]
        for s in candidates[cfg.max_selections_per_game :]:
            exclusions.append(
                {"game_id": gid, "quote_id": s["quote_id"], "reason": "max_selections_per_game"}
            )
        for s in chosen:
            spec = s["spec"]
            if gid in scores.index:
                hs, as_ = int(scores.loc[gid, "home_score"]), int(scores.loc[gid, "away_score"])
                outcome = settle(spec, hs, as_)
            else:
                outcome = "void"
            profit = settlement_profit(outcome, cfg.stake_units, s["decimal_odds"])
            rec = {
                "run_id": run_id,
                "quote_id": s["quote_id"],
                "prediction_id": f"{run_id}:{gid}:{pred.model_id}",
                "policy_id": policy_id(cfg),
                "game_id": gid,
                "market": s["market"],
                "selection": s["selection"],
                "line": np.nan if s["line"] is None else s["line"],
                "decimal_odds": s["decimal_odds"],
                "p_win": s["p_win"],
                "p_push": s["p_push"],
                "p_loss": s["p_loss"],
                "ev": s["ev"],
                "stake": float(cfg.stake_units),
                "settlement": outcome,
                "profit": profit,
                "season": int(pred.season),
                "week": int(pred.week),
                "kickoff_utc": pd.Timestamp(pred.kickoff_utc),
                "decision_time_utc": pd.Timestamp(pred.cutoff_utc),
                "line_clv": np.nan,
                "price_clv": np.nan,
                "closing_line": np.nan,
            }
            close = closing_proxy(
                odds,
                pd.Timestamp(pred.kickoff_utc),
                qpolicy,
                game_id=gid,
                window_minutes=cfg.closing_proxy_window_minutes,
            )
            cm = close[close["market"] == s["market"]] if len(close) else close
            if len(cm) == 2:
                same_side = cm[cm["selection"] == s["selection"]].iloc[0]
                other = cm[cm["selection"] != s["selection"]].iloc[0]
                if s["market"] != "moneyline":
                    rec["closing_line"] = float(same_side["line"])
                    rec["line_clv"] = _line_clv(spec, s["line"], float(same_side["line"]))
                    same_line = abs(float(same_side["line"]) - s["line"]) < 1e-9
                else:
                    same_line = True
                if same_line and same_side["settlement_rule"] == spec.settlement_rule:
                    q_sel, _q_other, _ = no_vig_pair(
                        float(same_side["decimal_odds"]), float(other["decimal_odds"])
                    )
                    rec["price_clv"] = s["decimal_odds"] * q_sel - 1.0
            bets.append(rec)
    bets_df = pd.DataFrame(bets)
    summary: dict[str, Any] = {
        "policy_id": policy_id(cfg),
        "bookmaker": cfg.bookmaker,
        "synthetic": bool(cfg.synthetic),
        "n_predictions": len(predictions),
        "n_games_with_market": n_with_market,
        "n_signals": len(signals),
        "n_bets": len(bets_df),
        "exclusions": _count(exclusions),
        "status": "ok" if n_with_market else UNAVAILABLE,
    }
    if bets_df.empty:
        summary["note"] = "no qualifying selections"
        return bets_df, summary
    validate_frame(bets_df, PAPER_BETS_SCHEMA)
    settled = bets_df[bets_df["settlement"] != "void"]
    staked = float(settled["stake"].sum())
    profit = float(bets_df["profit"].sum())
    ordered = bets_df.sort_values(["kickoff_utc", "game_id"])
    cum = ordered["profit"].cumsum().to_numpy()
    drawdown = float(
        np.max(np.maximum.accumulate(np.concatenate([[0.0], cum])) - np.concatenate([[0.0], cum]))
    )
    counts = bets_df["settlement"].value_counts().to_dict()
    summary.update(
        {
            "wins": int(counts.get("win", 0)),
            "losses": int(counts.get("loss", 0)),
            "pushes": int(counts.get("push", 0)),
            "voids": int(counts.get("void", 0)),
            "staked_units_non_void": staked,
            "net_profit_units": profit,
            "roi_on_non_void_stake": profit / staked if staked else None,
            "max_drawdown_units": drawdown,
            "mean_ev": float(bets_df["ev"].mean()),
            "line_clv_mean": float(bets_df["line_clv"].dropna().mean())
            if bets_df["line_clv"].notna().any()
            else None,
            "line_clv_n": int(bets_df["line_clv"].notna().sum()),
            "price_clv_mean": float(bets_df["price_clv"].dropna().mean())
            if bets_df["price_clv"].notna().any()
            else None,
            "price_clv_n": int(bets_df["price_clv"].notna().sum()),
            "games": int(bets_df["game_id"].nunique()),
        }
    )
    if len(settled):
        blocks = (settled["season"].astype(str) + "-" + settled["week"].astype(str)).to_numpy()
        summary["roi_bootstrap"] = block_bootstrap_ratio(
            settled["profit"].to_numpy(float),
            settled["stake"].to_numpy(float),
            blocks,
            replicates=bootstrap_replicates,
            seed=seed,
        )
    if cfg.synthetic:
        summary["warning"] = "SYNTHETIC odds: returns are illustrative mechanics, not evidence"
    return bets_df, summary


def _count(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        r = str(it.get("reason"))
        out[r] = out.get(r, 0) + 1
    return out
