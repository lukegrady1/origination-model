"""Model-versus-market comparison on saved predictions only (spec sections 11–12)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.market.asof import QuotePolicy, select_quotes
from nfl_origination.models.distribution import MarginalScoreDistribution
from nfl_origination.pricing.markets import MarketSpec, settlement_probabilities
from nfl_origination.pricing.odds import no_vig_pair

UNAVAILABLE = "unavailable: no eligible timestamped odds"


def load_distribution(dists: pd.DataFrame, game_id: str) -> MarginalScoreDistribution | None:
    row = dists[dists["game_id"] == game_id]
    if row.empty:
        return None
    r = row.iloc[0]
    return MarginalScoreDistribution(
        np.asarray(r["margin_pmf"]), np.asarray(r["total_pmf"]), int(r["max_score"])
    )


def compare_predictions(
    predictions: pd.DataFrame,
    distributions: pd.DataFrame | None,
    odds: pd.DataFrame,
    policy: QuotePolicy,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair each saved prediction with contemporaneous quotes selected at its cutoff."""
    rows: list[dict[str, Any]] = []
    n_with_market = 0
    for pred in predictions.itertuples(index=False):
        decision_time = pd.Timestamp(pred.cutoff_utc)
        eligible = select_quotes(odds, decision_time, policy, game_id=str(pred.game_id))
        rec: dict[str, Any] = {
            "game_id": pred.game_id,
            "cutoff_utc": decision_time,
            "model_id": pred.model_id,
            "mean_margin": pred.mean_margin,
            "mean_total": pred.mean_total,
            "fair_home_handicap": pred.fair_home_handicap,
            "fair_total": pred.fair_total,
            "market_available": not eligible.empty,
            "market_snapshot_utc": pd.NaT,
        }
        if eligible.empty:
            rows.append(rec)
            continue
        n_with_market += 1
        q = eligible.quotes
        rec["market_snapshot_utc"] = q["snapshot_at_utc"].max()
        dist = (
            load_distribution(distributions, str(pred.game_id))
            if distributions is not None
            else None
        )
        spread = q[q["market"] == "spread"]
        if len(spread) == 2:
            home = spread[spread["selection"] == "home"].iloc[0]
            away = spread[spread["selection"] == "away"].iloc[0]
            rec["market_home_handicap"] = float(home["line"])
            rec["market_expected_margin"] = -float(home["line"])
            rec["margin_price_reference_error"] = pred.mean_margin - rec["market_expected_margin"]
            qh, _qa, over = no_vig_pair(float(home["decimal_odds"]), float(away["decimal_odds"]))
            rec["market_spread_novig_home"] = qh
            rec["market_spread_overround"] = over
            if dist is not None:
                sp = settlement_probabilities(
                    dist, MarketSpec.from_line("spread", "home", float(home["line"]))
                )
                rec["model_spread_home_given_no_push"] = sp.p_win_given_no_push
        total = q[q["market"] == "total"]
        if len(total) == 2:
            over_q = total[total["selection"] == "over"].iloc[0]
            under_q = total[total["selection"] == "under"].iloc[0]
            rec["market_total"] = float(over_q["line"])
            rec["total_price_reference_error"] = pred.mean_total - float(over_q["line"])
            qo, _qu, over = no_vig_pair(
                float(over_q["decimal_odds"]), float(under_q["decimal_odds"])
            )
            rec["market_total_novig_over"] = qo
            rec["market_total_overround"] = over
            if dist is not None:
                tp = settlement_probabilities(
                    dist, MarketSpec.from_line("total", "over", float(over_q["line"]))
                )
                rec["model_total_over_given_no_push"] = tp.p_win_given_no_push
        ml = q[q["market"] == "moneyline"]
        if len(ml) == 2:
            home = ml[ml["selection"] == "home"].iloc[0]
            away = ml[ml["selection"] == "away"].iloc[0]
            qh, _qa, over = no_vig_pair(float(home["decimal_odds"]), float(away["decimal_odds"]))
            rec["market_moneyline_novig_home"] = qh
            rec["market_moneyline_overround"] = over
            rec["model_home_given_no_tie"] = pred.p_home_win_given_no_tie
            rec["moneyline_probability_gap"] = pred.p_home_win_given_no_tie - qh
        rows.append(rec)
    table = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_predictions": len(predictions),
        "n_with_market": n_with_market,
        "bookmaker": policy.bookmaker,
        "status": "ok" if n_with_market else UNAVAILABLE,
        "note": "price-reference errors compare model conditional means with bookmaker lines, "
        "which are not necessarily conditional means; a gap is not proof of mispricing",
    }
    if n_with_market:
        for col in (
            "margin_price_reference_error",
            "total_price_reference_error",
            "moneyline_probability_gap",
        ):
            if col in table:
                vals = table[col].dropna()
                summary[col] = {
                    "n": len(vals),
                    "mean": float(vals.mean()) if len(vals) else None,
                    "mean_abs": float(vals.abs().mean()) if len(vals) else None,
                }
    return table, summary
