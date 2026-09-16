"""Fair moneyline, handicap, and total derivation from a score distribution (spec section 10)."""

from __future__ import annotations

from dataclasses import dataclass, field

from nfl_origination.errors import ModelValidationError
from nfl_origination.models.distribution import MarginalScoreDistribution
from nfl_origination.pricing.markets import (
    MarketSpec,
    SettlementProbabilities,
    settlement_probabilities,
)
from nfl_origination.pricing.odds import probability_to_american, probability_to_decimal

HANDICAP_GRID_HALF_POINTS = 160  # +/-80 points
TOTAL_GRID_MAX_HALF_POINTS = 400  # 0..200 points


@dataclass(frozen=True)
class FairPrice:
    spec: MarketSpec
    p_win: float
    p_push: float
    p_loss: float
    fair_decimal: float | None  # conditional on no push
    fair_american: float | None
    degenerate: bool = False
    flags: list[str] = field(default_factory=list)


def price_market(dist: MarginalScoreDistribution, spec: MarketSpec) -> FairPrice:
    """Price one selection: win/push/loss and the fair price conditional on no push."""
    probs = settlement_probabilities(dist, spec)
    q = probs.p_win_given_no_push
    flags: list[str] = []
    if q is None:
        flags.append("all_mass_on_push_or_tie: conditional odds undefined")
        return FairPrice(spec, probs.p_win, probs.p_push, probs.p_loss, None, None, True, flags)
    if q in (0.0, 1.0):
        flags.append("degenerate_probability: infinite or unit fair odds")
    return FairPrice(
        spec,
        probs.p_win,
        probs.p_push,
        probs.p_loss,
        probability_to_decimal(q),
        probability_to_american(q),
        q in (0.0, 1.0),
        flags,
    )


@dataclass(frozen=True)
class FairMoneyline:
    p_home: float
    p_tie: float
    p_away: float
    q_home: float | None
    q_away: float | None
    decimal_home: float | None
    decimal_away: float | None
    american_home: float | None
    american_away: float | None
    flags: list[str]


def fair_moneyline(dist: MarginalScoreDistribution) -> FairMoneyline:
    p_home, p_tie, p_away = dist.p_home_win(), dist.p_tie(), dist.p_away_win()
    if abs(p_home + p_tie + p_away - 1.0) > 1e-8:
        raise ModelValidationError("home/tie/away probabilities do not sum to one")
    flags: list[str] = []
    if 1.0 - p_tie <= 0.0:
        flags.append("tie_probability_one: two-way moneyline void, conditional odds undefined")
        return FairMoneyline(p_home, p_tie, p_away, None, None, None, None, None, None, flags)
    q_home = p_home / (1.0 - p_tie)
    q_away = p_away / (1.0 - p_tie)
    if q_home in (0.0, 1.0):
        flags.append("degenerate_moneyline: one side has conditional probability 0 or 1")
    return FairMoneyline(
        p_home,
        p_tie,
        p_away,
        q_home,
        q_away,
        probability_to_decimal(q_home),
        probability_to_decimal(q_away),
        probability_to_american(q_home),
        probability_to_american(q_away),
        flags,
    )


@dataclass(frozen=True)
class FairLine:
    line: float
    half_points: int
    probs: SettlementProbabilities  # for the home (spread) or over (total) side
    imbalance: float


def _best_line(
    candidates: list[int], dist: MarginalScoreDistribution, market: str, anchor: float
) -> FairLine:
    best: FairLine | None = None
    for hp in candidates:
        spec = MarketSpec(market, "home" if market == "spread" else "over", hp)  # type: ignore[arg-type]
        probs = settlement_probabilities(dist, spec)
        q = probs.p_win_given_no_push
        imbalance = 1.0 if q is None else abs(q - 0.5)
        cand = FairLine(hp / 2.0, hp, probs, imbalance)
        if best is None or _better(cand, best, anchor):
            best = cand
    assert best is not None
    return best


def _better(cand: FairLine, best: FairLine, anchor: float) -> bool:
    tol = 1e-12
    if cand.imbalance < best.imbalance - tol:
        return True
    if cand.imbalance > best.imbalance + tol:
        return False
    d_cand, d_best = abs(cand.line - anchor), abs(best.line - anchor)
    if d_cand < d_best - tol:
        return True
    if d_cand > d_best + tol:
        return False
    return cand.line < best.line


def fair_home_handicap(dist: MarginalScoreDistribution) -> FairLine:
    """Half-point handicap minimizing |P(home covers | no push) - 0.5| over -80..80.

    The grid expands to the full PMF-supported margin range if the optimum sits on the edge.
    """
    anchor = -dist.mean_margin
    grid = list(range(-HANDICAP_GRID_HALF_POINTS, HANDICAP_GRID_HALF_POINTS + 1))
    best = _best_line(grid, dist, "spread", anchor)
    if best.half_points in (grid[0], grid[-1]):
        full = list(range(-2 * dist.max_score, 2 * dist.max_score + 1))
        best = _best_line(sorted(set(grid) | set(full)), dist, "spread", anchor)
    return best


def fair_total(dist: MarginalScoreDistribution) -> FairLine:
    anchor = dist.mean_total
    grid = list(range(0, TOTAL_GRID_MAX_HALF_POINTS + 1))
    best = _best_line(grid, dist, "total", anchor)
    if best.half_points == grid[-1]:
        full = list(range(0, 4 * dist.max_score + 1))
        best = _best_line(sorted(set(grid) | set(full)), dist, "total", anchor)
    return best
