"""Market definitions, half-point line handling, settlement probabilities and settlement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from nfl_origination.errors import InvalidInputError
from nfl_origination.models.distribution import MarginalScoreDistribution

Market = Literal["moneyline", "spread", "total"]
Selection = Literal["home", "away", "over", "under"]
SettlementOutcome = Literal["win", "loss", "push", "void"]
MARKET_ORDER: dict[str, int] = {"moneyline": 0, "spread": 1, "total": 2}
MONEYLINE_RULES = {"two_way_tie_void"}


def line_to_half_points(line: float) -> int:
    """Convert a line to integer half-point units; quarter/split lines are rejected."""
    if not math.isfinite(line):
        raise InvalidInputError(f"line must be finite, got {line!r}")
    doubled = line * 2.0
    if abs(doubled - round(doubled)) > 1e-9:
        raise InvalidInputError(
            f"line {line!r} is not on the half-point grid (quarter lines rejected)"
        )
    return round(doubled)


def half_points_to_line(half_points: int) -> float:
    return half_points / 2.0


@dataclass(frozen=True)
class MarketSpec:
    market: Market
    selection: Selection
    line_half_points: int | None = None
    settlement_rule: str = "two_way_tie_void"

    def __post_init__(self) -> None:
        if self.market == "moneyline":
            if self.selection not in ("home", "away"):
                raise InvalidInputError("moneyline selection must be home or away")
            if self.line_half_points is not None:
                raise InvalidInputError("moneyline has no line")
            if self.settlement_rule not in MONEYLINE_RULES:
                raise InvalidInputError(
                    f"unsupported moneyline settlement rule {self.settlement_rule!r}"
                )
        elif self.market == "spread":
            if self.selection not in ("home", "away") or self.line_half_points is None:
                raise InvalidInputError("spread requires home/away selection and a line")
        elif self.market == "total":
            if self.selection not in ("over", "under") or self.line_half_points is None:
                raise InvalidInputError("total requires over/under selection and a line")
        else:
            raise InvalidInputError(f"unknown market {self.market!r}")

    @property
    def line(self) -> float | None:
        return None if self.line_half_points is None else half_points_to_line(self.line_half_points)

    @classmethod
    def from_line(
        cls,
        market: str,
        selection: str,
        line: float | None,
        settlement_rule: str = "two_way_tie_void",
    ) -> MarketSpec:
        hp = (
            None
            if line is None or (isinstance(line, float) and math.isnan(line))
            else line_to_half_points(line)
        )
        return cls(market, selection, hp, settlement_rule)  # type: ignore[arg-type]

    def home_handicap_half_points(self) -> int:
        """Handicap added to the home score for a spread selection (away uses the opposite)."""
        assert self.line_half_points is not None
        return self.line_half_points if self.selection == "home" else -self.line_half_points


@dataclass(frozen=True)
class SettlementProbabilities:
    p_win: float
    p_push: float
    p_loss: float

    @property
    def p_win_given_no_push(self) -> float | None:
        denom = 1.0 - self.p_push
        return None if denom <= 0.0 else self.p_win / denom


def settlement_probabilities(
    dist: MarginalScoreDistribution, spec: MarketSpec
) -> SettlementProbabilities:
    """Win/push/loss probabilities of a selection under the joint score distribution."""
    if spec.market == "moneyline":
        p_tie = dist.p_tie()
        win = dist.p_home_win() if spec.selection == "home" else dist.p_away_win()
        return SettlementProbabilities(win, p_tie, 1.0 - win - p_tie)
    assert spec.line_half_points is not None
    if spec.market == "spread":
        s = spec.home_handicap_half_points()  # home wins if 2M + s > 0
        p_home_covers = dist.p_margin_gt(-s)
        p_push = dist.margin_probability(-s // 2) if s % 2 == 0 else 0.0
        p_home_loses = 1.0 - p_home_covers - p_push
        if spec.selection == "home":
            return SettlementProbabilities(p_home_covers, p_push, p_home_loses)
        return SettlementProbabilities(p_home_loses, p_push, p_home_covers)
    line = spec.line_half_points
    p_over = dist.p_total_gt(line)
    p_push = dist.total_probability(line // 2) if line % 2 == 0 else 0.0
    p_under = 1.0 - p_over - p_push
    if spec.selection == "over":
        return SettlementProbabilities(p_over, p_push, p_under)
    return SettlementProbabilities(p_under, p_push, p_over)


def settle(spec: MarketSpec, home_score: int | None, away_score: int | None) -> SettlementOutcome:
    """Settle a selection from final scores; missing scores (canceled game) void."""
    if home_score is None or away_score is None:
        return "void"
    margin2 = 2 * (home_score - away_score)
    total2 = 2 * (home_score + away_score)
    if spec.market == "moneyline":
        if margin2 == 0:
            return "void"  # two-way tie-void rule
        home_won = margin2 > 0
        return "win" if home_won == (spec.selection == "home") else "loss"
    assert spec.line_half_points is not None
    if spec.market == "spread":
        adjusted = margin2 + spec.home_handicap_half_points()
        if adjusted == 0:
            return "push"
        home_covers = adjusted > 0
        return "win" if home_covers == (spec.selection == "home") else "loss"
    diff = total2 - spec.line_half_points
    if diff == 0:
        return "push"
    over_wins = diff > 0
    return "win" if over_wins == (spec.selection == "over") else "loss"


def settlement_profit(outcome: SettlementOutcome, stake: float, decimal: float) -> float:
    if outcome == "win":
        return stake * (decimal - 1.0)
    if outcome == "loss":
        return -stake
    return 0.0
