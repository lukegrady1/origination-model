"""Exact odds arithmetic (spec section 10). Decimal odds are the internal representation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from nfl_origination.errors import InvalidInputError


def american_to_decimal(american: float) -> float:
    """+A -> 1 + A/100; -A -> 1 + 100/|A|. Zero or |A|<100 is rejected."""
    if not isinstance(american, int | float) or isinstance(american, bool):
        raise InvalidInputError(f"american odds must be numeric, got {american!r}")
    if not math.isfinite(american) or american == 0 or abs(american) < 100:
        raise InvalidInputError(f"invalid american odds {american!r}: magnitude must be >= 100")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise InvalidInputError(f"invalid decimal odds {decimal!r}: must be finite and > 1")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def validate_decimal(decimal: float) -> float:
    if not isinstance(decimal, int | float) or isinstance(decimal, bool):
        raise InvalidInputError(f"decimal odds must be numeric, got {decimal!r}")
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise InvalidInputError(f"invalid decimal odds {decimal!r}: must be finite and > 1")
    return float(decimal)


def probability_to_decimal(q: float) -> float:
    """Fair decimal odds 1/q. q=0 yields inf (documented degenerate output)."""
    if not 0.0 <= q <= 1.0:
        raise InvalidInputError(f"probability {q!r} outside [0, 1]")
    return math.inf if q == 0.0 else 1.0 / q


def probability_to_american(q: float) -> float:
    """-100*q/(1-q) for q>=0.5 else 100*(1-q)/q; q=0.5 -> +100; q in {0,1} -> +/-inf."""
    if not 0.0 <= q <= 1.0:
        raise InvalidInputError(f"probability {q!r} outside [0, 1]")
    if q == 0.5:
        return 100.0
    if q == 1.0:
        return -math.inf
    if q == 0.0:
        return math.inf
    if q > 0.5:
        return -100.0 * q / (1.0 - q)
    return 100.0 * (1.0 - q) / q


def implied_probability(decimal: float) -> float:
    return 1.0 / validate_decimal(decimal)


def no_vig_pair(decimal_a: float, decimal_b: float) -> tuple[float, float, float]:
    """Proportional no-vig probabilities for a matched two-sided quote plus the overround."""
    ra, rb = implied_probability(decimal_a), implied_probability(decimal_b)
    total = ra + rb
    return ra / total, rb / total, total - 1.0


def expected_value(p_win: float, p_push: float, p_loss: float, decimal: float) -> float:
    """EV per unit stake: p_win*(d-1) - p_loss; pushes/voids return stake."""
    if abs(p_win + p_push + p_loss - 1.0) > 1e-9:
        raise InvalidInputError("win/push/loss probabilities must sum to one")
    return p_win * (validate_decimal(decimal) - 1.0) - p_loss


@dataclass(frozen=True)
class IllustrativeQuote:
    """Fixed-overround illustration; not a risk-managed sportsbook price."""

    decimal_a: float
    decimal_b: float
    overround: float
    label: str = "illustrative_quote"


def illustrative_overround_quote(
    q_a: float, q_b: float, overround: float = 0.04
) -> IllustrativeQuote:
    if abs(q_a + q_b - 1.0) > 1e-9:
        raise InvalidInputError("conditional two-way probabilities must sum to one")
    ra, rb = (1.0 + overround) * q_a, (1.0 + overround) * q_b
    if ra >= 1.0 or rb >= 1.0:
        raise InvalidInputError("overround pushes an implied probability to >= 1; quote rejected")
    return IllustrativeQuote(1.0 / ra, 1.0 / rb, overround)
