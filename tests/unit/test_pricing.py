"""Exact pricing fixtures: moneyline, spreads, totals, ties, pushes, conversions, EV, vig."""

from __future__ import annotations

import math

import pytest

from nfl_origination.errors import InvalidInputError
from nfl_origination.pricing.fair_lines import (
    fair_home_handicap,
    fair_moneyline,
    fair_total,
    price_market,
)
from nfl_origination.pricing.markets import MarketSpec, settle, settlement_probabilities
from nfl_origination.pricing.odds import (
    american_to_decimal,
    decimal_to_american,
    expected_value,
    illustrative_overround_quote,
    no_vig_pair,
    probability_to_american,
    probability_to_decimal,
)
from tests.conftest import point_mass


@pytest.fixture(scope="module")
def d2421():
    return point_mass(24, 21)


@pytest.mark.parametrize(
    "market,selection,line,win,push,loss,outcome",
    [
        ("spread", "home", -3.0, 0.0, 1.0, 0.0, "push"),
        ("spread", "home", -3.5, 0.0, 0.0, 1.0, "loss"),
        ("spread", "home", -2.5, 1.0, 0.0, 0.0, "win"),
        ("spread", "away", 3.5, 1.0, 0.0, 0.0, "win"),
        ("spread", "away", 3.0, 0.0, 1.0, 0.0, "push"),
        ("total", "over", 45.0, 0.0, 1.0, 0.0, "push"),
        ("total", "over", 44.5, 1.0, 0.0, 0.0, "win"),
        ("total", "under", 44.5, 0.0, 0.0, 1.0, "loss"),
        ("moneyline", "home", None, 1.0, 0.0, 0.0, "win"),
        ("moneyline", "away", None, 0.0, 0.0, 1.0, "loss"),
    ],
)
def test_deterministic_24_21(d2421, market, selection, line, win, push, loss, outcome):
    spec = MarketSpec.from_line(market, selection, line)
    probs = settlement_probabilities(d2421, spec)
    assert (probs.p_win, probs.p_push, probs.p_loss) == (win, push, loss)
    assert settle(spec, 24, 21) == outcome


def test_deterministic_20_20_tie():
    d = point_mass(20, 20)
    ml = fair_moneyline(d)
    assert ml.p_tie == 1.0 and ml.q_home is None and ml.decimal_home is None
    assert any("tie_probability_one" in f for f in ml.flags)
    spec = MarketSpec.from_line("moneyline", "home", None)
    assert settle(spec, 20, 20) == "void"
    fp = price_market(d, spec)
    assert fp.degenerate and fp.fair_decimal is None


def test_fair_lines_on_point_mass(d2421):
    # An all-push line has an undefined conditional probability and is never "balanced";
    # the neighbours tie on imbalance and distance, so the lower numeric line wins.
    hc = fair_home_handicap(d2421)
    assert hc.line == -3.5 and hc.probs.p_push == 0.0 and hc.probs.p_win == 0.0
    tt = fair_total(d2421)
    assert tt.line == 44.5 and tt.probs.p_win == 1.0


def test_fair_lines_on_normal(normal_dist):
    hc = fair_home_handicap(normal_dist)
    q = hc.probs.p_win_given_no_push
    assert abs(q - 0.5) < 0.02
    tt = fair_total(normal_dist)
    assert abs(tt.probs.p_win_given_no_push - 0.5) < 0.02
    ml = fair_moneyline(normal_dist)
    assert ml.q_home + ml.q_away == pytest.approx(1.0)
    assert ml.decimal_home == pytest.approx(1.0 / ml.q_home)


def test_conversions_round_trip():
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert decimal_to_american(2.5) == 150.0
    assert decimal_to_american(1.5) == -200.0
    for a in (-110, -105, 100, 120, 350, -1000):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a, abs=1e-9)
    assert probability_to_american(0.5) == 100.0
    assert probability_to_american(0.6) == pytest.approx(-150.0)
    assert probability_to_american(0.25) == pytest.approx(300.0)
    assert probability_to_decimal(0.0) == math.inf
    assert probability_to_american(1.0) == -math.inf


@pytest.mark.parametrize("bad", [0, 50, -99, float("nan"), float("inf"), "x", True])
def test_malformed_american_rejected(bad):
    with pytest.raises(InvalidInputError):
        american_to_decimal(bad)


@pytest.mark.parametrize("bad", [1.0, 0.9, 0.0, float("nan")])
def test_malformed_decimal_rejected(bad):
    with pytest.raises(InvalidInputError):
        decimal_to_american(bad)


def test_expected_value_fixtures():
    assert expected_value(0.6, 0.0, 0.4, american_to_decimal(-110)) == pytest.approx(
        0.6 * (100 / 110) - 0.4
    )
    assert expected_value(0.6, 0.0, 0.4, american_to_decimal(-110)) == pytest.approx(0.1454545454)
    assert expected_value(0.5, 0.1, 0.4, 1.8) == pytest.approx(0.0)
    with pytest.raises(InvalidInputError):
        expected_value(0.5, 0.1, 0.5, 1.8)


def test_win_push_loss_sum_and_zero_ev_fair_odds(normal_dist):
    spec = MarketSpec.from_line("spread", "home", -3.0)
    probs = settlement_probabilities(normal_dist, spec)
    assert probs.p_win + probs.p_push + probs.p_loss == pytest.approx(1.0, abs=1e-12)
    fp = price_market(normal_dist, spec)
    assert expected_value(fp.p_win, fp.p_push, fp.p_loss, fp.fair_decimal) == pytest.approx(
        0.0, abs=1e-12
    )
    other = settlement_probabilities(normal_dist, MarketSpec.from_line("spread", "away", 3.0))
    assert other.p_win == pytest.approx(probs.p_loss) and other.p_push == pytest.approx(
        probs.p_push
    )


def test_both_sides_settle_consistently():
    for hs, as_ in ((24, 21), (21, 24), (20, 20), (30, 27), (27, 30)):
        home = settle(MarketSpec.from_line("spread", "home", -3.0), hs, as_)
        away = settle(MarketSpec.from_line("spread", "away", 3.0), hs, as_)
        assert {home, away} in ({"win", "loss"}, {"push"})
        over = settle(MarketSpec.from_line("total", "over", 45.0), hs, as_)
        under = settle(MarketSpec.from_line("total", "under", 45.0), hs, as_)
        assert {over, under} in ({"win", "loss"}, {"push"})
    assert settle(MarketSpec.from_line("spread", "home", -3.0), None, None) == "void"


def test_no_vig_and_overround():
    qa, qb, over = no_vig_pair(1.9091, 1.9091)
    assert qa == pytest.approx(0.5) and qb == pytest.approx(0.5)
    assert over == pytest.approx(2 / 1.9091 - 1)
    quote = illustrative_overround_quote(0.6, 0.4, 0.04)
    assert quote.decimal_a == pytest.approx(1 / (1.04 * 0.6))
    with pytest.raises(InvalidInputError):
        illustrative_overround_quote(0.97, 0.03, 0.04)
    with pytest.raises(InvalidInputError):
        illustrative_overround_quote(0.6, 0.5, 0.04)


def test_quarter_and_split_lines_rejected():
    with pytest.raises(InvalidInputError):
        MarketSpec.from_line("spread", "home", -3.25)
    with pytest.raises(InvalidInputError):
        MarketSpec.from_line("total", "over", 44.75)
    with pytest.raises(InvalidInputError):
        MarketSpec.from_line("moneyline", "home", -3.0)
    with pytest.raises(InvalidInputError):
        MarketSpec.from_line("moneyline", "home", None, settlement_rule="three_way")
