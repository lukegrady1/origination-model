"""Hand-calculated recency weights, opportunity-weighted shrinkage, and priors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_origination.features.rolling import (
    ShrinkageParams,
    league_priors,
    recency_weights,
    rolling_metrics,
    weighted_shrunk_rate,
)
from nfl_origination.schemas import RATE_METRICS


def test_recency_weights_and_offseason_discount():
    w = recency_weights(3, 8.0, np.array([True, True, False]), 0.5)
    assert w[0] == 1.0
    assert w[1] == pytest.approx(2 ** (-1 / 8))
    assert w[2] == pytest.approx(0.5 * 2 ** (-2 / 8))


def test_weighted_shrunk_rate_hand_calculation():
    num = np.array([10.0, 5.0])
    den = np.array([50.0, 40.0])
    w = np.array([1.0, 0.5])
    rate, n, d = weighted_shrunk_rate(num, den, w, k=200.0, r0=0.1)
    assert n == pytest.approx(12.5) and d == pytest.approx(70.0)
    assert rate == pytest.approx((12.5 + 20.0) / 270.0)


def test_null_pairs_contribute_nothing():
    rate, n, d = weighted_shrunk_rate(
        np.array([np.nan, 5.0]), np.array([50.0, 40.0]), np.array([1.0, 1.0]), 4.0, 22.0
    )
    assert n == 5.0 and d == 40.0
    assert rate == pytest.approx((5 + 88) / 44)


def _history(rows):
    cols = {num for num, _d, _k in RATE_METRICS.values()} | {
        den for _n, den, _k in RATE_METRICS.values()
    }
    frame = pd.DataFrame([{c: 0.0 for c in cols} | r for r in rows])
    return frame


def test_rolling_metrics_uses_opportunity_weighting_not_rate_average():
    hist = _history(
        [
            {"season": 2020, "off_epa_sum": 10.0, "off_epa_plays": 100.0, "game_count": 1},
            {"season": 2020, "off_epa_sum": 0.0, "off_epa_plays": 20.0, "game_count": 1},
        ]
    )
    priors = {m: 0.0 for m in RATE_METRICS}
    out = rolling_metrics(
        hist,
        2020,
        history_games=16,
        half_life=8,
        offseason_weight=0.5,
        shrinkage=ShrinkageParams(200, 100, 4),
        priors=priors,
    )
    w1, w0 = 1.0, 2 ** (-1 / 8)  # most recent (row 2) gets weight 1
    expected = (w0 * 10.0 + w1 * 0.0 + 200 * 0.0) / (w0 * 100.0 + w1 * 20.0 + 200)
    assert out["off_epa_per_play"] == pytest.approx(expected)


def test_cold_start_returns_prior():
    priors = {m: 0.123 for m in RATE_METRICS}
    out = rolling_metrics(
        _history([]),
        2020,
        history_games=16,
        half_life=8,
        offseason_weight=0.5,
        shrinkage=ShrinkageParams(),
        priors=priors,
    )
    assert all(v == 0.123 for v in out.values())


def test_history_window_caps_at_16():
    rows = [{"season": 2020, "points_for": float(i), "game_count": 1} for i in range(30)]
    hist = _history(rows)
    priors = {m: 0.0 for m in RATE_METRICS}
    out = rolling_metrics(
        hist,
        2020,
        history_games=16,
        half_life=8,
        offseason_weight=0.5,
        shrinkage=ShrinkageParams(200, 100, 0.0),
        priors=priors,
    )
    w = 2 ** (-np.arange(16) / 8)
    pts = np.arange(29, 13, -1)
    assert out["points_for"] == pytest.approx((w * pts).sum() / w.sum())


def test_league_priors_are_ratio_of_sums():
    hist = _history(
        [
            {
                "season": 2018,
                "off_epa_sum": 10.0,
                "off_epa_plays": 100.0,
                "game_count": 1,
                "points_for": 20,
            },
            {
                "season": 2019,
                "off_epa_sum": -5.0,
                "off_epa_plays": 50.0,
                "game_count": 1,
                "points_for": 30,
            },
            {
                "season": 2020,
                "off_epa_sum": 99.0,
                "off_epa_plays": 1.0,
                "game_count": 1,
                "points_for": 99,
            },
        ]
    )
    pri = league_priors(hist, [2018, 2019])
    assert pri["off_epa_per_play"] == pytest.approx(5.0 / 150.0)
    assert pri["points_for"] == pytest.approx(25.0)
