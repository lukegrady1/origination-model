"""Metrics, bootstrap sampling, ROI, and paired comparisons on tiny fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_origination.evaluation.bootstrap import block_bootstrap_differences, block_bootstrap_ratio
from nfl_origination.evaluation.metrics import (
    aggregate,
    paired_difference,
    per_game_scores,
    reliability_table,
)


def _pred(game_id, season, week, ph, pt, pa, mh, ma, actual_h, actual_a):
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "dist_mean_home_score": mh,
        "dist_mean_away_score": ma,
        "mean_margin": mh - ma,
        "mean_total": mh + ma,
        "p_home_win": ph,
        "p_tie": pt,
        "p_away_win": pa,
        "p_home_win_given_no_tie": ph / (1 - pt),
        "actual_home": actual_h,
        "actual_away": actual_a,
        **{f"margin_q{c}_lo": -10 for c in (50, 80, 95)},
        **{f"margin_q{c}_hi": 10 for c in (50, 80, 95)},
        **{f"total_q{c}_lo": 30 for c in (50, 80, 95)},
        **{f"total_q{c}_hi": 60 for c in (50, 80, 95)},
        "p_margin_0": pt,
        "p_margin_3": 0.05,
        "p_margin_-3": 0.05,
        "p_margin_7": 0.04,
        "p_margin_-7": 0.04,
        "p_total_even": 0.5,
    }


def test_log_loss_and_brier_hand_values():
    df = pd.DataFrame(
        [
            _pred("g1", 2020, 1, 0.7, 0.1, 0.2, 24, 20, 27, 20),
            _pred("g2", 2020, 1, 0.7, 0.1, 0.2, 24, 20, 20, 20),
        ]
    )
    s = per_game_scores(df)
    assert s.loc[0, "logloss_3way"] == pytest.approx(-np.log(0.7))
    assert s.loc[1, "logloss_3way"] == pytest.approx(-np.log(0.1))
    assert s.loc[0, "brier_3way"] == pytest.approx((0.3) ** 2 + 0.1**2 + 0.2**2)
    assert s.loc[0, "score_mae"] == pytest.approx(1.5)
    assert np.isnan(s.loc[1, "logloss_binary_no_tie"])
    q = 0.7 / 0.9
    assert s.loc[0, "logloss_binary_no_tie"] == pytest.approx(-np.log(q))
    assert s.loc[0, "margin_cover_80"] == 1.0
    agg = aggregate(s)
    assert agg["n_games"] == 2 and agg["n_ties"] == 1
    assert agg["observed_tie_rate"] == 0.5 and agg["predicted_tie_rate"] == pytest.approx(0.1)
    assert agg["score_rmse"] == pytest.approx(np.sqrt(np.mean([0.5 * (9 + 0), 0.5 * (16 + 0)])))


def test_reliability_bins_and_sparse_flag():
    prob = np.array([0.05, 0.15, 0.95, 0.96, 1.0])
    out = np.array([0, 0, 1, 1, 1])
    rows = reliability_table(prob, out, sparse_threshold=2)
    assert rows[0]["n"] == 1 and rows[0]["sparse"]
    assert rows[9]["n"] == 3 and not rows[9]["sparse"]
    assert rows[9]["observed_rate"] == 1.0


def test_paired_difference_uses_matched_games_only():
    a = pd.DataFrame({"game_id": ["g1", "g2", "g3"], "score_mae": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"game_id": ["g1", "g2"], "score_mae": [2.0, 2.0]})
    d = paired_difference(a, b, "score_mae")
    assert d["n_matched"] == 2 and d["difference"] == pytest.approx(-0.5)


def test_block_bootstrap_is_seeded_and_covers_point_estimate():
    rng = np.random.default_rng(1)
    rows = []
    for season in (2020, 2021):
        for week in range(1, 18):
            for g in range(8):
                rows.append(
                    {
                        "game_id": f"{season}_{week}_{g}",
                        "season": season,
                        "week": week,
                        "score_mae": rng.normal(7, 1),
                        "logloss_3way": rng.normal(0.7, 0.05),
                        "is_tie": 0.0,
                        "p_tie": 0.03,
                    }
                )
    a = pd.DataFrame(rows)
    b = a.copy()
    b["score_mae"] = b["score_mae"] + 0.5
    r1 = block_bootstrap_differences(
        a, b, replicates=300, seed=42, columns=["score_mae", "logloss_3way"]
    )
    r2 = block_bootstrap_differences(
        a, b, replicates=300, seed=42, columns=["score_mae", "logloss_3way"]
    )
    assert r1 == r2
    d = r1["score_mae"]
    assert d["difference"] == pytest.approx(-0.5)
    assert d["ci_low"] <= -0.5 <= d["ci_high"]
    assert d["ci_high"] - d["ci_low"] < 0.05  # constant shift -> tight interval
    assert d["n_blocks"] == 34


def test_roi_bootstrap_ratio():
    profit = np.array([1.0, -1.0, 0.0, 0.91, -1.0, 0.91])
    stake = np.ones(6)
    blocks = np.array(["a", "a", "b", "b", "c", "c"])
    r = block_bootstrap_ratio(profit, stake, blocks, replicates=200, seed=0)
    assert r["point"] == pytest.approx(profit.sum() / 6)
    assert r["ci_low"] <= r["point"] <= r["ci_high"]
