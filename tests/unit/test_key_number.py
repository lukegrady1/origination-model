"""Challenger mechanics: identity invariance, coherence, symmetry, gradients, tails, fitting."""

from __future__ import annotations

import numpy as np
import pytest

from nfl_origination.config import DistributionConfig
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.models.distribution import ResidualParams, score_distribution_from_normal
from nfl_origination.models.key_number import (
    adjust_pmf,
    adjusted_distribution,
    adjusted_omitted_bound,
    calibration_example,
    class_masses,
    feature_grid,
    fit_theta,
    identity_fit,
    objective_and_grad,
)
from nfl_origination.pricing.fair_lines import fair_home_handicap, fair_moneyline, fair_total

CFG = DistributionConfig()


def _base(mu=(24.3, 20.1), cov=((110.0, 15.0), (15.0, 95.0))):
    return score_distribution_from_normal(np.array(mu), np.array(cov), CFG)


def test_identity_reproduces_v1_within_tolerance():
    base = _base()
    adj = adjusted_distribution(base, np.zeros(4), CFG)
    assert np.abs(adj.pmf - base.pmf).max() <= 1e-8
    assert np.abs(adj.margin_pmf - base.margin_pmf).max() <= 1e-8
    assert fair_home_handicap(adj).line == fair_home_handicap(base).line
    assert fair_total(adj).line == fair_total(base).line
    assert abs(fair_moneyline(adj).p_home - fair_moneyline(base).p_home) <= 1e-8
    assert adj.margin_interval(0.8) == base.margin_interval(0.8)


def test_adjusted_pmf_is_coherent():
    base = _base()
    adj = adjusted_distribution(base, np.array([-1.0, 0.8, 0.5, 0.2]), CFG)
    assert np.isfinite(adj.pmf).all() and (adj.pmf >= 0).all()
    assert abs(adj.pmf.sum() - 1) <= 1e-8
    assert abs(adj.p_home_win() + adj.p_tie() + adj.p_away_win() - 1) <= 1e-12
    # margins/totals from the joint agree with pricing and parity; intervals nested
    assert abs(adj.margin_pmf.sum() - 1) <= 1e-8 and abs(adj.total_pmf.sum() - 1) <= 1e-8
    even_m = adj.margin_pmf[adj.margins % 2 == 0].sum()
    even_t = adj.total_pmf[adj.totals % 2 == 0].sum()
    assert abs(even_m - even_t) <= 1e-12
    lo50, hi50 = adj.margin_interval(0.5)
    lo95, hi95 = adj.margin_interval(0.95)
    assert lo95 <= lo50 <= hi50 <= hi95
    # means are derived from the adjusted joint, and may differ from the base
    assert adj.mean_margin != base.mean_margin


def test_tie_and_shared_key_margin_mechanics():
    base = _base()
    fewer_ties = adjusted_distribution(base, np.array([-1.0, 0, 0, 0]), CFG)
    assert fewer_ties.p_tie() < base.p_tie()
    more_threes = adjusted_distribution(base, np.array([0, 1.0, 0, 0]), CFG)
    assert more_threes.margin_probability(3) > base.margin_probability(3)
    assert more_threes.margin_probability(-3) > base.margin_probability(-3)
    r_plus = more_threes.margin_probability(3) / base.margin_probability(3)
    r_minus = more_threes.margin_probability(-3) / base.margin_probability(-3)
    assert abs(np.log(r_plus) - np.log(r_minus)) < 0.05  # shared coefficient, same tilt


def test_home_away_swap_transposes_adjusted_pmf():
    theta = np.array([0.3, -0.4, 0.2, 0.1])
    base = _base()
    swapped = _base(mu=(20.1, 24.3), cov=((95.0, 15.0), (15.0, 110.0)))
    a = adjusted_distribution(base, theta, CFG)
    b = adjusted_distribution(swapped, theta, CFG)
    assert np.abs(a.pmf - b.pmf.T).max() <= 1e-9


def test_class_masses_and_gradient_check():
    base = _base()
    classes, mass = class_masses(base.pmf)
    assert abs(mass.sum() - 1) <= 1e-8 and classes.shape[1] == 4
    f = feature_grid(base.max_score)
    assert f.shape == (4, 101, 101) and f[0].sum() == 101 and f[1].sum() == 2 * 98
    examples = [
        calibration_example("g", 2020, base, 24, 21),
        calibration_example("h", 2020, base, 17, 17),
        calibration_example("i", 2020, base, 27, 20),
    ]
    theta = np.array([0.2, -0.1, 0.3, 0.05])
    _obj, grad = objective_and_grad(theta, examples, 0.1)
    num = np.zeros(4)
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1e-6
        num[i] = (
            objective_and_grad(theta + e, examples, 0.1)[0]
            - objective_and_grad(theta - e, examples, 0.1)[0]
        ) / 2e-6
    assert np.abs(num - grad).max() < 1e-6
    # direct check of the tilted probability of the actual score for one example
    adj, _ = adjust_pmf(base.pmf, theta)
    direct = -np.log(adj[24, 21])
    single = objective_and_grad(theta, examples[:1], 0.0)[0]
    assert abs(direct - single) < 1e-9


def test_fit_is_deterministic_bounded_and_rejects_short_history():
    rng = np.random.default_rng(0)
    base = _base()
    ex = []
    for i in range(300):
        h, a = int(rng.integers(0, 45)), int(rng.integers(0, 45))
        if i % 9 == 0:
            a = h  # inject ties so a tie coefficient is identifiable
        ex.append(calibration_example(f"g{i}", 2020, base, h, a))
    f1 = fit_theta(ex, 0.1, seasons=[2020], game_ids_hash="h", min_games=250)
    f2 = fit_theta(ex, 0.1, seasons=[2020], game_ids_hash="h", min_games=250)
    assert np.array_equal(f1.theta, f2.theta) and f1.status in ("converged", "stopped")
    assert (np.abs(f1.theta) <= 3.0 + 1e-9).all() and f1.grad_norm < 1e-4
    assert f1.theta[0] > 0  # more ties than the base expects -> positive tie coefficient
    strong = fit_theta(ex, 100.0, seasons=[2020], game_ids_hash="h", min_games=250)
    assert np.abs(strong.theta).max() < np.abs(f1.theta).max()
    with pytest.raises(MissingDataError):
        fit_theta(ex[:100], 0.1, seasons=[2020], game_ids_hash="h", min_games=250)
    ident = identity_fit(300, [2020], "h")
    assert ident.identity and np.array_equal(ident.theta, np.zeros(4))
    with pytest.raises(ModelValidationError):
        fit_theta(ex, 0.0, seasons=[2020], game_ids_hash="h", min_games=250)


def test_omitted_mass_bound_and_support_expansion():
    assert adjusted_omitted_bound(1e-10, np.array([3, 3, 3, 3])) == pytest.approx(
        1e-10 * np.exp(12) / (1 - 1e-10)
    )
    high = score_distribution_from_normal(
        np.array([80.0, 20.0]), np.array([[150.0, 0.0], [0.0, 100.0]]), CFG
    )
    theta = np.array([3.0, 3.0, 3.0, 3.0])
    calls = []

    def rebuild(max_score: int):
        calls.append(max_score)
        return score_distribution_from_normal(
            np.array([80.0, 20.0]),
            np.array([[150.0, 0.0], [0.0, 100.0]]),
            CFG.model_copy(update={"initial_max_score": max_score}),
        )

    adj = adjusted_distribution(high, theta, CFG, rebuild=rebuild)
    assert adj.omitted_upper_mass <= CFG.max_omitted_mass
    assert adj.max_score >= high.max_score
    with pytest.raises(ModelValidationError):
        adjusted_distribution(
            high, theta, CFG.model_copy(update={"hard_max_score": high.max_score}), rebuild=rebuild
        )


def test_zero_mass_or_out_of_support_actual_score_is_rejected():
    base = _base()
    with pytest.raises(ModelValidationError):
        calibration_example("g", 2020, base, 101, 3)  # outside the support
    zero = base.pmf.copy()
    zero[24, 21] = 0.0
    from nfl_origination.models.distribution import ScoreDistribution

    with pytest.raises(ModelValidationError):
        calibration_example("g", 2020, ScoreDistribution.from_pmf(zero / zero.sum()), 24, 21)
    r = ResidualParams(np.zeros(2), np.eye(2) * 100, 600, [2019], "h")
    assert r.n_games == 600
