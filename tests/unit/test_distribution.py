"""Joint PMF invariants: finite, nonnegative, normalized, stable, coherent, well-supported."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from nfl_origination.config import DistributionConfig
from nfl_origination.errors import ModelValidationError
from nfl_origination.models.distribution import (
    ResidualParams,
    check_covariance,
    discrete_crps,
    fit_residual_params,
    score_distribution_from_normal,
)
from tests.conftest import point_mass


def test_pmf_basic_invariants(normal_dist):
    pmf = normal_dist.pmf
    assert np.isfinite(pmf).all()
    assert (pmf >= 0).all()
    assert abs(pmf.sum() - 1.0) <= 1e-8
    assert abs(normal_dist.margin_pmf.sum() - 1.0) <= 1e-8
    assert abs(normal_dist.total_pmf.sum() - 1.0) <= 1e-8
    assert normal_dist.omitted_upper_mass < 1e-8


def test_integration_matches_scipy_rectangles(dist_cfg):
    mu = np.array([24.3, 20.1])
    sigma = np.array([[110.0, 15.0], [15.0, 95.0]])
    d = score_distribution_from_normal(mu, sigma, dist_cfg)
    mvn = multivariate_normal(mu, sigma)

    def rect(h0, h1, a0, a1):
        return mvn.cdf([h1, a1]) - mvn.cdf([h0, a1]) - mvn.cdf([h1, a0]) + mvn.cdf([h0, a0])

    for h, a in [(24, 20), (30, 10), (0, 0), (3, 45), (50, 50), (0, 17)]:
        ref = rect(h - 0.5 if h > 0 else -1e6, h + 0.5, a - 0.5 if a > 0 else -1e6, a + 0.5)
        assert d.pmf[h, a] == pytest.approx(ref, abs=1e-9)
    neg = mvn.cdf([-0.5, 1e6]) + mvn.cdf([1e6, -0.5]) - mvn.cdf([-0.5, -0.5])
    assert d.negative_latent_mass == pytest.approx(neg, abs=1e-9)


def test_repeatable_and_deterministic(dist_cfg):
    mu = np.array([21.0, 27.5])
    sigma = np.array([[90.0, 4.0], [4.0, 88.0]])
    a = score_distribution_from_normal(mu, sigma, dist_cfg)
    b = score_distribution_from_normal(mu, sigma, dist_cfg)
    assert np.array_equal(a.pmf, b.pmf)


def test_support_expands_for_high_means():
    cfg = DistributionConfig()
    d = score_distribution_from_normal(
        np.array([90.0, 20.0]), np.array([[100.0, 0.0], [0.0, 100.0]]), cfg
    )
    assert d.max_score > 100
    assert d.omitted_upper_mass < cfg.max_omitted_mass
    with pytest.raises(ModelValidationError):
        score_distribution_from_normal(
            np.array([400.0, 20.0]), np.array([[100.0, 0.0], [0.0, 100.0]]), cfg
        )


def test_negative_latent_warning_flag():
    cfg = DistributionConfig()
    d = score_distribution_from_normal(
        np.array([3.0, 3.0]), np.array([[100.0, 0.0], [0.0, 100.0]]), cfg
    )
    assert d.negative_latent_mass > 0.01
    assert any("negative_latent_mass" in w for w in d.warnings)


def test_invalid_covariance_rejected():
    with pytest.raises(ModelValidationError):
        check_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ModelValidationError):
        check_covariance(np.array([[1.0, 0.5], [0.4, 1.0]]))


def test_parity_coherence_and_means(normal_dist):
    d = normal_dist
    # margin parity equals total parity because H+A and H-A share parity
    even_margin = d.margin_pmf[d.margins % 2 == 0].sum()
    even_total = d.total_pmf[d.totals % 2 == 0].sum()
    assert even_margin == pytest.approx(even_total, abs=1e-12)
    assert d.mean_margin == pytest.approx(d.mean_home - d.mean_away, abs=1e-10)
    assert d.mean_total == pytest.approx(d.mean_home + d.mean_away, abs=1e-10)
    assert d.p_home_win() + d.p_tie() + d.p_away_win() == pytest.approx(1.0, abs=1e-12)


def test_intervals_are_equal_tail_quantiles(normal_dist):
    lo, hi = normal_dist.margin_interval(0.8)
    cdf = np.cumsum(normal_dist.margin_pmf)
    margins = normal_dist.margins
    assert margins[np.searchsorted(cdf, 0.1)] == lo
    assert margins[np.searchsorted(cdf, 0.9)] == hi
    lo50, hi50 = normal_dist.margin_interval(0.5)
    assert lo <= lo50 <= hi50 <= hi


def test_point_mass_intervals_and_probabilities():
    d = point_mass(24, 21)
    assert d.margin_interval(0.95) == (3, 3)
    assert d.total_interval(0.5) == (45, 45)
    assert d.p_home_win() == 1.0 and d.p_tie() == 0.0


def test_discrete_crps_known_values():
    support = np.arange(-2, 3)
    pmf = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    assert discrete_crps(pmf, support, 0) == 0.0
    assert (
        discrete_crps(pmf, support, 2) == 2.0
    )  # F=1 for k=0,1 while indicator 0 -> two unit penalties
    assert point_mass(24, 21).crps_margin(3) == 0.0


def test_residual_fit_requirements():
    rng = np.random.default_rng(0)
    res = rng.normal(0, 10, size=(600, 2))
    params = fit_residual_params(
        res,
        seasons=[2016, 2017],
        game_ids_hash="x",
        min_games=500,
        diagonal_shrinkage=0.1,
        floor=1e-6,
    )
    assert params.n_games == 600
    s = np.cov(res, rowvar=False)
    expected = 0.9 * s + 0.1 * np.diag(np.diag(s))
    assert np.allclose(params.covariance, expected)
    assert np.allclose(params.mean, res.mean(axis=0))
    with pytest.raises(ModelValidationError):
        fit_residual_params(
            res[:499],
            seasons=[2016],
            game_ids_hash="x",
            min_games=500,
            diagonal_shrinkage=0.1,
            floor=1e-6,
        )
    rt = ResidualParams.from_dict(params.to_dict())
    assert np.allclose(rt.covariance, params.covariance)


def test_fast_tail_probabilities_match_masks(normal_dist):
    d = normal_dist
    for hp in (-7, -6, -1, 0, 1, 6, 7, 13, 100):
        assert d.p_margin_gt(hp) == pytest.approx(d.margin_pmf[d.margins > hp / 2].sum(), abs=1e-12)
        assert d.p_margin_lt(hp) == pytest.approx(d.margin_pmf[d.margins < hp / 2].sum(), abs=1e-12)
    for hp in (0, 1, 88, 89, 90, 300):
        assert d.p_total_gt(hp) == pytest.approx(d.total_pmf[d.totals > hp / 2].sum(), abs=1e-12)
        assert d.p_total_lt(hp) == pytest.approx(d.total_pmf[d.totals < hp / 2].sum(), abs=1e-12)
