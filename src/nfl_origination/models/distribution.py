"""Joint discrete score distribution from a bivariate normal (spec section 9).

The latent continuous score pair is integrated over half-point cells: score k has interval
[k-0.5, k+0.5) and score zero absorbs (-inf, 0.5). Integration is deterministic Gauss–Legendre
quadrature along the home axis with the exact conditional normal CDF along the away axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm

from nfl_origination.config import DistributionConfig
from nfl_origination.errors import ModelValidationError

INTERVAL_LEVELS = (0.5, 0.8, 0.95)
MIN_LATENT_SD = 1e-3


@dataclass(frozen=True)
class ResidualParams:
    """Residual mean/covariance estimated from prior chronological out-of-fold predictions."""

    mean: np.ndarray  # shape (2,) home, away
    covariance: np.ndarray  # shape (2,2)
    n_games: int
    seasons: list[int]
    game_ids_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "n_games": self.n_games,
            "seasons": self.seasons,
            "game_ids_hash": self.game_ids_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> ResidualParams:
        return cls(
            mean=np.asarray(d["mean"], dtype=float),
            covariance=np.asarray(d["covariance"], dtype=float),
            n_games=int(str(d["n_games"])),
            seasons=[int(s) for s in list(d["seasons"])],  # type: ignore[call-overload]
            game_ids_hash=str(d["game_ids_hash"]),
        )


def fit_residual_params(
    residuals: np.ndarray,
    *,
    seasons: list[int],
    game_ids_hash: str,
    min_games: int,
    diagonal_shrinkage: float,
    floor: float,
) -> ResidualParams:
    """Mean and shrunk sample covariance of (home residual, away residual) pairs."""
    r = np.asarray(residuals, dtype=float)
    if r.ndim != 2 or r.shape[1] != 2:
        raise ModelValidationError("residuals must have shape (n, 2)")
    if len(r) < min_games:
        raise ModelValidationError(
            f"insufficient residual history: {len(r)} games < required {min_games}"
        )
    if not np.isfinite(r).all():
        raise ModelValidationError("residuals contain non-finite values")
    mean = r.mean(axis=0)
    s = np.cov(r, rowvar=False, ddof=1)
    s = 0.5 * (s + s.T)
    sigma = (1.0 - diagonal_shrinkage) * s + diagonal_shrinkage * np.diag(np.diag(s))
    sigma[np.diag_indices(2)] = np.maximum(np.diag(sigma), floor)
    check_covariance(sigma)
    return ResidualParams(mean, sigma, len(r), list(seasons), game_ids_hash)


def check_covariance(sigma: np.ndarray) -> None:
    if sigma.shape != (2, 2) or not np.allclose(sigma, sigma.T):
        raise ModelValidationError("covariance must be a symmetric 2x2 matrix")
    eig = np.linalg.eigvalsh(sigma)
    if (eig <= 0).any():
        raise ModelValidationError(f"covariance is not positive definite: eigenvalues {eig}")


class MarginalScoreDistribution:
    """Margin and total PMFs on integer support; enough to price spreads, totals, moneylines."""

    def __init__(self, margin_pmf: np.ndarray, total_pmf: np.ndarray, max_score: int) -> None:
        self.margin_pmf = np.asarray(margin_pmf, dtype=float)
        self.total_pmf = np.asarray(total_pmf, dtype=float)
        self.max_score = int(max_score)
        if self.margin_pmf.shape != (2 * self.max_score + 1,):
            raise ModelValidationError("margin pmf has the wrong support length")
        if self.total_pmf.shape != (2 * self.max_score + 1,):
            raise ModelValidationError("total pmf has the wrong support length")
        self._margin_cdf = np.cumsum(self.margin_pmf)
        self._total_cdf = np.cumsum(self.total_pmf)

    @property
    def margins(self) -> np.ndarray:
        return np.arange(-self.max_score, self.max_score + 1, dtype=float)

    @property
    def totals(self) -> np.ndarray:
        return np.arange(0, 2 * self.max_score + 1, dtype=float)

    @property
    def mean_margin(self) -> float:
        return float((self.margin_pmf * self.margins).sum())

    @property
    def mean_total(self) -> float:
        return float((self.total_pmf * self.totals).sum())

    def margin_probability(self, value: int) -> float:
        """P(M = value) for an integer margin."""
        idx = value + self.max_score
        if idx < 0 or idx >= len(self.margin_pmf):
            return 0.0
        return float(self.margin_pmf[idx])

    def total_probability(self, value: int) -> float:
        if value < 0 or value >= len(self.total_pmf):
            return 0.0
        return float(self.total_pmf[value])

    @staticmethod
    def _p_le(cdf: np.ndarray, offset: int, half_points: int) -> float:
        """P(X <= half_points/2) for an integer variable with support starting at -offset."""
        # number of integers k <= t is floor(t) - (-offset) + 1
        k = int(np.floor(half_points / 2.0)) + offset
        if k < 0:
            return 0.0
        if k >= len(cdf):
            return float(cdf[-1])
        return float(cdf[k])

    @staticmethod
    def _p_lt(cdf: np.ndarray, offset: int, half_points: int) -> float:
        """P(X < half_points/2)."""
        t = half_points / 2.0
        k = int(np.ceil(t)) - 1 + offset  # largest integer strictly below t
        if k < 0:
            return 0.0
        if k >= len(cdf):
            return float(cdf[-1])
        return float(cdf[k])

    def p_margin_gt(self, half_points: int) -> float:
        """P(2M > half_points)."""
        return 1.0 - self._p_le(self._margin_cdf, self.max_score, half_points)

    def p_margin_lt(self, half_points: int) -> float:
        return self._p_lt(self._margin_cdf, self.max_score, half_points)

    def p_total_gt(self, half_points: int) -> float:
        return 1.0 - self._p_le(self._total_cdf, 0, half_points)

    def p_total_lt(self, half_points: int) -> float:
        return self._p_lt(self._total_cdf, 0, half_points)

    def p_home_win(self) -> float:
        return self.p_margin_gt(0)

    def p_tie(self) -> float:
        return self.margin_probability(0)

    def p_away_win(self) -> float:
        return self.p_margin_lt(0)

    @staticmethod
    def interval(pmf: np.ndarray, support: np.ndarray, level: float) -> tuple[int, int]:
        """Equal-tail interval: first integer with CDF >= (1-level)/2 and >= (1+level)/2."""
        cdf = np.cumsum(pmf)
        lo_idx = min(int(np.searchsorted(cdf, (1 - level) / 2, side="left")), len(support) - 1)
        hi_idx = min(int(np.searchsorted(cdf, (1 + level) / 2, side="left")), len(support) - 1)
        return int(support[lo_idx]), int(support[hi_idx])

    def margin_interval(self, level: float) -> tuple[int, int]:
        return self.interval(self.margin_pmf, self.margins, level)

    def total_interval(self, level: float) -> tuple[int, int]:
        return self.interval(self.total_pmf, self.totals, level)

    def crps_margin(self, actual: int) -> float:
        return discrete_crps(self.margin_pmf, self.margins, actual)

    def crps_total(self, actual: int) -> float:
        return discrete_crps(self.total_pmf, self.totals, actual)

    def validate_marginals(self, sum_tolerance: float = 1e-8) -> None:
        for name, pmf in (("margin", self.margin_pmf), ("total", self.total_pmf)):
            if not np.isfinite(pmf).all() or (pmf < 0).any():
                raise ModelValidationError(f"{name} pmf has negative or non-finite mass")
            if abs(float(pmf.sum()) - 1.0) > sum_tolerance:
                raise ModelValidationError(f"{name} pmf does not sum to one")


class ScoreDistribution(MarginalScoreDistribution):
    """Joint PMF over integer (home, away) scores 0..max_score with derived marginals."""

    def __init__(
        self,
        pmf: np.ndarray,
        max_score: int,
        mu_home: float,
        mu_away: float,
        negative_latent_mass: float = 0.0,
        omitted_upper_mass: float = 0.0,
        clipped_negative_cells: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        self.pmf = np.asarray(pmf, dtype=float)
        n = max_score + 1
        if self.pmf.shape != (n, n):
            raise ModelValidationError(f"pmf must be {n}x{n}, got {self.pmf.shape}")
        self.mu_home = mu_home
        self.mu_away = mu_away
        self.negative_latent_mass = negative_latent_mass
        self.omitted_upper_mass = omitted_upper_mass
        self.clipped_negative_cells = clipped_negative_cells
        self.warnings = list(warnings or [])
        h_idx, a_idx = np.indices(self.pmf.shape)
        flat = self.pmf.ravel()
        margin = np.bincount(
            (h_idx - a_idx + max_score).ravel(), weights=flat, minlength=2 * max_score + 1
        )
        total = np.bincount((h_idx + a_idx).ravel(), weights=flat, minlength=2 * max_score + 1)
        super().__init__(margin, total, max_score)

    def validate(self, sum_tolerance: float = 1e-8) -> None:
        if not np.isfinite(self.pmf).all():
            raise ModelValidationError("pmf contains non-finite values")
        if (self.pmf < 0).any():
            raise ModelValidationError("pmf contains negative mass")
        total = float(self.pmf.sum())
        if abs(total - 1.0) > sum_tolerance:
            raise ModelValidationError(f"pmf sums to {total}, not 1 within {sum_tolerance}")
        self.validate_marginals(sum_tolerance)

    @property
    def scores(self) -> np.ndarray:
        return np.arange(self.max_score + 1, dtype=float)

    @property
    def mean_home(self) -> float:
        return float((self.pmf.sum(axis=1) * self.scores).sum())

    @property
    def mean_away(self) -> float:
        return float((self.pmf.sum(axis=0) * self.scores).sum())

    @classmethod
    def from_pmf(cls, pmf: np.ndarray) -> ScoreDistribution:
        pmf = np.asarray(pmf, dtype=float)
        dist = cls(pmf=pmf, max_score=pmf.shape[0] - 1, mu_home=float("nan"), mu_away=float("nan"))
        dist.validate()
        return dist


def discrete_crps(pmf: np.ndarray, support: np.ndarray, actual: float) -> float:
    """CRPS of an integer-valued PMF: sum over integer support of (F(k) - 1[actual <= k])^2."""
    cdf = np.cumsum(pmf)
    indicator = (support >= actual).astype(float)
    return float(np.sum((cdf - indicator) ** 2))


def _cell_masses(
    mu: np.ndarray, sigma: np.ndarray, max_score: int, nodes: int
) -> tuple[np.ndarray, float]:
    """Integrate the bivariate normal over score cells. Returns (pmf, mass below -0.5 on both)."""
    mu_h, mu_a = float(mu[0]), float(mu[1])
    sh = float(np.sqrt(sigma[0, 0]))
    sa = float(np.sqrt(sigma[1, 1]))
    rho = float(sigma[0, 1] / (sh * sa))
    csd = sa * float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    if sh < MIN_LATENT_SD or csd < MIN_LATENT_SD:
        raise ModelValidationError("latent score standard deviation too small for quadrature")
    lower = min(-0.5, float(np.floor(mu_h - 12.0 * sh)) - 0.5)
    edges = np.arange(lower, max_score + 0.5 + 1e-9, 1.0)
    x, w = leggauss(nodes)
    mids = 0.5 * (edges[:-1] + edges[1:])
    hnodes = mids[:, None] + 0.5 * x[None, :]
    weights = 0.5 * w[None, :] * norm.pdf((hnodes - mu_h) / sh) / sh
    cond_mean = mu_a + rho * sa / sh * (hnodes - mu_h)
    a_upper = np.arange(0.5, max_score + 0.5 + 1e-9, 1.0)
    cdf = norm.cdf((a_upper[None, None, :] - cond_mean[..., None]) / csd)
    cell_a = np.diff(np.concatenate([np.zeros((*cdf.shape[:2], 1)), cdf], axis=-1), axis=-1)
    mass = np.einsum("ij,ijk->ik", weights, cell_a)
    # collapse all h intervals with upper edge <= 0.5 into the zero cell
    n_zero = int(np.sum(edges[1:] <= 0.5 + 1e-9))
    pmf = np.vstack([mass[:n_zero].sum(axis=0, keepdims=True), mass[n_zero:]])
    # joint mass with both latent scores below -0.5 (h < -0.5 and a < -0.5)
    below = norm.cdf((-0.5 - cond_mean) / csd)
    h_below = edges[1:] <= -0.5 + 1e-9
    both_negative = float(np.sum(weights[h_below] * below[h_below]))
    return pmf, both_negative


def score_distribution_from_normal(
    mu: np.ndarray, sigma: np.ndarray, cfg: DistributionConfig
) -> ScoreDistribution:
    """Deterministic joint PMF with adaptive upper support and tail/clipping diagnostics."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    check_covariance(sigma)
    warnings: list[str] = []
    max_score = cfg.initial_max_score
    while True:
        pmf, both_negative = _cell_masses(mu, sigma, max_score, cfg.quadrature_nodes_per_cell)
        omitted = 1.0 - float(pmf.sum())
        if omitted < cfg.max_omitted_mass or max_score >= cfg.hard_max_score:
            break
        max_score = min(max_score + cfg.score_step, cfg.hard_max_score)
    if omitted >= cfg.max_omitted_mass:
        raise ModelValidationError(
            f"omitted upper-tail mass {omitted:.3e} exceeds {cfg.max_omitted_mass} at hard limit"
        )
    neg_h = float(norm.cdf((-0.5 - mu[0]) / np.sqrt(sigma[0, 0])))
    neg_a = float(norm.cdf((-0.5 - mu[1]) / np.sqrt(sigma[1, 1])))
    negative_latent = neg_h + neg_a - both_negative
    if negative_latent > cfg.negative_latent_warn:
        warnings.append(
            f"negative_latent_mass={negative_latent:.4f} exceeds {cfg.negative_latent_warn}"
        )
    clipped = int(((pmf < 0) & (pmf >= -cfg.negative_cell_tolerance)).sum())
    if (pmf < -cfg.negative_cell_tolerance).any():
        raise ModelValidationError("integration produced negative cell mass beyond tolerance")
    pmf = np.clip(pmf, 0.0, None)
    pmf = pmf / pmf.sum()
    dist = ScoreDistribution(
        pmf=pmf,
        max_score=max_score,
        mu_home=float(mu[0]),
        mu_away=float(mu[1]),
        negative_latent_mass=negative_latent,
        omitted_upper_mass=omitted,
        clipped_negative_cells=clipped,
        warnings=warnings,
    )
    dist.validate(cfg.pmf_sum_tolerance)
    return dist


def predict_distribution(
    location: np.ndarray, residual: ResidualParams, cfg: DistributionConfig
) -> ScoreDistribution:
    """Shift the regression location by the residual mean and integrate."""
    mu = np.asarray(location, dtype=float) + residual.mean
    return score_distribution_from_normal(mu, residual.covariance, cfg)
