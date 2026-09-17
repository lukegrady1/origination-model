"""Bounded four-parameter joint-PMF challenger (V2 R5).

P_theta(h,a) = P_V1(h,a) * exp(theta . f(h,a)) / Z with
f = [I(M=0), I(|M|=3), I(|M|=7), I(T even)], M = h-a, T = h+a.

All probabilities, means, marginals, fair lines and intervals derive from the adjusted joint
PMF. The tilt depends on (h,a) only through f, so fitting uses per-game class masses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from nfl_origination.config import DistributionConfig, ExperimentConfig
from nfl_origination.errors import MissingDataError, ModelValidationError
from nfl_origination.models.bundle import ModelBundle, game_locations, predict_location
from nfl_origination.models.distribution import (
    ResidualParams,
    ScoreDistribution,
    score_distribution_from_normal,
)

FEATURE_NAMES = ("tie", "abs_margin_3", "abs_margin_7", "total_even")
FAMILY = "key_number_adjusted"


def feature_grid(max_score: int) -> np.ndarray:
    """Indicator features over the (h, a) grid; shape (4, n, n)."""
    h, a = np.indices((max_score + 1, max_score + 1))
    m = h - a
    t = h + a
    return np.stack([(m == 0), (np.abs(m) == 3), (np.abs(m) == 7), (t % 2 == 0)], axis=0).astype(
        float
    )


def class_masses(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a joint PMF into masses per distinct feature vector: (classes (K,4), mass (K,))."""
    f = feature_grid(pmf.shape[0] - 1).reshape(4, -1).T  # (cells, 4)
    keys = f @ np.array([1, 2, 4, 8])
    uniq, inverse = np.unique(keys, return_inverse=True)
    mass = np.bincount(inverse, weights=pmf.ravel(), minlength=len(uniq))
    classes = np.array([f[np.argmax(keys == k)] for k in uniq])
    return classes, mass


def feature_of(h: int, a: int) -> np.ndarray:
    m, t = h - a, h + a
    return np.array([m == 0, abs(m) == 3, abs(m) == 7, t % 2 == 0], dtype=float)


def adjust_pmf(base: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, float]:
    """Exponentially tilted joint PMF and log normalizer (log-sum-exp)."""
    f = feature_grid(base.shape[0] - 1)
    tilt = np.tensordot(np.asarray(theta, dtype=float), f, axes=1)
    with np.errstate(divide="ignore"):
        logp = np.log(base) + tilt
    log_z = float(logsumexp(logp[np.isfinite(logp)]))
    adjusted = np.exp(logp - log_z)
    adjusted[~np.isfinite(adjusted)] = 0.0
    return adjusted, log_z


def adjusted_omitted_bound(base_omitted: float, theta: np.ndarray) -> float:
    """Conservative omitted-mass bound: omitted * w_max / (w_min * (1 - omitted))."""
    th = np.asarray(theta, dtype=float)
    w_max = float(np.exp(np.clip(th, 0, None).sum()))
    w_min = float(np.exp(np.clip(th, None, 0).sum()))
    return base_omitted * w_max / (w_min * max(1.0 - base_omitted, 1e-300))


@dataclass
class CalibrationExample:
    game_id: str
    season: int
    classes: np.ndarray
    mass: np.ndarray
    f_actual: np.ndarray
    log_base_actual: float


@dataclass
class ThetaFit:
    theta: np.ndarray
    lam: float
    status: str
    message: str
    iterations: int
    objective: float
    grad_norm: float
    bound_hits: list[str]
    n_games: int
    seasons: list[int]
    game_ids_hash: str
    identity: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta": self.theta.tolist(),
            "lambda": self.lam,
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "objective": self.objective,
            "grad_norm": self.grad_norm,
            "bound_hits": self.bound_hits,
            "n_games": self.n_games,
            "seasons": self.seasons,
            "game_ids_hash": self.game_ids_hash,
            "identity": self.identity,
            "features": list(FEATURE_NAMES),
            **self.extra,
        }


def objective_and_grad(
    theta: np.ndarray, examples: list[CalibrationExample], lam: float
) -> tuple[float, np.ndarray]:
    """Mean negative log probability of the actual joint score + lam*||theta||^2/2, and gradient."""
    n = len(examples)
    total = 0.0
    grad = np.zeros(4)
    for ex in examples:
        scores = ex.classes @ theta  # (K,)
        with np.errstate(divide="ignore"):
            log_terms = np.log(ex.mass) + scores
        finite = np.isfinite(log_terms)
        log_z = float(logsumexp(log_terms[finite]))
        probs = np.exp(log_terms[finite] - log_z)
        expected = probs @ ex.classes[finite]
        total += -(ex.log_base_actual + float(ex.f_actual @ theta) - log_z)
        grad += expected - ex.f_actual
    obj = total / n + 0.5 * lam * float(theta @ theta)
    g = grad / n + lam * theta
    return obj, g


def fit_theta(
    examples: list[CalibrationExample],
    lam: float,
    *,
    bound: float = 3.0,
    seasons: list[int],
    game_ids_hash: str,
    min_games: int,
) -> ThetaFit:
    """Deterministic L-BFGS-B fit from zero with analytic gradient and box bounds."""
    if len(examples) < min_games:
        raise MissingDataError(f"insufficient calibration history: {len(examples)} < {min_games}")
    if lam <= 0:
        raise ModelValidationError("lambda must be positive")
    res = minimize(
        lambda th: objective_and_grad(th, examples, lam),
        np.zeros(4),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-bound, bound)] * 4,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta = np.asarray(res.x, dtype=float)
    _obj, grad = objective_and_grad(theta, examples, lam)
    hits = [FEATURE_NAMES[i] for i in range(4) if abs(abs(theta[i]) - bound) < 1e-9]
    if not np.isfinite(theta).all() or (not res.success and "ABNORMAL" in str(res.message).upper()):
        return ThetaFit(
            theta,
            lam,
            "failed",
            str(res.message),
            int(res.nit),
            float(res.fun),
            float(np.linalg.norm(grad)),
            hits,
            len(examples),
            seasons,
            game_ids_hash,
        )
    return ThetaFit(
        theta,
        lam,
        "converged" if res.success else "stopped",
        str(res.message),
        int(res.nit),
        float(res.fun),
        float(np.linalg.norm(grad)),
        hits,
        len(examples),
        seasons,
        game_ids_hash,
    )


def identity_fit(n_games: int, seasons: list[int], game_ids_hash: str) -> ThetaFit:
    return ThetaFit(
        np.zeros(4),
        0.0,
        "identity",
        "theta=0 by construction",
        0,
        float("nan"),
        0.0,
        [],
        n_games,
        seasons,
        game_ids_hash,
        identity=True,
    )


def calibration_example(
    game_id: str, season: int, base: ScoreDistribution, actual_home: int, actual_away: int
) -> CalibrationExample:
    classes, mass = class_masses(base.pmf)
    h, a = int(actual_home), int(actual_away)
    if h > base.max_score or a > base.max_score:
        raise ModelValidationError(
            f"actual score {h}-{a} outside the base support {base.max_score}"
        )
    p = float(base.pmf[h, a])
    if p <= 0:
        raise ModelValidationError(
            f"base distribution assigns zero mass to the actual score {h}-{a}"
        )
    return CalibrationExample(game_id, season, classes, mass, feature_of(h, a), float(np.log(p)))


def adjusted_distribution(
    base: ScoreDistribution, theta: np.ndarray, cfg: DistributionConfig, *, rebuild: Any = None
) -> ScoreDistribution:
    """Adjust a base distribution; expand the base support until the adjusted tail bound holds."""
    theta = np.asarray(theta, dtype=float)
    current = base
    bound = adjusted_omitted_bound(current.omitted_upper_mass, theta)
    while bound > cfg.max_omitted_mass:
        if rebuild is None or current.max_score >= cfg.hard_max_score:
            raise ModelValidationError(
                f"adjusted omitted-mass bound {bound:.3e} exceeds {cfg.max_omitted_mass} "
                f"at support {current.max_score}"
            )
        current = rebuild(min(current.max_score + cfg.score_step, cfg.hard_max_score))
        bound = adjusted_omitted_bound(current.omitted_upper_mass, theta)
    pmf, log_z = adjust_pmf(current.pmf, theta)
    dist = ScoreDistribution(
        pmf=pmf,
        max_score=current.max_score,
        mu_home=current.mu_home,
        mu_away=current.mu_away,
        negative_latent_mass=current.negative_latent_mass,
        omitted_upper_mass=bound,
        clipped_negative_cells=current.clipped_negative_cells,
        warnings=[
            *current.warnings,
            f"key_number_adjusted theta={theta.round(4).tolist()} logZ={log_z:.6f}",
        ],
    )
    dist.validate(cfg.pmf_sum_tolerance)
    return dist


def base_from_location(
    location: np.ndarray,
    residual: ResidualParams,
    cfg: DistributionConfig,
    max_score: int | None = None,
) -> ScoreDistribution:
    c = cfg if max_score is None else cfg.model_copy(update={"initial_max_score": max_score})
    return score_distribution_from_normal(
        np.asarray(location) + residual.mean, residual.covariance, c
    )


def build_challenger_bundle(
    base: ModelBundle, fit: ThetaFit, *, model_id: str, calibration: dict[str, Any]
) -> ModelBundle:
    """Wrap a fitted V1 bundle with the tilt parameters; the base is stored verbatim."""
    params = {
        "family": FAMILY,
        "base": base.model_dump(),
        "theta": fit.theta.tolist(),
        "lambda": fit.lam,
        "fit": fit.to_dict(),
        "calibration": calibration,
        "features": list(FEATURE_NAMES),
    }
    return base.model_copy(
        update={
            "model_id": model_id,
            "family": FAMILY,
            "model_params": params,
            "notes": [
                *base.notes,
                f"challenger over base {base.model_id}; identity={fit.identity}",
            ],
        }
    )


def predict_adjusted(
    bundle: ModelBundle, feature_rows: pd.DataFrame, config: ExperimentConfig
) -> tuple[ScoreDistribution, dict[str, Any], tuple[float, float]]:
    if bundle.family != FAMILY:
        raise ModelValidationError("bundle is not a key_number_adjusted challenger")
    base_bundle = ModelBundle.model_validate(bundle.model_params["base"])
    theta = np.asarray(bundle.model_params["theta"], dtype=float)
    model = base_bundle.score_model(config.seed)
    mu = predict_location(model, feature_rows)
    loc = game_locations(feature_rows, mu).iloc[0]
    location = np.array([float(loc["mu_home_score"]), float(loc["mu_away_score"])])
    residual = base_bundle.residual_params()
    base = base_from_location(location, residual, config.distribution)
    dist = adjusted_distribution(
        base,
        theta,
        config.distribution,
        rebuild=lambda m: base_from_location(location, residual, config.distribution, m),
    )
    diag = {
        "base_omitted_upper_mass": base.omitted_upper_mass,
        "adjusted_omitted_bound": dist.omitted_upper_mass,
        "base_negative_latent_mass": base.negative_latent_mass,
        "max_score": dist.max_score,
        "base_max_score": base.max_score,
        "theta": theta.tolist(),
    }
    return dist, diag, (location[0], location[1])
