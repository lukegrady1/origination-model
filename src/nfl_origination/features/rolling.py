"""Recency-weighted, opportunity-weighted rolling rates with shrinkage (spec section 8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_origination.schemas import RATE_METRICS


def recency_weights(
    n: int, half_life: float, same_season: np.ndarray, offseason_weight: float
) -> np.ndarray:
    """w_j = 2^(-j/half_life) for j=0 most recent, halved (offseason_weight) outside target season.

    ``same_season`` is ordered most-recent-first and aligned with the returned weights.
    """
    j = np.arange(n, dtype=float)
    w = np.power(2.0, -j / half_life)
    return np.where(same_season, w, w * offseason_weight)


def weighted_shrunk_rate(
    numerators: np.ndarray,
    denominators: np.ndarray,
    weights: np.ndarray,
    k: float,
    r0: float,
) -> tuple[float, float, float]:
    """(N + k*r0)/(D + k) with N=sum(w*num), D=sum(w*den); null pairs contribute nothing."""
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den)
    n_sum = float(np.sum(weights[ok] * num[ok])) if ok.any() else 0.0
    d_sum = float(np.sum(weights[ok] * den[ok])) if ok.any() else 0.0
    return (n_sum + k * r0) / (d_sum + k), n_sum, d_sum


@dataclass(frozen=True)
class ShrinkageParams:
    plays: float = 200.0
    split_plays: float = 100.0
    games: float = 4.0

    def k_for(self, kind: str) -> float:
        return {"plays": self.plays, "split": self.split_plays, "games": self.games}[kind]


def league_priors(team_games: pd.DataFrame, seasons: list[int]) -> dict[str, float]:
    """League rates r0 from the given completed seasons: sum(num)/sum(den) per metric."""
    sub = team_games[team_games["season"].isin(seasons)]
    priors: dict[str, float] = {}
    for metric, (num, den, _kind) in RATE_METRICS.items():
        n = sub[num].astype(float)
        d = sub[den].astype(float)
        ok = n.notna() & d.notna()
        dsum = float(d[ok].sum())
        priors[metric] = float(n[ok].sum()) / dsum if dsum > 0 else float("nan")
    return priors


def rolling_metrics(
    history: pd.DataFrame,
    target_season: int,
    *,
    history_games: int,
    half_life: float,
    offseason_weight: float,
    shrinkage: ShrinkageParams,
    priors: dict[str, float],
) -> dict[str, float]:
    """Compute all shrunk metrics from a team's eligible history (most recent last).

    ``history`` rows must already be eligible under the as-of policy and sorted by kickoff.
    """
    recent = history.tail(history_games)
    n = len(recent)
    out: dict[str, float] = {}
    if n == 0:
        for metric in RATE_METRICS:
            out[metric] = priors[metric]
        return out
    seasons = recent["season"].to_numpy()[::-1]
    w = recency_weights(n, half_life, seasons == target_season, offseason_weight)
    for metric, (num, den, kind) in RATE_METRICS.items():
        rate, _n, _d = weighted_shrunk_rate(
            recent[num].to_numpy(dtype=float)[::-1],
            recent[den].to_numpy(dtype=float)[::-1],
            w,
            shrinkage.k_for(kind),
            priors[metric],
        )
        out[metric] = rate
    return out
