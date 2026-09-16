"""Per-game scoring columns and aggregate metrics (spec section 11)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-15
INTERVALS = (50, 80, 95)
KEY_MARGINS = (0, 3, -3, 7, -7)

MEAN_METRICS = [
    "score_mae",
    "score_rmse_sq",
    "margin_mae",
    "margin_rmse_sq",
    "total_mae",
    "total_rmse_sq",
    "logloss_3way",
    "brier_3way",
    "logloss_binary_no_tie",
    "brier_binary_no_tie",
    "crps_margin",
    "crps_total",
    *[f"margin_cover_{c}" for c in INTERVALS],
    *[f"margin_width_{c}" for c in INTERVALS],
    *[f"total_cover_{c}" for c in INTERVALS],
    *[f"total_width_{c}" for c in INTERVALS],
    "p_tie",
    "is_tie",
]


def per_game_scores(pred: pd.DataFrame) -> pd.DataFrame:
    """Add per-game scoring columns. ``pred`` needs actual scores and prediction fields."""
    df = pred.copy()
    h, a = df["actual_home"].to_numpy(float), df["actual_away"].to_numpy(float)
    mh, ma = df["dist_mean_home_score"].to_numpy(float), df["dist_mean_away_score"].to_numpy(float)
    df["score_mae"] = 0.5 * (np.abs(h - mh) + np.abs(a - ma))
    df["score_rmse_sq"] = 0.5 * ((h - mh) ** 2 + (a - ma) ** 2)
    margin, total = h - a, h + a
    df["actual_margin"], df["actual_total"] = margin, total
    df["margin_mae"] = np.abs(margin - df["mean_margin"].to_numpy(float))
    df["margin_rmse_sq"] = (margin - df["mean_margin"].to_numpy(float)) ** 2
    df["total_mae"] = np.abs(total - df["mean_total"].to_numpy(float))
    df["total_rmse_sq"] = (total - df["mean_total"].to_numpy(float)) ** 2
    ph = np.clip(df["p_home_win"].to_numpy(float), EPS, 1.0)
    pt = np.clip(df["p_tie"].to_numpy(float), EPS, 1.0)
    pa = np.clip(df["p_away_win"].to_numpy(float), EPS, 1.0)
    yh, yt, ya = (margin > 0).astype(float), (margin == 0).astype(float), (margin < 0).astype(float)
    df["is_tie"] = yt
    df["home_win"] = yh
    df["logloss_3way"] = -(yh * np.log(ph) + yt * np.log(pt) + ya * np.log(pa))
    df["brier_3way"] = (ph - yh) ** 2 + (pt - yt) ** 2 + (pa - ya) ** 2
    q = np.clip(df["p_home_win_given_no_tie"].to_numpy(float), EPS, 1 - EPS)
    non_tie = margin != 0
    df["logloss_binary_no_tie"] = np.where(
        non_tie, -(yh * np.log(q) + (1 - yh) * np.log(1 - q)), np.nan
    )
    df["brier_binary_no_tie"] = np.where(non_tie, (q - yh) ** 2, np.nan)
    for c in INTERVALS:
        for kind, actual in (("margin", margin), ("total", total)):
            lo = df[f"{kind}_q{c}_lo"].to_numpy(float)
            hi = df[f"{kind}_q{c}_hi"].to_numpy(float)
            df[f"{kind}_cover_{c}"] = ((actual >= lo) & (actual <= hi)).astype(float)
            df[f"{kind}_width_{c}"] = hi - lo
    for m in KEY_MARGINS:
        df[f"obs_margin_{m}"] = (margin == m).astype(float)
    df["obs_total_even"] = (total % 2 == 0).astype(float)
    return df


def aggregate(scored: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    """Weighted means of per-game columns, with RMSEs derived from squared errors."""
    if scored.empty:
        return {"n_games": 0}
    w = np.ones(len(scored)) if weights is None else np.asarray(weights, dtype=float)
    out: dict[str, float] = {"n_games": float(w.sum())}
    for col in MEAN_METRICS:
        if col not in scored.columns:
            continue
        vals = scored[col].to_numpy(float)
        ok = np.isfinite(vals)
        denom = float(w[ok].sum())
        out[col] = float((w[ok] * vals[ok]).sum() / denom) if denom > 0 else float("nan")
    for kind in ("score", "margin", "total"):
        out[f"{kind}_rmse"] = float(np.sqrt(out.pop(f"{kind}_rmse_sq")))
    out["n_ties"] = float((w * scored["is_tie"].to_numpy(float)).sum())
    out["predicted_tie_rate"] = out.pop("p_tie")
    out["observed_tie_rate"] = out.pop("is_tie")
    for m in KEY_MARGINS:
        pcol, ocol = f"p_margin_{m}", f"obs_margin_{m}"
        if pcol in scored.columns:
            out[f"pred_margin_{m}"] = float((w * scored[pcol].to_numpy(float)).sum() / w.sum())
            out[f"obs_margin_{m}"] = float((w * scored[ocol].to_numpy(float)).sum() / w.sum())
    if "p_total_even" in scored.columns:
        out["pred_total_even"] = float((w * scored["p_total_even"].to_numpy(float)).sum() / w.sum())
        out["obs_total_even"] = float(
            (w * scored["obs_total_even"].to_numpy(float)).sum() / w.sum()
        )
    return out


def reliability_table(
    prob: np.ndarray, outcome: np.ndarray, *, sparse_threshold: int = 30
) -> list[dict[str, Any]]:
    """Fixed 0.1 bins: mean predicted, observed frequency, count, sparse flag."""
    prob = np.asarray(prob, float)
    outcome = np.asarray(outcome, float)
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, 9)
    rows = []
    for b in range(10):
        m = idx == b
        n = int(m.sum())
        rows.append(
            {
                "bin": f"[{edges[b]:.1f}, {edges[b + 1]:.1f}{']' if b == 9 else ')'}",
                "n": n,
                "mean_predicted": float(prob[m].mean()) if n else None,
                "observed_rate": float(outcome[m].mean()) if n else None,
                "sparse": n < sparse_threshold,
            }
        )
    return rows


def slice_metrics(scored: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Season, early-season, neutral-site, and cold-start slices with sample sizes."""
    out: dict[str, dict[str, Any]] = {}
    for season, part in scored.groupby("season"):
        out[f"season_{season}"] = aggregate(part)
    out["early_weeks_1_4"] = aggregate(scored[scored["week"] <= 4])
    if "neutral_site" in scored.columns:
        out["neutral_site"] = aggregate(scored[scored["neutral_site"].astype(bool)])
    if "cold_start" in scored.columns:
        out["cold_start"] = aggregate(scored[scored["cold_start"].astype(bool)])
    return out


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> dict[str, float]:
    """Difference of aggregate metric on exactly matched games (a minus b)."""
    key = ["game_id"]
    merged = a[[*key, metric]].merge(b[[*key, metric]], on=key, suffixes=("_a", "_b"))
    da = merged[f"{metric}_a"].to_numpy(float)
    db = merged[f"{metric}_b"].to_numpy(float)
    ok = np.isfinite(da) & np.isfinite(db)
    return {
        "n_matched": int(ok.sum()),
        "mean_a": float(da[ok].mean()) if ok.any() else float("nan"),
        "mean_b": float(db[ok].mean()) if ok.any() else float("nan"),
        "difference": float((da[ok] - db[ok]).mean()) if ok.any() else float("nan"),
    }
