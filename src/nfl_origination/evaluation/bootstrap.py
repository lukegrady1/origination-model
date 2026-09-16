"""Season-week block bootstrap for paired metric differences (spec section 11)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nfl_origination.evaluation.metrics import MEAN_METRICS

PAIRED_COLUMNS = [c for c in MEAN_METRICS if not c.endswith("_sq") and c not in ("p_tie", "is_tie")]


def _block_sums(
    scored: pd.DataFrame, blocks: pd.Index, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    key = pd.MultiIndex.from_frame(scored[["season", "week"]])
    sums = np.zeros((len(blocks), len(columns)))
    counts = np.zeros((len(blocks), len(columns)))
    pos = {b: i for i, b in enumerate(blocks)}
    idx = np.array([pos[k] for k in key])
    for j, col in enumerate(columns):
        vals = scored[col].to_numpy(float)
        ok = np.isfinite(vals)
        np.add.at(sums[:, j], idx[ok], vals[ok])
        np.add.at(counts[:, j], idx[ok], 1.0)
    return sums, counts


def block_bootstrap_differences(
    scored_a: pd.DataFrame,
    scored_b: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    columns: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Percentile 95% intervals for pooled-mean differences (a minus b) on matched games.

    Blocks are (season, week); each replicate samples blocks with replacement and recomputes
    game-weighted pooled means with all games of each selected block included.
    """
    cols = [
        c for c in (columns or PAIRED_COLUMNS) if c in scored_a.columns and c in scored_b.columns
    ]
    a = scored_a.merge(scored_b[["game_id"]], on="game_id")
    b = scored_b.merge(scored_a[["game_id"]], on="game_id")
    a = a.sort_values("game_id").reset_index(drop=True)
    b = b.sort_values("game_id").reset_index(drop=True)
    blocks = pd.MultiIndex.from_frame(
        a[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    )
    sums_a, counts_a = _block_sums(a, blocks, cols)
    sums_b, counts_b = _block_sums(b, blocks, cols)
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=replicates).astype(
        float
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = (weights @ sums_a) / (weights @ counts_a)
        mean_b = (weights @ sums_b) / (weights @ counts_b)
    diff = mean_a - mean_b
    point_a = sums_a.sum(axis=0) / np.maximum(counts_a.sum(axis=0), 1)
    point_b = sums_b.sum(axis=0) / np.maximum(counts_b.sum(axis=0), 1)
    out: dict[str, dict[str, Any]] = {}
    for j, col in enumerate(cols):
        d = diff[:, j]
        d = d[np.isfinite(d)]
        out[col] = {
            "mean_a": float(point_a[j]),
            "mean_b": float(point_b[j]),
            "difference": float(point_a[j] - point_b[j]),
            "ci_low": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
            "ci_high": float(np.percentile(d, 97.5)) if len(d) else float("nan"),
            "replicates": len(d),
            "n_blocks": int(n_blocks),
            "n_games": len(a),
        }
    return out


def block_bootstrap_ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    block_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a ratio statistic (e.g. ROI = profit / stake) over (season, week) blocks."""
    blocks, idx = np.unique(block_ids, return_inverse=True)
    num = np.zeros(len(blocks))
    den = np.zeros(len(blocks))
    np.add.at(num, idx, numerators)
    np.add.at(den, idx, denominators)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=replicates)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = (w @ num) / (w @ den)
    ratios = ratios[np.isfinite(ratios)]
    point = float(num.sum() / den.sum()) if den.sum() else float("nan")
    return {
        "point": point,
        "ci_low": float(np.percentile(ratios, 2.5)) if len(ratios) else float("nan"),
        "ci_high": float(np.percentile(ratios, 97.5)) if len(ratios) else float("nan"),
        "n_blocks": len(blocks),
    }
