"""Markdown report, PNG charts, metrics JSON and prediction exports (spec section 16)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nfl_origination.errors import InvalidInputError
from nfl_origination.provenance import read_json

LIMITATIONS = [
    "Historical reconstruction uses current pinned nflverse files with a 48-hour eligibility "
    "convention; it is not a point-in-time-certified backtest. Upstream EPA values may have been "
    "revised or computed with models trained later (see the EPA-free ablation).",
    "No injuries, projected starters, weather, or depth charts are used. This limits real pregame "
    "usefulness and is disclosed rather than hidden.",
    "Scores come from a rounded, zero-floored bivariate normal. It permits implausible scores and "
    "smooths NFL key numbers (3, 7) and total parity; key-number diagnostics expose this.",
    "The 2025 holdout is a retrospective project holdout, not proof of no prior knowledge of "
    "its outcomes.",
    "Bookmaker lines are price references, not conditional means. A model-market gap is not "
    "proof of mispricing, and no positive ROI is claimed.",
]

KEY_METRICS = [
    ("score_mae", "Score MAE (team-game)"),
    ("score_rmse", "Score RMSE"),
    ("margin_mae", "Margin MAE"),
    ("margin_rmse", "Margin RMSE"),
    ("total_mae", "Total MAE"),
    ("total_rmse", "Total RMSE"),
    ("logloss_3way", "3-way log loss"),
    ("brier_3way", "3-way Brier (sum over classes)"),
    ("logloss_binary_no_tie", "Binary log loss (non-tied)"),
    ("brier_binary_no_tie", "Binary Brier (non-tied)"),
    ("crps_margin", "CRPS margin"),
    ("crps_total", "CRPS total"),
    ("margin_cover_50", "Margin 50% coverage"),
    ("margin_cover_80", "Margin 80% coverage"),
    ("margin_cover_95", "Margin 95% coverage"),
    ("margin_width_80", "Margin 80% mean width"),
    ("total_cover_80", "Total 80% coverage"),
    ("predicted_tie_rate", "Predicted tie rate"),
    ("observed_tie_rate", "Observed tie rate"),
]


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _chart_reliability(metrics: dict[str, Any], primary: str, baseline: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
        (axes[0], "reliability_home_win_3way", "Home win (3-way probability)"),
        (axes[1], "reliability_home_win_given_no_tie", "Home win given no tie"),
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for mid, marker in ((primary, "o"), (baseline, "s")):
            rows = metrics["models"].get(mid, {}).get(key, [])
            xs = [r["mean_predicted"] for r in rows if r["n"]]
            ys = [r["observed_rate"] for r in rows if r["n"]]
            sparse = [r["sparse"] for r in rows if r["n"]]
            ax.plot(xs, ys, marker=marker, lw=1, label=mid)
            for x, y, sp in zip(xs, ys, sparse, strict=True):
                if sp:
                    ax.annotate("sparse", (x, y), fontsize=6, alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_season_mae(metrics: dict[str, Any], model_ids: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    seasons = sorted(
        {
            int(k.split("_")[1])
            for mid in model_ids
            for k in metrics["models"].get(mid, {}).get("slices", {})
            if k.startswith("season_")
        }
    )
    width = 0.8 / max(len(model_ids), 1)
    for i, mid in enumerate(model_ids):
        sl = metrics["models"].get(mid, {}).get("slices", {})
        vals = [sl.get(f"season_{s}", {}).get("score_mae", np.nan) for s in seasons]
        ax.bar(np.arange(len(seasons)) + i * width, vals, width, label=mid)
    ax.set_xticks(np.arange(len(seasons)) + width * (len(model_ids) - 1) / 2)
    ax.set_xticklabels([str(s) for s in seasons])
    ax.set_ylabel("score MAE (points)")
    ax.set_title("Season score MAE by model")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_coverage(metrics: dict[str, Any], model_ids: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    levels = [50, 80, 95]
    width = 0.8 / max(len(model_ids), 1)
    for i, mid in enumerate(model_ids):
        pooled = metrics["models"].get(mid, {}).get("pooled", {})
        vals = [pooled.get(f"margin_cover_{c}", np.nan) for c in levels]
        ax.bar(np.arange(3) + i * width, vals, width, label=mid)
    ax.plot(
        np.arange(3) + width * (len(model_ids) - 1) / 2,
        [c / 100 for c in levels],
        "k_",
        ms=25,
        label="nominal",
    )
    ax.set_xticks(np.arange(3) + width * (len(model_ids) - 1) / 2)
    ax.set_xticklabels([f"{c}%" for c in levels])
    ax.set_ylabel("empirical coverage")
    ax.set_title("Margin interval coverage")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_key_numbers(metrics: dict[str, Any], primary: str, path: Path) -> None:
    kn = metrics["models"].get(primary, {}).get("key_numbers", {})
    keys = [k for k in kn if k != "n_games"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [kn[k]["predicted"] for k in keys], 0.4, label="predicted")
    ax.bar(x + 0.2, [kn[k]["observed"] for k in keys], 0.4, label="observed")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30)
    ax.set_title(f"Key-number diagnostics ({primary})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _chart_differences(metrics: dict[str, Any], comparison_key: str, path: Path) -> None:
    comp = metrics["comparisons"].get(comparison_key, {})
    cols = [
        c
        for c, _ in KEY_METRICS
        if c in comp and c not in ("predicted_tie_rate", "observed_tie_rate")
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(cols):
        d = comp[c]
        ax.errorbar(
            d["difference"],
            i,
            xerr=[[d["difference"] - d["ci_low"]], [d["ci_high"] - d["difference"]]],
            fmt="o",
            color="C0",
            capsize=3,
        )
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("difference (candidate minus baseline); negative is better for errors")
    ax.set_title(f"Paired block-bootstrap 95% intervals: {comparison_key}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def recommendation(metrics: dict[str, Any]) -> str:
    primary, baseline = metrics["primary_model_id"], metrics["baseline_model_id"]
    comp = metrics["comparisons"].get(f"{primary}_vs_{baseline}")
    if not comp:
        return "No paired comparison available; baseline preferred by default."
    wins = []
    losses = []
    for c in ("score_mae", "logloss_3way", "crps_margin", "crps_total"):
        d = comp.get(c)
        if not d:
            continue
        if d["ci_high"] < 0:
            wins.append(c)
        elif d["ci_low"] > 0:
            losses.append(c)
    if wins and not losses:
        return (
            f"{primary} improves on {baseline} with bootstrap intervals excluding zero on "
            f"{', '.join(wins)}. Recommend the candidate as the V1 origination model, with the "
            "stated limitations. Further modeling is still needed before any real pregame use."
        )
    if losses and not wins:
        return (
            f"Baseline preferred: {primary} is worse on {', '.join(losses)}. "
            "Further modeling needed."
        )
    return (
        "Inconclusive: bootstrap intervals include zero on the headline metrics. "
        "Baseline preferred until further modeling shows a reliable improvement."
    )


def build_report(run_dir: Path, reports_dir: Path | None = None) -> Path:
    """Build report.md, charts, metrics.json and predictions CSV for a run directory."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise InvalidInputError(f"run directory {run_dir} has no manifest.json")
    manifest = read_json(manifest_path)
    out = run_dir / "report"
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    kind = manifest["run_kind"]
    lines: list[str] = []
    synthetic = any("SYNTHETIC" in n for n in manifest.get("notes", []))
    title = f"# NFL Origination Model — {kind} run `{manifest['run_id']}`"
    lines.append(title)
    if synthetic:
        lines.append(
            "\n> **SYNTHETIC DATA.** This run uses generated fixtures, not NFL observations.\n"
        )
    lines.append(
        f"\nCreated {manifest['created_at_utc']} · data mode `{manifest['data_mode']}` · "
        "forecast policy "
        f"`{manifest['forecast_policy']}` · git `{manifest.get('git_commit')}`"
        f"{' (dirty)' if manifest.get('git_dirty') else ''} · "
        f"config hash `{manifest['config_hash'][:12]}`"
    )
    if kind == "forecast":
        _forecast_report(run_dir, manifest, lines)
    else:
        metrics = read_json(run_dir / "metrics.json")
        _backtest_report(run_dir, manifest, metrics, lines, charts)
        shutil.copy(run_dir / "metrics.json", out / "metrics.json")
        preds = pd.read_parquet(run_dir / "predictions.parquet")
        preds.to_csv(out / "predictions.csv", index=False)
        shutil.copy(run_dir / "predictions.parquet", out / "predictions.parquet")
    market_path = run_dir / "paper_backtest.json"
    lines.append("\n## Market comparison and paper backtest\n")
    if market_path.exists():
        pb = read_json(market_path)
        lines.append("```json\n" + json.dumps(pb, indent=1, default=str)[:4000] + "\n```")
        if pb.get("synthetic"):
            lines.append(
                "\n**Synthetic odds: settlement mechanics only, not evidence of returns.**"
            )
    else:
        lines.append("unavailable: no eligible timestamped odds")
    compare_path = run_dir / "market_comparison.json"
    if compare_path.exists():
        lines.append("\n### Comparison summary\n")
        lines.append(
            "```json\n"
            + json.dumps(read_json(compare_path), indent=1, default=str)[:3000]
            + "\n```"
        )
    lines.append("\n## Limitations\n")
    lines += [f"- {item}" for item in LIMITATIONS]
    lines.append("\n## Provenance\n")
    lines.append(
        "```json\n"
        + json.dumps(
            {
                k: manifest[k]
                for k in (
                    "run_id",
                    "config_hash",
                    "source_file_hashes",
                    "output_hashes",
                    "dependency_versions",
                    "notes",
                )
            },
            indent=1,
        )[:4000]
        + "\n```"
    )
    report_path = out / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    if reports_dir is not None:
        dest = Path(reports_dir) / manifest["run_id"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(out, dest)
    return report_path


def _backtest_report(
    run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any], lines: list[str], charts: Path
) -> None:
    primary, baseline = metrics["primary_model_id"], metrics["baseline_model_id"]
    model_ids = list(metrics["selected_model_ids"])
    lines.append(
        f"\nScored seasons: {metrics['score_seasons']} · primary model `{primary}` · "
        f"baseline `{baseline}`"
    )
    lines.append("\n## Methodology\n")
    lines.append(
        "Expanding chronological fits (train on 2012…Y−1, freeze for season Y), residual "
        "mean/covariance from the five most recent out-of-fold seasons, rounded bivariate-normal "
        "score PMF, fair prices from the PMF. Each game's cutoff is kickoff minus 24 hours; "
        "source games need kickoff + 48h ≤ cutoff."
    )
    cov = metrics.get("coverage")
    lines.append("\n## Samples and exclusions\n")
    if cov:
        lines.append(
            _table(
                ["season", "scheduled", "expected", "final", "with PBP", "excluded", "complete"],
                [
                    [
                        s["season"],
                        s["scheduled_games"],
                        s["expected_games"],
                        s["final_games"],
                        s["games_with_pbp"],
                        s["excluded_games"],
                        s["complete"],
                    ]
                    for s in cov["seasons"]
                ],
            )
        )
        if cov.get("notes"):
            lines.append("\nNotes: " + "; ".join(cov["notes"]))
    lines.append(f"\nExclusion rows: {metrics['exclusions']['n']} (see exclusions.csv).")
    lines.append("\n## Pooled metrics\n")
    rows = []
    for key, label in KEY_METRICS:
        rows.append(
            [label, *[metrics["models"].get(m, {}).get("pooled", {}).get(key) for m in model_ids]]
        )
    rows.append(
        [
            "Games",
            *[metrics["models"].get(m, {}).get("pooled", {}).get("n_games") for m in model_ids],
        ]
    )
    lines.append(_table(["metric", *model_ids], rows))
    lines.append("\n## Paired differences vs baseline (block bootstrap 95% intervals)\n")
    for key, comp in metrics["comparisons"].items():
        if not key.startswith(tuple(m for m in model_ids if m != baseline)):
            continue
        lines.append(f"\n**{key}**\n")
        lines.append(
            _table(
                ["metric", "candidate", "baseline", "difference", "ci low", "ci high", "games"],
                [
                    [
                        c,
                        d["mean_a"],
                        d["mean_b"],
                        d["difference"],
                        d["ci_low"],
                        d["ci_high"],
                        d["n_games"],
                    ]
                    for c, d in comp.items()
                    if c in dict(KEY_METRICS)
                ],
            )
        )
    lines.append("\n## Season and slice results (primary model)\n")
    sl = metrics["models"].get(primary, {}).get("slices", {})
    lines.append(
        _table(
            [
                "slice",
                "games",
                "score MAE",
                "margin MAE",
                "total MAE",
                "3-way log loss",
                "CRPS margin",
                "80% cover",
            ],
            [
                [
                    k,
                    v.get("n_games"),
                    v.get("score_mae"),
                    v.get("margin_mae"),
                    v.get("total_mae"),
                    v.get("logloss_3way"),
                    v.get("crps_margin"),
                    v.get("margin_cover_80"),
                ]
                for k, v in sl.items()
            ],
        )
    )
    lines.append("\n## Reliability (primary model, 0.1 bins; sparse = fewer than 30 games)\n")
    rel = metrics["models"].get(primary, {}).get("reliability_home_win_3way", [])
    lines.append(
        _table(
            ["bin", "n", "mean predicted", "observed", "sparse"],
            [[r["bin"], r["n"], r["mean_predicted"], r["observed_rate"], r["sparse"]] for r in rel],
        )
    )
    lines.append("\n## Key-number diagnostics (primary model)\n")
    kn = metrics["models"].get(primary, {}).get("key_numbers", {})
    lines.append(
        _table(
            ["event", "predicted", "observed"],
            [[k, v["predicted"], v["observed"]] for k, v in kn.items() if k != "n_games"],
        )
    )
    if "decision_record" in metrics:
        lines.append("\n## Alpha decision record\n")
        for fs, rec in metrics["decision_record"].items():
            if not isinstance(rec, dict) or "candidates" not in rec:
                continue
            lines.append(
                f"\nFeature set `{fs}`: selected alpha **{rec['selected_alpha']}** "
                f"({rec['rule']}).\n"
            )
            lines.append(
                _table(
                    ["alpha", "pooled score MAE", "games"],
                    [[c["alpha"], c["pooled_score_mae"], c["n_games"]] for c in rec["candidates"]],
                )
            )
    lines.append("\n## Recommendation\n")
    lines.append(recommendation(metrics))
    _chart_reliability(metrics, primary, baseline, charts / "reliability.png")
    _chart_season_mae(metrics, model_ids, charts / "season_score_mae.png")
    _chart_coverage(metrics, model_ids, charts / "interval_coverage.png")
    _chart_key_numbers(metrics, primary, charts / "key_numbers.png")
    comp_key = f"{primary}_vs_{baseline}"
    if comp_key in metrics["comparisons"]:
        _chart_differences(metrics, comp_key, charts / "paired_differences.png")
    lines.append("\n## Charts\n")
    for name in (
        "reliability",
        "season_score_mae",
        "interval_coverage",
        "key_numbers",
        "paired_differences",
    ):
        if (charts / f"{name}.png").exists():
            lines.append(f"![{name}](charts/{name}.png)")


def _forecast_report(run_dir: Path, manifest: dict[str, Any], lines: list[str]) -> None:
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    lines.append(
        f"\nForecast slate `{manifest['label']}` · model `{manifest['model_id']}` · "
        f"policy `{manifest['forecast_policy']}`"
    )
    if preds.empty:
        lines.append("\nEmpty slate: no games matched the request.")
        return
    primary = preds[preds["model_id"] == manifest["model_id"]]
    lines.append("\n## Slate\n")
    lines.append(
        _table(
            [
                "game",
                "kickoff (UTC)",
                "cutoff (UTC)",
                "fair home handicap",
                "fair total",
                "home ML",
                "away ML",
                "P(tie)",
                "flags",
            ],
            [
                [
                    f"{r.away_team} @ {r.home_team}",
                    str(r.kickoff_utc)[:16],
                    str(r.cutoff_utc)[:16],
                    r.fair_home_handicap,
                    r.fair_total,
                    r.fair_american_home,
                    r.fair_american_away,
                    r.p_tie,
                    r.data_quality_flags,
                ]
                for r in primary.itertuples()
            ],
        )
    )
    primary.to_csv(run_dir / "report" / "slate.csv", index=False)
