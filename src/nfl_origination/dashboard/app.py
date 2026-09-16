"""Small artifact-driven Streamlit dashboard (spec section 16).

Run: uv run streamlit run src/nfl_origination/dashboard/app.py
It only reads saved run artifacts. No training or ad hoc recalculation happens on refresh.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACT_ROOTS = [
    Path(p)
    for p in os.environ.get("NFL_ORIGINATION_ARTIFACTS", "artifacts:artifacts/demo").split(":")
]
UNAVAILABLE = "unavailable: no eligible timestamped odds"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def list_runs() -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for root in ARTIFACT_ROOTS:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            continue
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if (d / "manifest.json").exists():
                runs.append((f"{root.name}/{d.name}" if root.name != "artifacts" else d.name, d))
    return runs


@st.cache_data(show_spinner=False)
def load_predictions(run_dir: str) -> pd.DataFrame:
    path = Path(run_dir) / "predictions.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_predictions(run_dir: str, model_id: str) -> pd.DataFrame:
    path = Path(run_dir) / f"predictions_{model_id}.parquet"
    if not path.exists():
        preds = load_predictions(run_dir)
        return preds[preds["model_id"] == model_id] if len(preds) else preds
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_distributions(run_dir: str, model_id: str) -> pd.DataFrame:
    for name in (f"distributions_{model_id}.parquet", "distributions.parquet"):
        path = Path(run_dir) / name
        if path.exists():
            df = pd.read_parquet(path)
            return df[df["model_id"] == model_id] if "model_id" in df else df
    return pd.DataFrame()


def banner(manifest: dict) -> None:
    if any("SYNTHETIC" in n for n in manifest.get("notes", [])) or manifest["run_kind"] == "demo":
        st.warning("SYNTHETIC / DEMO DATA — generated fixtures, not real NFL observations.")
    st.caption(
        f"run `{manifest['run_id']}` · kind `{manifest['run_kind']}` · "
        f"data mode `{manifest['data_mode']}` · "
        f"forecast policy `{manifest['forecast_policy']}` · model `{manifest.get('model_id')}` · "
        f"git `{(manifest.get('git_commit') or 'n/a')[:10]}`"
        f"{' (dirty)' if manifest.get('git_dirty') else ''}"
    )


def fmt_american(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "unavailable"
    return f"{v:+.0f}"


def page_slate(run_dir: Path, manifest: dict) -> None:
    preds = load_predictions(str(run_dir))
    if preds.empty:
        st.info("Empty slate: no predictions in this run.")
        return
    primary = preds[preds["model_id"] == manifest.get("model_id")] if "model_id" in preds else preds
    if primary.empty:
        primary = preds
    seasons = sorted(primary["season"].unique())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)
    weeks = sorted(primary[primary["season"] == season]["week"].unique())
    week = st.selectbox("Week", weeks, index=len(weeks) - 1)
    view = primary[(primary["season"] == season) & (primary["week"] == week)].copy()
    view["game"] = view["away_team"] + " @ " + view["home_team"]
    view["home ML"] = view["fair_american_home"].map(fmt_american)
    view["away ML"] = view["fair_american_away"].map(fmt_american)
    cols = [
        "game",
        "kickoff_utc",
        "cutoff_utc",
        "fair_home_handicap",
        "mean_home_handicap",
        "fair_total",
        "mean_total",
        "home ML",
        "away ML",
        "p_tie",
        "model_id",
        "feature_version",
        "data_quality_flags",
        "warnings",
    ]
    st.dataframe(view[cols].sort_values("kickoff_utc"), width="stretch", hide_index=True)
    st.caption(
        "Fair prices are model outputs at the stated cutoff. Missing-history and warning flags "
        "are shown verbatim; unavailable values are labeled, never filled."
    )


def page_game(run_dir: Path, manifest: dict) -> None:
    preds = load_predictions(str(run_dir))
    if preds.empty:
        st.info("No predictions in this run.")
        return
    game_ids = sorted(preds["game_id"].unique())
    gid = st.selectbox("Game", game_ids)
    rows = preds[preds["game_id"] == gid]
    models = sorted(rows["model_id"].unique())
    metrics = _read_json(run_dir / "metrics.json") or {}
    baseline_id = metrics.get("baseline_model_id", "B0_league_baseline")
    base_rows = load_model_predictions(str(run_dir), baseline_id)
    base_rows = base_rows[base_rows["game_id"] == gid] if len(base_rows) else base_rows
    st.subheader("Baseline versus candidate")
    compare = pd.concat([rows, base_rows]).drop_duplicates(subset=["model_id"])
    show = [
        "model_id",
        "dist_mean_home_score",
        "dist_mean_away_score",
        "mean_margin",
        "mean_total",
        "p_home_win",
        "p_tie",
        "p_away_win",
        "fair_home_handicap",
        "fair_total",
        "margin_q50_lo",
        "margin_q50_hi",
        "margin_q80_lo",
        "margin_q80_hi",
        "margin_q95_lo",
        "margin_q95_hi",
        "total_q80_lo",
        "total_q80_hi",
    ]
    st.dataframe(
        compare[[c for c in show if c in compare.columns]],
        hide_index=True,
        width="stretch",
    )
    r = rows.iloc[0]
    st.caption(
        f"Cutoff {r['cutoff_utc']} · kickoff {r['kickoff_utc']} · policy {r['forecast_policy']} · "
        f"flags: {r['data_quality_flags'] or 'none'} · warnings: {r['warnings'] or 'none'}"
    )
    feats_path = run_dir / "features.parquet"
    if feats_path.exists():
        feats = pd.read_parquet(feats_path)
        gf = feats[feats["game_id"] == gid]
        st.subheader("Feature rows (as-of cutoff)")
        st.dataframe(gf.T, width="stretch")
    st.subheader("Margin and total distributions")
    for model_id in models:
        dists = load_distributions(str(run_dir), model_id)
        d = dists[dists["game_id"] == gid] if len(dists) else dists
        if d.empty:
            st.write(f"{model_id}: distribution not saved")
            continue
        d0 = d.iloc[0]
        max_score = int(d0["max_score"])
        margin = np.asarray(d0["margin_pmf"])
        total = np.asarray(d0["total_pmf"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        m = np.arange(-max_score, max_score + 1)
        sel = (m >= -40) & (m <= 40)
        axes[0].bar(m[sel], margin[sel], width=1.0)
        axes[0].set_title(f"{model_id}: margin (home − away)")
        t = np.arange(0, 2 * max_score + 1)
        sel = t <= 90
        axes[1].bar(t[sel], total[sel], width=1.0)
        axes[1].set_title("total")
        st.pyplot(fig)
        plt.close(fig)
        if "actual_home" in rows.columns and pd.notna(r.get("actual_home")):
            st.caption(f"Actual result: {int(r['actual_home'])}-{int(r['actual_away'])}")
    st.subheader("Matched market quotes")
    comp_path = run_dir / "market_comparison.parquet"
    if comp_path.exists():
        comp = pd.read_parquet(comp_path)
        c = comp[comp["game_id"] == gid]
        if len(c) and bool(c.iloc[0].get("market_available", False)):
            st.dataframe(c.T, width="stretch")
            st.caption(
                "A model-market gap is a price-reference difference, not proof of mispricing."
            )
        else:
            st.write(UNAVAILABLE)
    else:
        st.write(UNAVAILABLE)


def page_evaluation(run_dir: Path, manifest: dict) -> None:
    metrics = _read_json(run_dir / "metrics.json")
    if not metrics:
        st.info("No evaluation metrics for this run (forecast runs have no actual results).")
        return
    models = metrics["selected_model_ids"]
    st.subheader("Pooled metrics")
    rows = []
    keys = [
        "n_games",
        "score_mae",
        "score_rmse",
        "margin_mae",
        "total_mae",
        "logloss_3way",
        "brier_3way",
        "logloss_binary_no_tie",
        "crps_margin",
        "crps_total",
        "margin_cover_50",
        "margin_cover_80",
        "margin_cover_95",
        "predicted_tie_rate",
        "observed_tie_rate",
    ]
    for m in models:
        pooled = metrics["models"].get(m, {}).get("pooled", {})
        rows.append({"model": m, **{k: pooled.get(k) for k in keys}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Season table (primary model)")
    primary = metrics["primary_model_id"]
    sl = metrics["models"].get(primary, {}).get("slices", {})
    st.dataframe(
        pd.DataFrame([{"slice": k, **{kk: v.get(kk) for kk in keys}} for k, v in sl.items()]),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Paired differences versus baseline (block bootstrap 95%)")
    for key, comp in metrics["comparisons"].items():
        if not key.startswith(primary):
            continue
        st.dataframe(
            pd.DataFrame([{"metric": c, **d} for c, d in comp.items()]),
            hide_index=True,
            width="stretch",
        )
    st.subheader("Reliability (0.1 bins; sparse = fewer than 30 games)")
    rel = metrics["models"].get(primary, {}).get("reliability_home_win_3way", [])
    st.dataframe(pd.DataFrame(rel), hide_index=True, width="stretch")
    st.subheader("Key-number diagnostics")
    kn = metrics["models"].get(primary, {}).get("key_numbers", {})
    st.dataframe(
        pd.DataFrame([{"event": k, **v} for k, v in kn.items() if k != "n_games"]), hide_index=True
    )
    st.caption(
        "These expose the rounded-normal distribution's shortcomings at NFL key numbers; "
        "they are not solved in V1."
    )
    st.subheader("Exclusions")
    ex = run_dir / "exclusions.csv"
    if ex.exists():
        st.dataframe(pd.read_csv(ex), hide_index=True, width="stretch")
    charts = run_dir / "report" / "charts"
    if charts.exists():
        for png in sorted(charts.glob("*.png")):
            st.image(str(png), caption=png.stem)
    st.subheader("Market panels")
    pb = _read_json(run_dir / "paper_backtest.json")
    if pb and pb.get("status") == "ok":
        if pb.get("synthetic"):
            st.warning("Synthetic odds: settlement mechanics only, not evidence of returns.")
        st.json({k: v for k, v in pb.items() if k not in ("exclusions",)})
    else:
        st.write(UNAVAILABLE)


def page_audit(run_dir: Path, manifest: dict) -> None:
    st.subheader("Run manifest")
    st.json({k: v for k, v in manifest.items() if k != "config"})
    st.subheader("Resolved configuration")
    st.json(manifest.get("config", {}))
    st.subheader("Limitations")
    st.markdown(
        "- Historical reconstruction with a 48h eligibility convention; "
        "not point-in-time certified.\n"
        "- No injuries, starters, weather or depth charts.\n"
        "- Rounded bivariate-normal scores smooth key numbers.\n"
        "- Bookmaker lines are price references, not conditional means.\n"
        "- Synthetic/demo runs are labeled and are not evidence."
    )
    st.subheader("Downloads")
    for name in (
        "manifest.json",
        "metrics.json",
        "predictions.parquet",
        "report/predictions.csv",
        "report/report.md",
        "exclusions.csv",
        "decision_record.json",
        "paper_backtest.json",
    ):
        path = run_dir / name
        if path.exists():
            st.download_button(name, path.read_bytes(), file_name=path.name, key=name)
    st.caption(
        "Sources: nflverse (schedules by Lee Sharpe, nflfastR play-by-play). "
        "Model coefficients are diagnostics, not causal explanations."
    )


def main() -> None:
    st.set_page_config(page_title="NFL Origination Model", layout="wide")
    st.title("NFL Origination Model — artifact browser")
    runs = list_runs()
    if not runs:
        st.info(
            "No runs found under artifacts/. Run the demo: "
            "`uv run nfl-origination demo --config configs/demo.yaml --offline`"
        )
        return
    labels = [r[0] for r in runs]
    choice = st.sidebar.selectbox("Run", labels)
    run_dir = dict(runs)[choice]
    manifest = _read_json(run_dir / "manifest.json") or {}
    banner(manifest)
    page = st.sidebar.radio("Page", ["Slate", "Game detail", "Evaluation", "Run audit"])
    if page == "Slate":
        page_slate(run_dir, manifest)
    elif page == "Game detail":
        page_game(run_dir, manifest)
    elif page == "Evaluation":
        page_evaluation(run_dir, manifest)
    else:
        page_audit(run_dir, manifest)


if __name__ == "__main__":
    main()
else:
    main()
