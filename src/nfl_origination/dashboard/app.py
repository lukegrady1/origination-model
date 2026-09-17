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
    for p in os.environ.get(
        "NFL_ORIGINATION_ARTIFACTS",
        "artifacts:artifacts/demo:artifacts/v2/research:artifacts/v2/demo/research",
    ).split(":")
]
# Prospective epochs live under <root>/prospective/; discovered separately from runs.
PROSPECTIVE_ROOTS = [
    Path(p)
    for p in os.environ.get("NFL_ORIGINATION_PROSPECTIVE", "artifacts/v2:artifacts/v2/demo").split(
        ":"
    )
]

# Human-readable labels and tooltips for raw field names (never a generic "verified" badge).
LABELS: dict[str, tuple[str, str]] = {
    "fair_home_handicap": (
        "Fair home spread (pts)",
        "Half-point line closest to 50/50 from the model PMF",
    ),
    "mean_home_handicap": ("Mean home spread (pts)", "Negative expected margin; not a 50/50 line"),
    "fair_total": ("Fair total (pts)", "Half-point total closest to 50/50 from the model PMF"),
    "mean_total": ("Expected total (pts)", "Mean of the model total distribution"),
    "p_home_win": ("P(home win)", "Three-way probability, ties separate"),
    "p_tie": ("P(tie)", "Model tie probability; the rounded-normal V1 model overstates ties"),
    "p_away_win": ("P(away win)", "Three-way probability"),
    "score_mae": ("Score MAE (pts)", "Mean absolute team-score error, points"),
    "margin_mae": ("Margin MAE (pts)", "Points"),
    "total_mae": ("Total MAE (pts)", "Points"),
    "logloss_3way": ("3-way log loss", "Unitless probability loss, lower is better"),
    "brier_3way": ("3-way Brier", "Unitless, summed over classes"),
    "crps_margin": ("Margin CRPS (pts)", "Points; lower is better"),
    "crps_total": ("Total CRPS (pts)", "Points; lower is better"),
    "margin_cover_80": ("80% margin coverage", "Share of games inside the 80% interval"),
    "cutoff_utc": ("Information cutoff (UTC)", "Last time any input could be observed"),
    "kickoff_utc": ("Kickoff (UTC)", ""),
    "data_quality_flags": ("Warnings", "Missing-history or observation flags"),
}


def label(col: str) -> str:
    return LABELS.get(col, (col.replace("_", " "), ""))[0]


def pct(v: float | None) -> str:
    return "unavailable" if v is None or not np.isfinite(v) else f"{100 * v:.1f}%"


def signed(v: float | None) -> str:
    return "unavailable" if v is None or not np.isfinite(v) else f"{v:+.1f}"


def evidence_badge(kind: str, *, synthetic: bool, evidence_mode: str | None = None) -> str:
    """The persistent evidence label. Retrospective work is never called verified."""
    if synthetic:
        return "SYNTHETIC — generated fixtures, not evidence"
    if evidence_mode == "observed_prospective" or kind in ("forecast_prospective",):
        return "PROSPECTIVE LOCAL RECORDING — locally auditable, not independently certified"
    if kind == "legacy":
        return "LEGACY OBSERVATION METADATA — V1 collector timestamps, not independent evidence"
    return "RETROSPECTIVE RECONSTRUCTION — historical files as observed later"


def list_epochs() -> list[tuple[str, Path, Path]]:
    """(label, artifacts root, epoch json path) for every discovered epoch."""
    out = []
    for root in PROSPECTIVE_ROOTS:
        edir = root / "prospective" / "epochs"
        if not edir.exists():
            continue
        for path in sorted(edir.glob("epoch-*.json"), reverse=True):
            out.append((f"{root.as_posix()} / {path.stem}", root, path))
    return out


def page_prospective(root: Path, epoch_path: Path) -> None:
    epoch = _read_json(epoch_path) or {}
    pid = epoch.get("protocol_id", "")
    st.warning(
        evidence_badge(
            "forecast_prospective",
            synthetic=bool(epoch.get("synthetic")),
            evidence_mode=epoch.get("evidence_mode"),
        )
    )
    st.subheader("Selected protocol")
    st.caption(
        f"epoch `{pid}` · activated {epoch.get('activated_at_utc')} · champion `{epoch.get('champion', {}).get('model_id')}` "
        f"· challenger `{(epoch.get('challenger') or {}).get('model_id') or 'none'}` · code digest `{str(epoch.get('code_digest'))[:12]}` "
        f"· horizon {epoch.get('horizon', {}).get('target_hours_before_kickoff')}h ±{epoch.get('horizon', {}).get('window_minutes')} min"
    )
    ledger_dir = root / "prospective" / pid
    reports = (
        sorted((ledger_dir / "reports").glob("prospective_*.json"))
        if (ledger_dir / "reports").exists()
        else []
    )
    report = _read_json(reports[-1]) if reports else None
    if report is None:
        st.info("No prospective report yet for this epoch. Run `report-prospective`.")
    else:
        cov = report["coverage"]
        cols = st.columns(6)
        for c, (k, v) in zip(
            cols,
            [
                ("Eligible", cov["eligible_forecasts"]),
                ("Missed window", cov["missed_forecast_window"]),
                ("Pending", cov["pending"]),
                ("Settled matched", cov["settled_matched"]),
                ("Collection failures", cov["collection_failures"]),
                ("Integrity", report["integrity"]),
            ],
            strict=True,
        ):
            c.metric(k, v)
        st.caption(
            f"results as of {report['results_as_of_utc']} · next review gate: {report['review_gate']['status']} ({report['review_gate']['settled_eligible_games']}/{report['review_gate']['planned_review_games']} settled eligible games, {report['review_gate']['season_week_blocks']}/{report['review_gate']['planned_review_blocks']} blocks)"
        )
        st.subheader("Model metrics on settled eligible games")
        st.dataframe(pd.DataFrame(report["model_metrics"]).T, use_container_width=True)
        st.subheader("Paper ledger (frozen policy; not realized profit)")
        if report["paper"]:
            for role, summary in report["paper"].items():
                st.write(
                    f"**{role}** — decisions {summary.get('n_decisions')} · counts {summary.get('counts')} · settled stake {summary.get('settled_non_void_stake')} · net {summary.get('net_profit_units')} · ROI {summary.get('roi_on_settled_stake')} · max drawdown {summary.get('max_drawdown_units')} · ROI bootstrap {summary.get('roi_bootstrap') if isinstance(summary.get('roi_bootstrap'), str) else 'interval available'} · line CLV spread n={summary.get('line_clv_spread', {}).get('n')} total n={summary.get('line_clv_total', {}).get('n')} · price CLV n={summary.get('price_clv', {}).get('n')}"
                )
        else:
            st.write("no paper decisions yet")
        st.subheader("Market benchmark (matched games, one orientation)")
        mb = report["market_benchmark"]
        if isinstance(mb, dict) and mb and "status" not in mb:
            st.dataframe(
                pd.DataFrame(mb).T.drop(columns=["paired_model_minus_market"], errors="ignore"),
                use_container_width=True,
            )
        else:
            st.write("unavailable: no eligible timestamped odds")
    st.subheader("Committed forecasts")
    rows = []
    fdir = ledger_dir / "forecasts"
    if fdir.exists():
        for m in sorted(fdir.glob("fc-*.manifest.json")):
            man = _read_json(m) or {}
            rec = _read_json(m.with_name(m.name.replace(".manifest.json", ".json"))) or {}
            rows.append(
                {
                    "game": f"{rec.get('away_team')} @ {rec.get('home_team')}",
                    "role": rec.get("role"),
                    "model": rec.get("model_id"),
                    "kickoff": rec.get("kickoff_at_forecast_utc"),
                    "information cutoff": rec.get("information_cutoff_utc"),
                    "committed": man.get("committed_at_utc"),
                    "actual horizon (h)": round(man.get("actual_horizon_hours", 0), 2),
                    "eligible": man.get("eligible_for_scoring"),
                    "reason": man.get("eligibility_reason"),
                    "source freshness": rec.get("max_source_observed_utc"),
                    "fair spread": signed(rec.get("fair_home_handicap")),
                    "fair total": rec.get("fair_total"),
                    "P(home)": pct(rec.get("p_home_win")),
                    "P(tie)": pct(rec.get("p_tie")),
                    "flags": ",".join(rec.get("quality_flags", [])),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("no committed forecasts")
    st.subheader("Paper signals under frozen policy")
    ddir = ledger_dir / "decisions"
    drows = []
    if ddir.exists():
        for d in sorted(ddir.glob("fc-*.json")):
            dec = _read_json(d) or {}
            sel = dec.get("selection") or {}
            drows.append(
                {
                    "game": dec.get("game_id"),
                    "role": dec.get("role"),
                    "status": dec.get("status"),
                    "reason": dec.get("reason"),
                    "market": sel.get("market"),
                    "selection": sel.get("selection"),
                    "line": sel.get("line"),
                    "price": sel.get("decimal_odds"),
                    "EV": sel.get("ev"),
                    "decision time": dec.get("decision_time_utc"),
                }
            )
    if drows:
        st.dataframe(pd.DataFrame(drows), use_container_width=True, hide_index=True)
        st.caption(
            "'no odds' (market_unavailable) is distinct from 'no model prediction' (a missing forecast)."
        )
    st.subheader("Events: collection failures, missed windows, reschedules")
    ev = ledger_dir / "events.jsonl"
    if ev.exists():
        events = [json.loads(line) for line in ev.read_text().splitlines() if line.strip()]
        shown = [e for e in events if e.get("event") != "tick"]
        st.dataframe(
            pd.DataFrame(shown) if shown else pd.DataFrame([{"event": "none"}]),
            use_container_width=True,
            hide_index=True,
        )
        ticks = [e for e in events if e.get("event") == "tick"]
        if ticks:
            st.caption(
                f"latest tick {ticks[-1].get('recorded_at_utc')} · collection {ticks[-1].get('collection')}"
            )
    st.subheader("Audit: ids, hashes, lineage")
    st.json(
        {
            "protocol_id": pid,
            "payload_hash": epoch.get("payload_hash"),
            "champion": epoch.get("champion"),
            "challenger": epoch.get("challenger"),
            "code_digest": epoch.get("code_digest"),
            "lock_digest": epoch.get("lock_digest"),
            "config_hash": epoch.get("config_hash"),
        }
    )
    st.caption(
        "Hashes prove local content consistency only. This is not tamper-proof or independently certified; synthetic and real epochs are never merged."
    )


def page_research(run_dir: Path, manifest: dict) -> None:
    metrics = _read_json(run_dir / "metrics.json") or {}
    st.warning(
        evidence_badge(
            "research",
            synthetic=bool(
                manifest.get("config", {}).get("v2", {}).get("evidence", {}).get("synthetic")
            ),
        )
    )
    st.subheader("Candidates on identical development games")
    dev = metrics.get("development_metrics", {})
    if dev:
        df = pd.DataFrame(dev).T
        keep = [
            c
            for c in (
                "n_games",
                "crps_margin",
                "crps_total",
                "logloss_3way",
                "brier_3way",
                "score_mae",
                "margin_mae",
                "total_mae",
                "margin_cover_80",
                "total_cover_80",
                "predicted_tie_rate",
                "observed_tie_rate",
                "n_ties",
                "key_gap",
            )
            if c in df.columns
        ]
        df = df[keep].rename(columns={c: label(c) for c in keep})
        st.dataframe(df, use_container_width=True)
    st.subheader("Decision")
    st.json(metrics.get("decision", {}))
    st.subheader("Retrospective checks (2024–2025 were previously inspected; not for tuning)")
    st.json(metrics.get("retrospective_check_metrics", {}))
    st.subheader("Paired differences vs reconstructed V1 (development, block bootstrap)")
    st.json(metrics.get("paired_vs_V1_development", {}))
    st.subheader("Exclusions")
    st.json(metrics.get("exclusions", []))


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
    st.caption("Read-only: refreshing this page never ingests, fits or collects anything.")
    runs = list_runs()
    epochs = list_epochs()
    if not runs and not epochs:
        st.info(
            "No runs found under artifacts/. Run the demo: "
            "`uv run nfl-origination demo --config configs/demo.yaml --offline`"
        )
        return
    page = st.sidebar.radio(
        "Page", ["Slate", "Game detail", "Evaluation", "Prospective", "Run audit"]
    )
    if page == "Prospective":
        if not epochs:
            st.info("No prospective epochs found. Run `demo-v2` or `freeze-prospective`.")
            return
        choice = st.sidebar.selectbox("Epoch", [e[0] for e in epochs])
        _label, root, epoch_path = next(e for e in epochs if e[0] == choice)
        page_prospective(root, epoch_path)
        return
    if not runs:
        st.info("No runs found; only prospective epochs are available.")
        return
    labels = [r[0] for r in runs]
    choice = st.sidebar.selectbox("Run", labels)
    run_dir = dict(runs)[choice]
    manifest = _read_json(run_dir / "manifest.json") or {}
    banner(manifest)
    st.warning(
        evidence_badge(
            manifest.get("run_kind", ""),
            synthetic=any("SYNTHETIC" in n for n in manifest.get("notes", []))
            or manifest.get("run_kind") in ("demo", "v2_demo"),
        )
    )
    if manifest.get("run_kind") == "v2_research":
        if page == "Run audit":
            page_audit(run_dir, manifest)
        else:
            page_research(run_dir, manifest)
        return
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
