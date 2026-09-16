# Review packet

Entry point for the planning assistant. Everything below is reproducible from the CLI in
`README.md`; real-data artifacts referenced here are committed under `reports/final/`.

## What was built

A local, reproducible research application (`uv sync --frozen`, Python 3.12) that ingests
nflverse schedules and play-by-play, builds leakage-safe as-of features, fits a baseline and a
shared Ridge score model chronologically, integrates a bivariate-normal residual model into a
discrete joint score PMF, derives fair moneylines/spreads/totals with exact half-point
settlement, evaluates under a frozen protocol with block-bootstrap uncertainty, imports and
settles timestamped odds on paper, and shows saved artifacts in a Streamlit dashboard.

## Where to look

| Question | Location |
|---|---|
| Architecture and enforced boundaries | `docs/architecture.md` |
| Source verification, field mapping, coverage audit | `docs/data_sources.md` |
| Feature and artifact definitions | `docs/data_dictionary.md` |
| Frozen protocol, decisions, post-freeze changes | `docs/experiment_protocol.md`, `reports/final/frozen_protocol.json` |
| Results, limitations, recommendation | `docs/model_card.md`, `reports/final/holdout/report.md` |
| Milestone evidence, commands run, deviations | `docs/implementation_status.md` |
| Alpha decision record | `reports/final/development/decision_record.json` |
| 2026 forecast example (week 2, standard horizon) | `reports/final/forecast_2026_week2/slate.csv` |

## Acceptance criteria

| ID | Status | Evidence |
|---|---|---|
| A01 | Met | `uv sync --frozen` then `demo --offline` runs from fixtures in ~37 s; CI does the same with network blocked (`tests/conftest.py`, `.github/workflows/ci.yml`) |
| A02 | Met with note | `validate-data --seasons 2010:2025`: 16 seasons audited; one 2022 game absent from the source is reported in `coverage_report.json` notes and every report's samples table, not as an exclusion row (see deviations) |
| A03 | Met | `labels.parquet` separate from features; `tests/unit/test_temporal.py`, `test_models_and_leakage.py`; data mode appears in every manifest, prediction row and report |
| A04 | Met | Development/confirmation/holdout runs score B0, M1 and the EPA-free ablation on identical grouped folds (`evaluation/splits.py`, `test_fold_grouping_and_scaler_isolation`) |
| A05 | Met | Residual pools use only prior out-of-fold seasons (≥512 games); PMF/support/covariance checks in `tests/unit/test_distribution.py` |
| A06 | Met | `tests/unit/test_pricing.py` exact fixtures (24–21, 20–20, ±150/−200, EV 0.1454545, 1.8 zero-EV, quarter lines rejected) |
| A07 | Met | `reports/final/holdout/report.md` + `metrics.json`: all required measures, baselines, bootstrap intervals, counts, limitations |
| A08 | Met | `tests/test_end_to_end_demo.py::test_fixed_inputs_reproduce_within_tolerance` (1e-8 probabilities, 1e-6 metrics) |
| A09 | Met | `predict --season 2026 --week 3 --as-of …` exports traceable prices labeled `custom_horizon`; dashboard renders all pages and shows "unavailable: no eligible timestamped odds" (`tests/integration/test_dashboard.py`) |
| A10 | Met | `tests/unit/test_odds_import.py` rejects stale/future/incomplete/mismatched/duplicate/ambiguous quotes; synthetic settlement in the demo; absence of paid data disclosed in reports and model card |
| A11 | Met | ruff, ruff format, mypy, 109 pytest tests pass locally with network blocked; real-source smoke = `validate-data` on the 2010–2025 cache; CI workflow committed (not yet observed running on GitHub) |
| A12 | Met | README, model card, data dictionary, data sources/attribution, experiment protocol, this packet |

## Real-data results in one paragraph

On the frozen 2025 holdout (272 games) the Ridge candidate reaches team-game score MAE 7.44
versus 7.88 for the league baseline (difference −0.44, 95% block-bootstrap interval −0.67 to
−0.22), 3-way log loss 0.671 versus 0.730, and margin CRPS 7.29 versus 7.99. Development
(1,583 games) and confirmation (272) show the same direction and size. The EPA-free ablation is
within noise of the full model. Interval coverage is close to nominal (80% margin interval covers
80.1%). Key-number diagnostics show the rounded normal puts 2.8% on a 3-point margin against 8.5%
observed and predicts ties seven times too often. No market comparison or ROI was run on real
odds because none were supplied.

## Deviations and open items

- The canceled 2022 BUF–CIN game is missing from the source schedule entirely, so it cannot be
  listed by game ID; it is surfaced as a coverage note (271 of 272) rather than an exclusion row.
- Two code changes landed after the freeze (forecast output layout and failed-run cleanup); they
  do not touch backtest numerics and the holdout was not rerun. Logged in the protocol doc.
- `recorded_asof` mode is implemented and tested at the policy level, but no prospective
  snapshots exist yet: the cache's first-observed times are all 2026-09-16, so recorded-asof
  backtests correctly report ineligible data.
- The GitHub Actions workflow is committed but has not been observed passing on GitHub from this
  machine (no remote configured).
- Real timestamped odds were not purchased; the market path is verified on synthetic fixtures.

## Suggested resume claim (measured)

"Built a reproducible NFL pregame pricing pipeline with chronological validation, joint outcome
probabilities and tested spread/total settlement; on a frozen 2025 holdout (272 games) the Ridge
score model cut team-game score MAE from 7.88 to 7.44 and 3-way log loss from 0.730 to 0.671
versus a league baseline, with bootstrap intervals excluding zero."
