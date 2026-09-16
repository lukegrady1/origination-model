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
| A03 | Met after R1 + F1/F2 fixes | `labels.parquet` separate from features; `tests/unit/test_temporal.py`, `test_models_and_leakage.py`, `test_review_regressions.py::test_r1_*`, `test_followup_regressions.py`; data mode appears in every manifest, prediction row and report. Availability fields are mandatory at every training/forecast boundary; stale caches are invalidated; unknown observation times are ineligible. `recorded_asof` still has no prospective snapshots and no cached-version selection, so it is enforced, not exercised, on real data |
| A04 | Met | Development/confirmation/holdout runs score B0, M1 and the EPA-free ablation on identical grouped folds (`evaluation/splits.py`, `test_fold_grouping_and_scaler_isolation`) |
| A05 | Met | Residual pools use only prior out-of-fold seasons (≥512 games); PMF/support/covariance checks in `tests/unit/test_distribution.py` |
| A06 | Met | `tests/unit/test_pricing.py` exact fixtures (24–21, 20–20, ±150/−200, EV 0.1454545, 1.8 zero-EV, quarter lines rejected) |
| A07 | Met with provenance caveat | `reports/final/holdout/report.md` + `metrics.json`: all required measures, baselines, bootstrap intervals, counts, limitations. The holdout's frozen protocol predates code/lockfile digests and was frozen on a dirty tree, so its exact code state is not reconstructible (R3) |
| A08 | Met with provenance caveat | `tests/test_end_to_end_demo.py::test_fixed_inputs_reproduce_within_tolerance` (1e-8 probabilities, 1e-6 metrics); code/lock digests are now part of the protocol for future runs |
| A09 | Met after R4/R5 fixes | Live `predict` uses generation time and is labeled `custom_horizon`; `--as-of` and `--reconstruct-standard-horizon` are explicit; bundles are contract-checked; dashboard renders all pages and shows "unavailable: no eligible timestamped odds" |
| A10 | Met after R2/R6 fixes | Distribution lookup keyed by (game, model); away-spread CLV sign fixed; `tests/unit/test_odds_import.py` and `test_review_regressions.py::test_r2_*`, `test_r6_*`; absence of paid data disclosed |
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

## Follow-up: implementation review findings (2026-09-16)

| Finding | Fix | Tests |
|---|---|---|
| R1 recorded_asof incomplete | `features/builder.py`: priors filtered by eligibility at the cutoff and cached per cutoff when not fully eligible, `prior_games_hash` and prior observation times in provenance, `unobserved_inputs` flag for an unobserved schedule row, rest uses observed schedule rows; `features/aggregate.py`: `first_observed_utc` = max(schedule, PBP); `build_labels` uses the later of `kickoff+lag` and observation in recorded mode; `experiment.chronological_fits` / `residual_pool` drop labels unavailable at fit time and record counts | `test_review_regressions.py::test_r1_*` (end-to-end feature/fit path, not just the mask) |
| R2 wrong model's distribution | `market/compare.load_distribution(dists, game_id, model_id)`: exact one match, PMF validated; callers pass the prediction's model; CLI prefers per-model files | `test_r2_*` |
| R3 protocol did not verify code | `protocol.py`: scoped `code_digest` (excludes dashboard, CLI, report rendering) and `lock_digest` in the checksummed payload; legacy protocols refused outright; labeled protocol files per `run.label` | `test_r3_*` |
| R4 bundle compatibility | `models/bundle.py`: `feature_contract` + `contract_hash` stored and checked (`check_bundle_compatible`); `predict_game` rejects rows from another data mode or with unobserved inputs; recorded_asof requires bundle creation before the decision time | `test_r4_*` |
| R5 future cutoffs labeled standard | `experiment.predict_slate`: live horizon = generation time (started games excluded, `custom_horizon`), explicit `--as-of` (past only), explicit `--reconstruct-standard-horizon` (only once cutoffs have passed); `model_created_utc` on forecast rows; comparison/paper backtest exclude decisions before model availability | `test_r5_*` |
| R6 away-spread CLV sign | `market/settle._line_clv` uses `bet − close` for both spread sides | `test_r6_*` (helper and full paper-backtest path) |
| R7 overstated support | `docs/model_card.md`, `RECAP.md` state exactly which intervals exclude zero | n/a |

Numerical research outputs affected: none for the committed development/confirmation/holdout
artifacts (all `historical_reconstruction`, where the new filters select the same rows). The
2026 forecast bundles were refit and the forecast slates regenerated under the new horizon
semantics.

## Follow-up review (F1, F2)

| Finding | Fix | Tests |
|---|---|---|
| F1 old cached features bypassed the corrected builder | `FEATURE_VERSION` → `"2"`, `SCHEMA_VERSION` → `2` (new cache key); `experiment._load_valid_feature_cache` validates any cache hit against the current features schema, feature version and data mode, moves an invalid file aside with an `.invalidated.json` record, and rebuilds; `require_availability_fields` makes `insufficient_warmup`/`unobserved_inputs` and the current feature version mandatory in `usable_rows`, `require_forecastable`, `fit_score_model` and `predict_game`. The 2026 bundles were refit under version 2 (`fit-20260916T185415Z-b671ea`) and the slates regenerated | `test_followup_regressions.py::test_f1_*` (cache-loading path with a seeded old-format cache and a wrong-mode cache; boundary rejection of missing fields and stale versions) |
| F2 missing PBP observation timestamps were treated as known | `aggregate_team_games` normalizes both observation columns to tz-aware UTC (naive or unparseable → `ModelValidationError`), and sets the combined time only when **both** schedule and PBP times are known; otherwise `NaT`, which `AsOfPolicy.eligible_mask` treats as ineligible. A missing PBP observation column is unknown, not known. `eligible_mask` also validates the dtype of the observation series | `test_followup_regressions.py::test_f2_*` (all four known/unknown combinations, missing column, deliberate dtype errors, aggregation-to-feature mutation test) |

Frozen research artifacts remain untouched; the version bump changes the feature cache key and
bundle versions only. Old bundles (schema 1) are refused by `ModelBundle.load`, so pre-fix
bundles can no longer be used for forecasting.

## Deviations and open items

- The canceled 2022 BUF–CIN game is missing from the source schedule entirely, so it cannot be
  listed by game ID; it is surfaced as a coverage note (271 of 272) rather than an exclusion row.
- Code changed after the freeze (forecast output layout, failed-run cleanup, and the R1–R6
  correctness pass); none touch the frozen runs' numerics and the holdout was not rerun. The
  original protocol is now unverifiable by construction and refuses reruns; see the protocol doc.
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
versus a league baseline, with paired block-bootstrap intervals excluding zero on score MAE,
margin MAE, log loss, Brier and margin CRPS (total MAE improved in point estimate only)."
