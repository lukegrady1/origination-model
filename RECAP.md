# Recap: V1 NFL Origination Model build (2026-09-16)

This is a plain account of what was done in one session against `V1_SPEC.md`, what was
implemented, and what the real-data runs returned. Nothing here is a claim beyond the measured
artifacts committed under `reports/final/`.

## 1. Setup

- Machine had no `uv` and only Python 3.9. Installed `uv` 0.12.15 and CPython 3.12.14.
- Created the project (`pyproject.toml`, committed `uv.lock`, 75 packages). Key versions:
  numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 25.0.1, duckdb 1.5.5,
  pydantic 2.13.5, typer 0.27.2, streamlit 1.64.0, matplotlib 3.11.2.
- Downloaded nflverse `schedules/games.csv` and `pbp/play_by_play_<season>.parquet` for
  2010–2026 (317 MB, 86 s) into a checksummed immutable cache under `data/raw/` (gitignored).
- Inspected the nflverse schemas before building features and confirmed the semantic mapping:
  `pass` = dropback including sacks and scrambles, `rush` = designed rush only, `no_play`,
  `qb_kneel`, `qb_spike`, `two_point_attempt`, `qtr` (5 = OT). Schedule kickoffs are Eastern
  `gameday` + `gametime`, converted with `America/New_York` (DST-aware). Franchise aliases
  STL→LA, SD→LAC, OAK→LV mapped explicitly.

## 2. What was implemented (package `src/nfl_origination/`)

| Area | Files | What it does |
|---|---|---|
| Config | `config.py` | Strict Pydantic config mirroring spec section 20; unknown keys fail; dotted CLI overrides; config hash. Added `run.kind`, `model.ablation_selected_alpha`, `data.source: synthetic`. |
| Contracts | `schemas.py` | Versioned artifact schemas (games, results, team_games, features, labels, predictions, odds, paper_bets), feature allowlists (full and EPA-free), market-column quarantine, dtype/key validation. |
| Provenance | `provenance.py` | SHA-256 hashing of files/frames/JSON, git commit + dirty flag, dependency versions, `RunManifest`, run registry with `latest-<kind>` aliases. |
| Data | `data/download.py`, `normalize.py`, `validate.py`, `storage.py` | Atomic, retried, checksummed downloads; `--offline` prohibits network; failed refresh keeps last snapshot; first-observed timestamps for `recorded_asof`. Schedule → games/results/`untimed_reference` (betting fields quarantined). PBP → canonical plays with semantic invariant checks. Season coverage audit and exclusion table. Parquet helpers and read-only DuckDB views. |
| Features | `features/aggregate.py`, `rolling.py`, `asof.py`, `builder.py` | Play filter per spec section 8; per-team-game numerators and denominators; recency weights `2^(-j/8)`, offseason ×0.5, last 16 games; shrinkage `(N + k·r0)/(D + k)` with k = 200/100/4; league priors from the two preceding seasons; as-of policy (kickoff + 48h ≤ cutoff; plus observed ≤ cutoff in recorded_asof); two perspective rows per game with venue, rest difference, history counts, cold-start flags, provenance columns; separate labels table; generated feature dictionary. |
| Models | `models/baseline.py`, `ridge.py`, `distribution.py`, `bundle.py` | B0 league mean + home/away offset. M1 shared Ridge (intercept, row weight 0.5) with training-only median imputer, missingness indicators, StandardScaler; >5% unexpected nulls fails. Residual mean/covariance with `0.9·S + 0.1·diag(S)` and 1e-6 floor from ≥500 prior out-of-fold games. Deterministic joint PMF by Gauss–Legendre quadrature over half-point cells (zero absorbs the negative tail), adaptive support 100→250, tail/negative-mass diagnostics, margin/total PMFs, equal-tail intervals, CRPS. JSON model bundles (no pickle) with training IDs, cutoff, residual game IDs, versions. |
| Pricing | `pricing/odds.py`, `markets.py`, `fair_lines.py` | American/decimal conversion with validation, no-vig pairs, EV, 4% overround illustration. Integer half-point lines (quarter lines rejected). Win/push/loss probabilities and settlement for moneyline (two-way tie-void), spread, total. Fair handicap/total grid search with the spec's tie-breaks. |
| Evaluation | `evaluation/splits.py`, `metrics.py`, `bootstrap.py`, `report.py` | Grouped chronological folds; all required measures; reliability bins with sparse flags; key-number diagnostics; 1,000-replicate season-week block bootstrap for paired differences; Markdown report + PNG charts + metrics JSON + prediction CSV. |
| Market (optional) | `market/import_csv.py`, `asof.py`, `compare.py`, `settle.py` | Canonical odds CSV import with per-row rejection reasons; timestamp-safe quote selection (≤60 min old, updated ≤ snapshot, complete pairs); model-vs-market comparison; paper backtest with frozen policy (EV ≥ 0.03, flat 1 unit, one selection per game), settlement, ROI, drawdown, ROI bootstrap, line and price CLV against a closing proxy. |
| Orchestration | `experiment.py`, `protocol.py` | Dataset preparation with feature cache; chronological backtest (fit per season on prior seasons, residual pool from prior OOF seasons, price, score, select alpha, bootstrap); frozen-protocol checksum and holdout gate (refuses unfrozen/mismatched; reruns need `--rerun-reason`); forecast-bundle fit; slate prediction with `--as-of` and `custom_horizon` labeling. |
| CLI | `cli.py` (`nfl-origination`) | `demo`, `ingest`, `validate-data`, `build-features`, `backtest`, `freeze-protocol`, `report`, `fit`, `predict`, `import-odds`, `compare`, `paper-backtest`. Exit codes 0/2/3/4 as specified. |
| Dashboard | `dashboard/app.py` | Streamlit, artifact-only: Slate, Game detail, Evaluation, Run audit; synthetic banner; "unavailable: no eligible timestamped odds" for missing market panels. |
| Fixtures | `synthetic.py`, `fixtures.py`, `tests/fixtures/synthetic/` | Deterministic synthetic multi-season generator (32 teams, 17 weeks); committed mini nflverse-shaped schedule + PBP CSV covering every filter rule; synthetic odds CSV (8,711 rows) with deliberate rejects. |
| Tests | `tests/` (109 tests) | Play filtering, rolling shrinkage, temporal joins (boundary, observation, Thursday game, DST, reschedule), feature isolation, leakage sentinel, fold grouping and scaler isolation, perspective swaps, PMF invariants vs SciPy, pricing fixtures (24–21, 20–20, conversions, EV, vig), odds import/selection failure paths, metrics/bootstrap, config/exit codes, mocked downloads, bundle round-trip, ingestion-to-aggregation integration, dashboard AppTest smoke, end-to-end demo CLI, fixed-input reproducibility (1e-8 / 1e-6), holdout gate. |
| CI / docs | `.github/workflows/ci.yml`, `CLAUDE.md`, `README.md`, `docs/*` | Offline CI (network blocked in tests via env var), ruff, mypy, pytest, demo. Docs: architecture, data sources, data dictionary, model card, experiment protocol, implementation status, review packet. |

Configs: `configs/v1.yaml`, `development.yaml`, `confirmation.yaml`, `holdout.yaml`,
`demo.yaml` (synthetic seasons 2010–2020, dev 2018, confirmation 2019, holdout 2020).

## 3. What the real-data runs returned

Data audit (`validate-data --seasons 2010:2025`): 4,175 completed regular-season games,
737,117 plays, every game has play-by-play. 2022 has 271 of 272 games because the canceled
Week 17 BUF–CIN game is absent from the source; it is reported as a coverage note.

Development backtest (2018–2023, 1,583 games, 72 s):

| Alpha | Pooled score MAE (full) | Pooled score MAE (EPA-free) |
|---|---|---|
| 1 | 7.5664 | 7.5774 |
| 10 | 7.5657 | 7.5773 |
| 100 | 7.5660 | 7.5772 |
| 1000 | 7.5788 | 7.6131 |

Selection rule (min MAE, ties within 0.01 → larger alpha) chose **alpha 100** for both feature
sets. M1 vs B0 on development: score MAE 7.566 vs 8.041 (−0.475, 95% CI −0.589 to −0.365);
3-way log loss 0.677 vs 0.730.

Confirmation 2024 (272 games): M1 score MAE 7.352 vs B0 7.828 (−0.476, CI −0.736 to −0.235);
log loss 0.652 vs 0.721. No methodological change was made after seeing it.

Protocol frozen at 2026-09-16T17:50:27Z (checksum `a31f98d0…`), then the **2025 holdout ran
once** (`holdout-20260916T175029Z-4385c4`, 272 games):

| Metric | B0 | M1 alpha 100 | M1 EPA-free | M1 − B0 (95% block bootstrap) |
|---|---|---|---|---|
| Score MAE | 7.877 | 7.439 | 7.464 | −0.438 [−0.666, −0.218] |
| Margin MAE | 11.064 | 10.149 | 10.191 | −0.915 [−1.367, −0.474] |
| Total MAE | 10.971 | 10.616 | 10.650 | −0.355 [−0.695, +0.018] |
| 3-way log loss | 0.7295 | 0.6706 | 0.6704 | −0.059 [−0.086, −0.032] |
| 3-way Brier | 0.502 | 0.448 | 0.447 | −0.054 [−0.079, −0.030] |
| CRPS margin | 7.987 | 7.287 | 7.308 | −0.700 [−1.001, −0.390] |
| CRPS total | 7.797 | 7.589 | 7.588 | −0.207 [−0.387, −0.018] |
| Margin coverage 50/80/95% | 0.566/0.776/0.949 | 0.563/0.801/0.956 | 0.548/0.813/0.952 | |
| Predicted vs observed tie rate | 0.028 vs 0.004 | 0.028 vs 0.004 | 0.028 vs 0.004 | |

Key numbers (holdout, M1): P(margin = 3) predicted 0.028 vs observed 0.085; P(margin = −3)
0.027 vs 0.066; P(margin = ±7) ≈ 0.025 vs 0.048; P(total even) 0.500 vs 0.445. The rounded
normal under-weights 3 and 7 and over-predicts ties about seven-fold, as the spec anticipated.

Recommendation written in the model card: M1 alpha 100 is the preferred V1 research model.
On the holdout the bootstrap intervals exclude zero for score MAE, margin MAE, 3-way log loss,
3-way Brier, margin CRPS and (narrowly) total CRPS; total MAE improves in point estimate but its
interval includes zero, and the same holds for total metrics on confirmation. The EPA-free
ablation is within noise of M1, which suggests the displayed performance does not depend heavily
on EPA features (not proof that data-revision concerns are gone). Further modeling is needed
before any real pregame use.

Forecast path: `fit --through-season 2025` produced 2026 bundles (trained 2012–2025,
3,663 games; residuals from 2021–2025, 1,359 games). `predict --season 2026 --week 3 --as-of
2026-09-16T12:00:00Z` produced 16 priced games labeled `custom_horizon`; week 2 at the standard
horizon produced 16 games flagged `cutoff_in_future_features_from_current_cache`; week 1 as-of
now returned a valid empty slate. Largest standardized coefficients in the 2026 bundle:
`team_points_for` +1.31, `opp_points_against` +1.12, `venue_advantage` +0.99,
`team_off_dropback_epa` +0.72 points per SD.

Market: no real timestamped odds were supplied or purchased. `compare` and `paper-backtest`
on real runs return "unavailable: no eligible timestamped odds". The synthetic demo exercises
import (8,708 accepted, 3 rejected with reasons), settlement, ROI, drawdown, ROI bootstrap and
CLV; those numbers are labeled synthetic and are not evidence.

Quality gates: `ruff check`, `ruff format --check`, `mypy src/nfl_origination` all pass;
109 pytest tests pass with network blocked; offline demo runs in 37 s. Everything was committed
in one commit (`32b16ea`) with sanitized final reports under `reports/final/`.

## 4. Bugs hit and fixed during the build

- nflverse placeholder rows ("play under review") have `sack=1` with no dropback flag; the
  semantic check now applies to real scrimmage plays only.
- Labels were merged twice in the fit path; made idempotent.
- `recorded_asof` mask mutated a read-only NumPy view.
- Forecast slate with three models violated the (run, game, cutoff) key; outputs now per model.
  This and a failed-run cleanup landed after the freeze; they do not touch backtest numerics and
  the holdout was not rerun (logged in `docs/experiment_protocol.md`).
- Config errors exited with code 1 instead of 2.

## 5. Documented deviations and open items

- Repo root replaces the spec's `nfl-origination/` folder; `configs/confirmation.yaml` and
  `model.ablation_selected_alpha` were added.
- Regular-season games only, everywhere. Rest uses the prior same-season game with kickoff
  before the cutoff, fractional days clipped to [3, 14]. Bundles are JSON, not pickles.
- An all-push line is treated as maximally unbalanced in the fair-line search (a point mass
  24–21 prices −3.5 / 44.5).
- The missing 2022 game is a coverage note, not an exclusion row, because it has no game ID in
  the source.
- `recorded_asof` has no prospective snapshots yet (all first-observed times are 2026-09-16).
- CI workflow is committed but has not been observed running on GitHub (no remote configured).
- Key-number weakness is structural to the V1 distribution; a discrete key-number-aware model is
  the obvious next planning decision.

## 6. Correctness pass after the implementation review (same day)

The review at `NFL_V1_IMPLEMENTATION_REVIEW.md` reported seven findings. All were fixed without
changing the architecture, each with regression tests (`tests/unit/test_review_regressions.py`,
23 tests):

- R1: `recorded_asof` now covers league priors, PBP observation times, schedule metadata for
  rest, labels (later of kickoff+48h and observation) and residual pools; fits drop labels not
  available at fit time; feature rows flag `unobserved_inputs` and record prior provenance.
- R2: saved distributions are matched by (game, model) with exactly-one semantics and PMF
  validation; the earlier lookup could return B0's distribution under the M1 label.
- R3: the frozen protocol now digests the research source tree and `uv.lock`. The original
  protocol predates this and is refused for reruns; its holdout artifacts are preserved and the
  code-state limitation is disclosed.
- R4: bundles carry a feature/policy contract checked before forecasting; the 2026 bundles were
  refit and forecasts regenerated.
- R5: forecast horizons are explicit: live (generation time, labeled `custom_horizon`, started
  games excluded), `--as-of` (past only), `--reconstruct-standard-horizon` (only after the
  cutoff has passed). Comparison and paper backtest exclude decisions made before the model
  existed. The mislabeled week 2 slate was removed from `reports/final/`.
- R6: away-spread line CLV sign fixed (both spread sides use bet − close).
- R7: model card, README and this recap now state exactly which intervals exclude zero.

The frozen development/confirmation/holdout artifacts were not regenerated; in
`historical_reconstruction` the new availability filters select the same rows those runs used.
