# Implementation status

Planning authority: `V1_SPEC.md` v1.0 (2026-09-16). This log records completed requirements,
commands run, failures, and deviations for review.

## Checklist mapped to acceptance criteria

| ID | Criterion (short) | Status |
|---|---|---|
| A01 | Fresh checkout installs from lockfile, offline demo runs | done |
| A02 | Real coverage audited, exclusions exported | done (2022 gap as coverage note; see deviations) |
| A03 | Features/labels separated, as-of and future-mutation tests, data mode on reports | done (R1 fix extended recorded_asof to priors, labels, residuals, rest) |
| A04 | B0, M1, ablation on identical grouped chronological splits | done |
| A05 | Residuals from prior chronological predictions; PMF/support/covariance checks | done |
| A06 | Pricing fixtures exact | done |
| A07 | Frozen 2025 report complete | done; code-state provenance of the original freeze not reconstructible (R3) |
| A08 | Fixed-input reproducibility within tolerances | done; code/lock digests now frozen for future runs |
| A09 | CLI forecast exports; dashboard shows artifacts and handles missing odds | done after R4/R5 (explicit horizons, bundle contracts) |
| A10 | Market module rejects ineligible odds, settles synthetic fixtures, paid data absence disclosed | done after R2/R6 |
| A11 | Ruff, mypy, pytest, offline CI, real-source smoke | done locally; CI unobserved on GitHub |
| A12 | Docs complete | done |

## Milestones

| Step | Output | Verification run | Result |
|---|---|---|---|
| 1 Scaffold | `pyproject.toml`, `uv.lock` (75 packages), configs, CLI, CLAUDE.md, fixtures | `uv sync --frozen`; `ruff check`; `mypy` | pass |
| 2 Data contracts | `data/download.py`, `normalize.py`, `validate.py`, schemas, provenance | `ingest --seasons 2010:2026` (317 MB, 86 s); `validate-data --seasons 2010:2025`; mocked retry/corruption tests | 2010–2025 complete except one 2022 game absent from source |
| 3 Features | aggregation, rolling shrinkage, as-of policy, builder, dictionary | hand-worked fixture tests; leakage sentinel; 8,350 rows in 7.4 s | pass |
| 4 Baseline | B0, folds, residual estimation, JSON bundles | development run; bundle round-trip test | pass |
| 5 Candidate | Ridge pipeline, alpha grid, OOF residual pools | `backtest --config configs/development.yaml` (72 s) → alpha 100 | decision record saved |
| 6 Distribution/pricing | quadrature PMF, fair lines, vig illustration | `tests/unit/test_distribution.py`, `test_pricing.py`; matches SciPy to 1e-9 | pass |
| 7 Evaluation | metrics, ablation, bootstrap, 2024 confirmation | `backtest --config configs/confirmation.yaml` (5 s) | pass, no protocol change |
| 8 Market contracts | CSV import, as-of selection, comparison, settlement, CLV | `tests/unit/test_odds_import.py`; demo paper backtest on synthetic odds | pass; real odds unavailable |
| 9 Dashboard/report | Streamlit pages, Markdown/PNG/JSON/CSV report | `streamlit run` health check; `AppTest` page smoke; empty-state test | pass |
| 10 Final run | `freeze-protocol`, holdout once, `fit --through-season 2025`, 2026 forecasts | see `docs/experiment_protocol.md` | done |

## Commands run (real data path)

```bash
uv sync --frozen
uv run nfl-origination ingest --seasons 2010:2026            # 17 PBP files + schedule
uv run nfl-origination validate-data --seasons 2010:2025 --offline
uv run nfl-origination build-features --config configs/v1.yaml --offline
uv run nfl-origination backtest --config configs/development.yaml --offline
uv run nfl-origination backtest --config configs/confirmation.yaml --offline
uv run nfl-origination freeze-protocol --config configs/holdout.yaml
uv run nfl-origination backtest --config configs/holdout.yaml --offline
uv run nfl-origination report --run latest-holdout --config configs/holdout.yaml
uv run nfl-origination fit --through-season 2025 --config configs/v1.yaml --offline
uv run nfl-origination predict --season 2026 --week 3 --as-of 2026-09-16T12:00:00Z --config configs/v1.yaml --offline
uv run nfl-origination predict --season 2026 --week 2 --config configs/v1.yaml --offline
uv run nfl-origination compare --run latest-forecast          # unavailable: no eligible timestamped odds
uv run nfl-origination paper-backtest --run latest-holdout --config configs/holdout.yaml   # unavailable
uv run nfl-origination demo --config configs/demo.yaml --offline
uv run ruff check . && uv run ruff format --check . && uv run mypy src/nfl_origination
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run pytest -q             # 109 passed
```

## Failures encountered and fixed

- nflverse PBP contains placeholder rows (`*** play under review ***`) with `sack=1` but no
  dropback flag; the semantic invariant is now checked on real scrimmage plays only.
- `align_labels` was called twice in the fit path (duplicate `points` column); made idempotent.
- `recorded_asof` eligibility mask mutated a read-only NumPy view; fixed to allocate.
- Forecast slate validation failed with three models in one frame; outputs are now per model.
  The failed run directory was removed and failed runs now clean up automatically.
- CLI config errors surfaced as exit code 1; config loading now maps to exit code 2.
- pandas 3 uses microsecond UTC timestamps; a test asserting `ns` resolution was relaxed.

## Deviations from the spec (documented, none statistical)

- Repository root replaces the `nfl-origination/` folder; `configs/confirmation.yaml` added so
  the 2024 confirmation is its own run kind; `model.ablation_selected_alpha` added to the config
  contract so the ablation's frozen alpha is explicit.
- The missing 2022 game cannot appear in the exclusion table by game ID (it is absent from the
  source); it is reported in coverage notes and report tables instead.
- Bundles are JSON rather than pickled objects (routine engineering choice; safer loading).
- The undefined conditional probability at an all-push line is treated as maximally unbalanced
  in the fair-line search (documented in `docs/architecture.md`).

## Correctness pass after the implementation review (2026-09-16)

Seven findings (R1–R7) were repaired without changing the architecture; see
`docs/review_packet.md` (follow-up table) and `docs/experiment_protocol.md`. Commands run:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/nfl_origination
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run pytest -q                    # 132 tests incl. 23 regressions
uv run nfl-origination fit --through-season 2025 --config configs/v1.yaml --offline   # bundles refit with contracts
uv run nfl-origination predict --season 2026 --week 3 --config configs/v1.yaml --offline                 # live
uv run nfl-origination predict --season 2026 --week 3 --as-of 2026-09-16T12:00:00Z --config configs/v1.yaml --offline
uv run nfl-origination predict --season 2026 --week 1 --reconstruct-standard-horizon --config configs/v1.yaml --offline
uv run nfl-origination backtest --config configs/holdout.yaml --offline --rerun-reason "gate check"   # refused (legacy protocol)
```

The holdout was not rerun and its artifacts are unchanged.

Follow-up review (F1 stale caches, F2 unknown observation times): feature/schema version 2,
validated cache loading with invalidation records, mandatory availability fields at every
boundary, NaT-safe observation-time combination. Commands: `build-features` (new cache key
`features_b023b9b97005b5dd`), `fit --through-season 2025` (`fit-20260916T185415Z-b671ea`),
live week 3 and week 1 reconstruction forecasts regenerated. Tests: 142 pass with network
blocked (10 new in `tests/unit/test_followup_regressions.py`).

## Unresolved blockers / open questions for planning

- No timestamped odds source; market comparison and ROI remain "unavailable" on real data.
- Key-number behavior is poor by design of the V1 distribution; a discrete key-number-aware
  model is the obvious next planning decision.
- `recorded_asof` needs prospective snapshots accumulated over time before it can back a
  certified backtest.

## Hardware and runtimes (measured)

Apple M5, 10 cores, 24 GB, macOS 26.6.2, Python 3.12.14, uv 0.12.15. Ingest 86 s (network
bound); normalize+aggregate 0.5 s; features 7.4 s; development backtest 72 s; confirmation and
holdout 4.7 s each; demo 37 s; full pytest ≈ 3 min (dominated by end-to-end synthetic runs).
