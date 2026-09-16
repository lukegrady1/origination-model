# Architecture

The pipeline follows `V1_SPEC.md` section 4. Every arrow below is a function with a typed
interface; every box on the right is a Parquet/JSON artifact under `data/` or `artifacts/`.

```text
nflverse assets ──ingest()──▶ data/raw/<dataset>/<season>/<sha256>.<ext> + entry.json (immutable)
                                   │
                    normalize_sources() / validate_normalized()
                                   ▼
        data/normalized/{games,results,team_games,labels,untimed_reference}.parquet + exclusions.csv
                                   │
                      aggregate_team_games() ─▶ build_features(policy)
                                   ▼
                     data/features/features_<key>.parquet   (two perspective rows per game)
                                   │
              chronological_fits() → fit_residual_params() → predict_distribution()
                                   ▼
   artifacts/runs/<run_id>/{predictions_*.parquet, distributions_*.parquet, bundles/, metrics.json,
                           decision_record.json, manifest.json, report/}
                                   │
   odds CSV ──import_odds()──▶ data/user/odds.parquet ──select_quotes()──▶ compare / paper_backtest
                                                            (saved predictions only)
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | Strict Pydantic config (unknown keys fail), dotted overrides, config hash |
| `schemas.py` | Versioned artifact contracts, feature allowlists, market-column quarantine |
| `provenance.py` | Hashes, git/dependency capture, `RunManifest`, run registry (`latest-*`) |
| `data/download.py` | nflverse adapter: checksummed, atomic, retried downloads; offline mode |
| `data/normalize.py` | Schedule/PBP → canonical tables; franchise aliases; DST-aware kickoff UTC |
| `data/validate.py` | Season coverage audit and exclusion table |
| `features/aggregate.py` | Play filter and per-team-game numerators/denominators |
| `features/rolling.py` | Recency weights, opportunity-weighted shrinkage, league priors |
| `features/asof.py` | Availability policy (`historical_reconstruction` / `recorded_asof`) |
| `features/builder.py` | As-of feature rows, labels, provenance columns, feature dictionary |
| `models/ridge.py`, `models/baseline.py` | M1 shared Ridge with fold-local preprocessing; B0 |
| `models/distribution.py` | Residual fit, deterministic joint PMF, margins/totals, intervals, CRPS |
| `models/bundle.py` | JSON model bundles, `fit_bundle`-style helpers, `predict_game` |
| `pricing/odds.py`, `pricing/markets.py`, `pricing/fair_lines.py` | Odds arithmetic, half-point settlement, fair lines |
| `evaluation/*` | Splits, per-game metrics, block bootstrap, Markdown/PNG report |
| `market/*` | Odds import, as-of quote selection, comparison, paper settlement and CLV |
| `experiment.py` | Backtest, forecast-bundle fit, slate prediction orchestration |
| `protocol.py` | Frozen protocol checksum and holdout gate |
| `cli.py` | Typer entry point `nfl-origination` |
| `dashboard/app.py` | Streamlit artifact browser |
| `synthetic.py`, `fixtures.py` | Deterministic synthetic fixtures and the mini raw fixture |

## Boundaries enforced in code

- `models/` and `features/` do not import `market/` (checked by reading imports; the market
  package docstring restates it).
- `schemas.assert_no_market_columns` runs on every frame entering `build_features`, the
  preprocessor, and `predict_slate`.
- `schemas.assert_feature_allowlist` runs at fit time; the allowlist is the only path to `X`.
- Labels live in `labels.parquet` and are joined only inside `fit_score_model` and evaluation.
- `pricing` takes a `MarginalScoreDistribution` and a `MarketSpec`; it never sees football data.
- The dashboard reads artifacts only; there is no training on refresh.

## Run manifests

Each run writes `manifest.json` with run ID, UTC creation time, git commit and dirty flag,
Python and dependency versions, seed, full resolved config and hash, source file hashes, schema
and feature versions, model ID, training cutoff, forecast policy, data mode, evaluation mode,
output hashes and notes. Rerunning the same config on the same cache reproduces predictions to
1e-8 in probability and metrics to 1e-6 (tested in `tests/test_end_to_end_demo.py`).

## Decisions made during implementation (routine, documented)

- The repository root is the project root (the spec's `nfl-origination/` folder).
- Regular-season games only, everywhere: history, labels, evaluation (`data.game_type: REG`).
- Rest days use the prior same-season game whose kickoff is before the cutoff (the game has
  happened even when its statistics are not yet eligible); fractional days, then clipped to
  [3, 14]; season openers get 7 days plus the missing-rest indicator.
- Model bundles are JSON (no pickle), so loading never executes serialized code.
- Alpha selection pools all development games (game-weighted) rather than averaging season
  means; both are in the decision record.
- A line whose conditional win probability is undefined (all mass on a push) is treated as
  maximally unbalanced in the fair-line search; a point-mass 24–21 therefore prices −3.5/44.5.
- Games without play-by-play would keep points and schedule information with null efficiency
  columns; none occurred in 2010–2025.
- League priors are as-of inputs: they use only prior-season rows eligible at the cutoff and are
  cached per season only when every prior row is eligible (always true in historical
  reconstruction inside a season). Their game IDs and observation times are in provenance.
- A team-game row counts as observed only when both its schedule row and its play-by-play were
  observed; rest uses observed schedule rows; labels become available at the later of
  `kickoff + 48h` and their observation in `recorded_asof`.
- Forecast horizons are explicit: live (generation time, `custom_horizon`), `--as-of` (past
  research time), or `--reconstruct-standard-horizon` (kickoff − 24h, only after it has passed).
- Bundles store a feature/policy contract; forecasting refuses a bundle whose contract differs
  from the current configuration or whose training data mode differs from the feature rows.
- The frozen protocol digests the research-relevant source tree (excluding dashboard, CLI and
  report rendering) and `uv.lock`; protocols frozen before this exist are refused for reruns.
