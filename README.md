# NFL Sportsbook Origination Model — V1

An independent NFL pregame pricing pipeline built for sports modeling research. It turns public
football data into a joint score distribution, then derives fair moneylines, point spreads,
and totals. Sportsbook odds never enter the prediction model.

**V1 is complete as a local research and portfolio project.** The latest verified implementation
passes **148 tests**, including regression, integration, dashboard, and offline demo tests, plus
lint, formatting, and type checks. The project does not place bets or claim a profitable edge.

## Results at a glance

The original retrospective 2025 holdout contains **272 regular-season games**. The Ridge model
improved on a simple league scoring baseline:

| Metric | B0 league baseline | M1 Ridge | M1 − B0, with 95% bootstrap interval |
|---|---:|---:|---:|
| Team-game score MAE, points | 7.88 | 7.44 | −0.44 [−0.67, −0.22] |
| Margin MAE, points | 11.06 | 10.15 | −0.92 [−1.37, −0.47] |
| Total MAE, points | 10.97 | 10.62 | −0.36 [−0.70, +0.02] |
| Three-way log loss | 0.730 | 0.671 | −0.059 [−0.086, −0.032] |
| Margin CRPS | 7.99 | 7.29 | −0.70 [−1.00, −0.39] |
| 80% margin interval coverage | 77.6% | 80.1% | — |

Lower error, log loss, and CRPS are better; interval coverage should be near its nominal level.
Uncertainty is estimated using paired season-week block resampling. Total MAE improves in point
estimate, but its interval includes no improvement. These are comparisons with B0, not evidence
of outperforming sportsbook prices.

See the [full holdout report](reports/final/holdout/report.md) and
[model card](docs/model_card.md) for calibration, sample sizes, ablations, and limitations.
The original holdout predates the current code-digest enforcement and was run from a dirty
working tree. Its exact implementation cannot be fully reconstructed from the recorded commit.
Those artifacts are preserved as historical evidence; they are not presented as a newly verified
run of the corrected implementation. Details are in the [experiment protocol](docs/experiment_protocol.md).

## What the project demonstrates

- **Football data engineering:** normalized nflverse schedules and play-by-play, explicit play
  filters, franchise mappings, data-quality checks, and traceable source files.
- **Independent modeling:** recency- and opportunity-weighted team metrics, shrinkage toward
  league priors, a shared Ridge score regression, and an EPA-free ablation.
- **Chronological evaluation:** expanding annual fits, training-only preprocessing, and residual
  distributions estimated from earlier out-of-fold predictions.
- **Probability pricing:** a deterministic joint score distribution with coherent moneyline,
  spread, and total probabilities, including ties, pushes, and settlement rules.
- **Reproducible engineering:** versioned data contracts, checked model bundles, immutable raw
  snapshots, code/lockfile protocol digests, saved run manifests, and an offline test suite.
- **Analyst workflow:** a command-line pipeline and a Streamlit dashboard for slates, individual
  games, evaluation, and run provenance. The dashboard reads saved artifacts rather than
  retraining models on refresh.

```text
nflverse schedules + play-by-play
                |
       normalize and validate
                |
       features at a defined cutoff
                |
    baseline / shared Ridge score model
                |
       joint score distribution
                |
    fair moneylines, spreads, totals
                |
      evaluation + local dashboard
                |
    optional timestamped-odds comparison
```

The statistical core uses Python, pandas/NumPy, scikit-learn, and SciPy. Artifacts use Parquet
and JSON, with DuckDB views for inspection. Typer provides the CLI; Streamlit provides the UI.
Dependencies are pinned in `uv.lock`.

## V2 (prospective collection, market benchmarks, distribution challenger)

V2 adds immutable source receipts with as-of version selection, an optional Odds API adapter,
a frozen prospective epoch with a one-shot `prospective-tick`, an immutable forecast/decision
ledger, aligned market benchmarks, and a bounded four-parameter key-number challenger. See
`V2_SPEC.md`, `docs/v2_operations.md`, `docs/v2_model_card.md` and `V2_IMPLEMENTATION_RECAP.md`.

```bash
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run nfl-origination demo-v2 --config configs/v2_demo.yaml --offline
uv run nfl-origination research-v2 --config configs/v2_research.yaml --offline
uv run nfl-origination prospective-tick --config configs/v2_prospective.yaml --offline
```

## Quickstart: offline demo

Requirements: `uv` and Python 3.12, on macOS or Linux. Run commands from the repository root.
The initial dependency installation may require network access; the demo itself needs neither
network access nor credentials.

```bash
uv sync --frozen
uv run nfl-origination demo --config configs/demo.yaml --offline
uv run streamlit run src/nfl_origination/dashboard/app.py
```

The demo generates synthetic seasons, fits models, prices games, produces evaluation charts,
and exercises paper settlement with synthetic odds. Its artifacts are saved under
`artifacts/demo/` and `reports/demo/`. The dashboard discovers both demo and real research runs.
**Synthetic results demonstrate mechanics, not NFL forecasting performance or betting returns.**

For a review without installing anything, start with the
[model card](docs/model_card.md), [architecture](docs/architecture.md), and
[review packet](docs/review_packet.md).

## Work with real football data

The default research dataset covers 2010–2025. Seasons 2010–2011 provide feature warm-up;
model training starts in 2012; 2018–2023 are development seasons; 2024 is confirmation; 2025
is the original retrospective holdout. Only regular-season games are modeled.

```bash
uv run nfl-origination ingest --seasons 2010:2025
uv run nfl-origination validate-data --seasons 2010:2025
uv run nfl-origination build-features --config configs/v1.yaml
uv run nfl-origination backtest --config configs/development.yaml
uv run nfl-origination backtest --config configs/confirmation.yaml
uv run nfl-origination report --run latest-development --config configs/development.yaml
```

`ingest` downloads or refreshes source files. Subsequent research commands default to cached,
local data. Raw data, features, and model artifacts are gitignored; a fresh checkout must
build them. Selected historical reports are committed under `reports/final/`.

### Holdout preservation and separately labeled reproductions

The original frozen protocol cannot authorize reruns under the corrected implementation.
Preserve it and its existing results. If a reproduction is needed, freeze a new, separately
labeled protocol after fixing the configuration and code:

```bash
uv run nfl-origination freeze-protocol --config configs/holdout.yaml --set run.label=v1-reproduction
uv run nfl-origination backtest --config configs/holdout.yaml --set run.label=v1-reproduction
uv run nfl-origination report --run latest-holdout --config configs/holdout.yaml
```

These commands execute a new experiment; they are not part of the quickstart. Save its exact
run ID for later inspection. A reproduction on already-viewed 2025 outcomes is not a new,
untouched test set. New protocols verify research-code, dependency-lock, configuration, and
source-data digests before evaluation.

## Generate forecasts

Fit the separate 2026 model bundle, refresh current-season data, and price remaining games:

```bash
uv run nfl-origination fit --through-season 2025 --config configs/v1.yaml
uv run nfl-origination ingest --seasons 2026
uv run nfl-origination predict --season 2026
```

Add `--week 3`, for example, to restrict the slate. A live forecast uses generation time as its
information cutoff and excludes games that have already started. Its horizon is usually labeled
`custom_horizon`, distinct from the historical evaluation's kickoff-minus-24-hours horizon.

Two explicit research options are also available:

```bash
# Reconstruct an explicit past information time.
uv run nfl-origination predict --season 2026 --week 3 --as-of 2026-09-16T12:00:00Z

# Reconstruct the standard horizon only after all selected cutoffs have passed.
uv run nfl-origination predict --season 2026 --week 1 --reconstruct-standard-horizon
```

The dates and weeks illustrate the CLI; available games depend on the cached schedule and
execution date. Past-time reconstruction is not evidence that the saved model existed then.
Forecast comparisons reject decisions preceding the recorded model creation time.

## Data-availability safeguards

Two data modes distinguish reconstruction from observed availability:

- **`historical_reconstruction`**, the default, uses pinned historical files and only source
  games eligible by the prediction cutoff. The default eligibility lag is 48 hours after
  kickoff. This is a research convention, not proof of the original publication time.
- **`recorded_asof`** additionally requires observed source availability. Unknown observation
  times are ineligible. It currently uses the selected cache entry; historical selection among
  multiple cached versions remains deferred. The real historical results were not produced in
  this mode.

The latest safeguards cover league priors, rolling features, labels, rest metadata, and
residual inputs. Missing play timestamps remain unknown through team aggregation rather than
being replaced by another play's known timestamp. Feature caches include source hashes,
first-observed metadata, requested seasons, configuration, and feature/schema versions, and
are validated before reuse. Training and forecasting reject missing versions or invalid safety
flags. Saved model bundles must match the forecast's feature and availability contract.

## Optional market comparison

Import a canonical timestamped odds CSV, then compare it with saved predictions or run paper
settlement. Set `market.bookmaker` in a copy of `configs/v1.yaml` to the bookmaker identifier
in the imported file, and use that configuration consistently:

```bash
uv run nfl-origination import-odds --path data/user/odds.csv --config configs/market.yaml
uv run nfl-origination compare --run latest-forecast --config configs/market.yaml
uv run nfl-origination paper-backtest --run latest-holdout --config configs/market.yaml
```

`configs/market.yaml` is the user-created copy, not a bundled configuration. See the
[data dictionary](docs/data_dictionary.md) and [V1 specification](V1_SPEC.md) for the CSV
contract and quote eligibility rules. Distribution lookup matches both game and model.

No real timestamped odds were supplied for the published results. Missing eligible odds are
reported as unavailable; no real ROI or closing-line-value claim is made. This project
originates prices and supports comparison. Actual market making—liability management, bettor
profiling, flow-based line moves, and automated execution—is out of scope.

## Verification

Latest local verification: **148 tests passed**, including the offline demo and dashboard
checks; lint, formatting, and type checks passed. A read-only check of cached 2025 data
validated 272 completed games and 544 team-game rows. These checks do not rerun the historical
holdout or remove its provenance limitation.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/nfl_origination
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run pytest -q
```

The [GitHub Actions workflow](.github/workflows/ci.yml) is included. A passing hosted CI run has
not been independently verified in this project's recorded review evidence.

## Known limitations and next research questions

- The rounded-normal distribution smooths NFL key margins such as 3 and 7, allows implausible
  scores, and overpredicts ties. Holdout tie probability averaged about 2.8%, versus 0.37%
  observed. This matters for spread pushes and moneyline settlement.
- Historical source revisions, including upstream EPA models, prevent a fully certified
  point-in-time claim. Prospective snapshots and historical version selection remain future work.
- Injuries, starting-quarterback projections, depth charts, and weather are not modeled.
- Performance against B0 does not establish competitiveness with sportsbooks; real timestamped
  market evaluation remains necessary.
- V1 covers pregame regular-season game markets. It excludes playoffs, live betting, player
  props, parlays, production hosting, and real-money execution.

The next research priorities are prospective data/odds collection, market benchmarking, and a
better representation of key margins and ties. Keep V1 as the baseline and document a new
evaluation protocol before changing the model.

## Repository guide

| Location | Purpose |
|---|---|
| `src/nfl_origination/` | Data, features, models, pricing, evaluation, market tools, CLI, dashboard |
| `configs/` | Research, forecast, and synthetic-demo configurations |
| `tests/` | Unit, regression, integration, dashboard, and end-to-end tests |
| [Architecture](docs/architecture.md) | Component boundaries and statistical decisions |
| [Data sources](docs/data_sources.md) / [dictionary](docs/data_dictionary.md) | Inputs, definitions, availability, attribution |
| [Model card](docs/model_card.md) / [experiment protocol](docs/experiment_protocol.md) | Results, methodology, provenance, limitations |
| [Implementation status](docs/implementation_status.md) / [review packet](docs/review_packet.md) | Verification history and review evidence |
| `reports/final/` | Preserved research reports and illustrative forecast outputs |

## Attribution

Data comes from [nflverse](https://github.com/nflverse/nflverse-data), with schedules maintained
by Lee Sharpe and play-by-play from nflfastR. Review upstream dataset terms before redistributing
data. No sportsbook affiliation is implied.
