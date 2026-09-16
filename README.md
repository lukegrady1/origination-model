# NFL Sportsbook Origination Model (V1)

A local, reproducible research pipeline that originates independent NFL regular-season prices:
it estimates each game's joint score distribution from public play-by-play data and derives
fair moneylines, spreads and totals, then validates itself chronologically against a frozen
protocol. It is a portfolio research project, not a trading system.

## Origination, comparison, and market making are different jobs

- **Price origination** (this project): produce an independent probability distribution over
  outcomes and the fair lines implied by it, using only information available before a defined
  cutoff. The output is a number you could defend on its own.
- **Market comparison** (optional module here): line up those numbers against timestamped
  bookmaker quotes selected at the same decision time, and settle paper bets under a frozen
  policy. A gap between the two is a price-reference difference, not proof of mispricing.
- **Market making** (explicitly out of scope): managing liability, moving lines on flow,
  profiling bettors, and pricing to a book's risk. Nothing here does that.

## What it does

1. Downloads nflverse schedules and play-by-play (2010–2026) into an immutable, checksummed cache.
2. Builds as-of features at each game's own cutoff (kickoff − 24h): recency-weighted,
   opportunity-weighted, shrunk EPA/success/explosive rates, play volume, points, venue and rest.
3. Fits a league baseline (B0) and a shared Ridge score regression (M1) season by season on
   prior seasons only, with residual mean/covariance from earlier out-of-fold predictions.
4. Integrates a bivariate normal over half-point cells to get a deterministic joint PMF over
   integer scores; derives margin/total PMFs, fair prices and prediction intervals.
5. Evaluates with score/margin/total errors, 3-way and conditional log loss/Brier, CRPS,
   interval coverage, reliability, key-number diagnostics, an EPA-free ablation and paired
   season-week block-bootstrap intervals — then reports what it found, good or bad.
6. Optionally imports a timestamped odds CSV, compares at the cutoff, and settles paper bets.

## Measured results (real data, frozen 2025 holdout, 272 games)

| | B0 baseline | M1 Ridge | difference (95% bootstrap) |
|---|---|---|---|
| Team-game score MAE | 7.88 | 7.44 | −0.44 [−0.67, −0.22] |
| 3-way log loss | 0.730 | 0.671 | −0.059 [−0.086, −0.032] |
| Margin CRPS | 7.99 | 7.29 | −0.70 [−1.00, −0.39] |
| 80% margin interval coverage | 0.776 | 0.801 | |

Intervals exclude zero for score MAE, margin MAE, log loss, Brier and margin CRPS; total MAE
improves in point estimate only (interval includes zero). Full tables, slices, reliability and
key-number diagnostics: `reports/final/holdout/report.md` and `docs/model_card.md`. The rounded-normal score model under-weights 3- and 7-point margins
and over-predicts ties; no injuries, starters or weather are used; no ROI is claimed.

## Quickstart

```bash
uv sync --frozen
uv run nfl-origination demo --config configs/demo.yaml --offline     # synthetic, no network
uv run streamlit run src/nfl_origination/dashboard/app.py            # browse saved runs
```

Real research path (network needed once for `ingest`):

```bash
uv run nfl-origination ingest --seasons 2010:2025
uv run nfl-origination validate-data --seasons 2010:2025
uv run nfl-origination build-features --config configs/v1.yaml
uv run nfl-origination backtest --config configs/development.yaml
uv run nfl-origination backtest --config configs/confirmation.yaml
uv run nfl-origination freeze-protocol --config configs/holdout.yaml
uv run nfl-origination backtest --config configs/holdout.yaml
uv run nfl-origination report --run latest-holdout --config configs/holdout.yaml
uv run nfl-origination fit --through-season 2025 --config configs/v1.yaml
uv run nfl-origination ingest --seasons 2026
uv run nfl-origination predict --season 2026 --week 3                     # live: information time = now
uv run nfl-origination predict --season 2026 --week 3 --as-of 2026-09-16T12:00:00Z   # explicit past time
uv run nfl-origination predict --season 2026 --week 1 --reconstruct-standard-horizon # kickoff-24h, after the fact
uv run nfl-origination import-odds --path data/user/odds.csv        # optional
uv run nfl-origination compare --run latest-forecast                 # optional
uv run nfl-origination paper-backtest --run latest-holdout           # optional
```

Quality gates: `uv run ruff check .`, `uv run ruff format --check .`,
`uv run mypy src/nfl_origination`, `uv run pytest -q`.

## Repository map

`src/nfl_origination/` package (see `docs/architecture.md`), `configs/` YAML contracts,
`tests/` unit, integration and end-to-end suites with committed synthetic fixtures,
`docs/` architecture, data sources, data dictionary, model card, experiment protocol,
implementation status and review packet, `reports/final/` committed sanitized outputs.
Raw data, features and model artifacts are reproducible and gitignored.

## Attribution

Data from nflverse (https://github.com/nflverse/nflverse-data): schedules maintained by
Lee Sharpe, play-by-play from nflfastR. Review upstream terms before redistributing data.
