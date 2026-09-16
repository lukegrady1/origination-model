# NFL Sportsbook Origination Model — V1 Engineering Specification

Version: 1.0 • Planning date: September 16, 2026 • Status: implementation handoff

## 1. Purpose and operating agreement

Build a local, reproducible research application that independently estimates NFL regular-season game outcomes and derives fair moneylines, spreads, and totals. Demonstrate football feature engineering, chronological model validation, coherent probability pricing, and dependable software engineering to a sportsbook modeling or trading team.

This document is the planning authority for Claude Code. Implement the specified system; do not silently change its statistical design, add product scope, or claim profitable results. The planning assistant owns scope and methodological revisions; Claude Code owns implementation, tests, and evidence. Make routine implementation decisions autonomously. If a substantive ambiguity blocks work, document the question and continue independent milestones. Maintain `docs/implementation_status.md` with completed requirements, commands run, failures, and deviations for subsequent review.

The requested deliverable here is a specification, not an already implemented or validated model. All commands below are interfaces to implement, not claims that they currently exist.

### User stories

1. As a researcher, I can reproduce historical predictions without using the target game's results or future games.
2. As an analyst, I can inspect an upcoming slate's independent fair prices, underlying features, and data limitations.
3. As a reviewer, I can reproduce metrics and trace every prediction to data, code, configuration, and training cutoffs.
4. If timestamped odds are supplied, I can compare model probabilities with contemporaneous offers and run a paper-only settlement analysis.

### Definition of V1

Required: free-data ingestion, historical feature generation, baselines, one regularized score model, a joint score distribution, pricing, chronological evaluation, a CLI, a small local dashboard, offline demo fixtures, tests, and a documented results report.

Optional extension within V1: odds CSV import, timestamp-safe comparisons, and paper betting evaluation. Implement its import/pricing/settlement contracts and synthetic tests even if real odds are unavailable. A paid data subscription is not a completion prerequisite. Live provider API integration is deferred.

Success means technically correct, reproducible work with honest measured performance. Beating a bookmaker, positive ROI, or a minimum win rate is not an acceptance requirement.

## 2. Explicit non-goals

- Real-money wagers, automated execution, sportsbook account access, staking recommendations, or production trading.
- Live/in-play models, player props, parlays, futures, playoffs, or preseason pricing.
- A market-making engine: no liability management, bettor profiling, exposure-based line moves, or market microstructure simulation.
- Market odds as model inputs, target encodings, calibration inputs, or model-selection objectives.
- Injury scraping, projected starting-QB overrides, historical weather forecasts, depth charts, roster valuation, and personnel simulations.
- Neural networks, boosted-tree tournaments, Bayesian sampling, play-by-play game simulation, or exhaustive hyperparameter search.
- Distributed services, cloud deployment, accounts, authentication, queues, containers as a requirement, or a custom JavaScript frontend.
- Claims that a rounded continuous score distribution captures NFL key numbers or touchdown/field-goal mechanics accurately.

These are intentional limits. In particular, missing personnel and weather information will limit real pregame usefulness. Do not conceal those limitations behind elaborate UI or fabricated confidence labels.

## 3. Fixed design decisions

| Area | V1 decision |
|---|---|
| Runtime | Python 3.12; `uv` project and committed `uv.lock` |
| Data | nflverse public schedule and play-by-play releases; cached raw files |
| Storage | Parquet artifacts plus DuckDB analytical views; local filesystem |
| Transformation | pandas/NumPy, vectorized season-at-a-time processing |
| Validation | Pydantic for config/artifacts; explicit dataframe checks |
| Modeling | scikit-learn preprocessing and Ridge; SciPy joint-normal integration |
| CLI | Typer, entry point `nfl-origination` |
| UI | Streamlit reading saved artifacts; no training on page refresh |
| Charts | Plotly or matplotlib; use one consistently |
| Quality | pytest, Ruff, mypy on package code; GitHub Actions offline CI |
| Experiment tracking | JSON manifests and run directories; no MLflow dependency |

Resolve compatible dependency versions during setup and commit the lockfile. Do not invent a dependency version in documentation before testing it. Support macOS and Linux. Document tested hardware and measured runtimes instead of promising an unmeasured performance budget.

## 4. Architecture and boundaries

```text
nflverse assets --> immutable raw cache + source manifest
                         |
                         v
                  normalize + validate --> schedule/results tables
                         |
                         v
                 team-game aggregates
                         |
                         v
                as-of feature builder
                         |
                         v
              chronological fit / predict
                         |
                         v
          joint score distribution --> fair prices
                         |                  |
                         v                  v
                evaluation artifacts   slate artifacts
                         \                  /
                          local dashboard

odds CSV --> normalize/as-of select --> comparison + paper settlement
                                       ^
                                       |
                         saved predictions only
```

Enforce separation in code: `models/` and `features/` cannot import `market/`; training inputs use an explicit column allowlist. Store market fields separately from the feature dataset. The pricing module takes a distribution and a market definition, not raw football data. The dashboard displays artifacts, not ad hoc alternative calculations.

Each run has an immutable manifest: run ID, UTC creation time, Git commit and dirty flag, Python/dependency versions, seed, full resolved config/hash, source file hashes, schema version, feature version, model ID, training cutoff, forecast policy, evaluation mode, and output hashes. Re-running the same experiment must reproduce numerical results within stated floating-point tolerances. Operational timestamps may differ.

## 5. Data sources and source verification

### Required free sources

- [nflverse data repository and releases](https://github.com/nflverse/nflverse-data): discover and download the season PBP assets from documented releases. Use a small adapter around direct asset downloads; keep resolved URLs in manifests. Do not scrape rendered web tables.
- [nflreadr schedule reference](https://nflreadr.nflverse.com/reference/load_schedules.html): canonical game/schedule data maintained by Lee Sharpe. It contains both football and betting fields; normalize only approved pregame metadata into feature inputs.
- [Schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html) and [PBP dictionary](https://nflreadr.nflverse.com/articles/dictionary_pbp.html): confirm field definitions against downloaded schemas during implementation.

Download 2010–2025. Use 2010–2011 as feature warm-up, 2012–2017 as initial model history, 2018–2023 for chronological development, 2024 for final selection confirmation, and 2025 as the locked retrospective holdout. Only completed regular-season games with valid final scores are scored. Season means NFL season, not calendar year; January regular-season games retain the preceding year's season label. Forecast mode may ingest a later current-season schedule without changing these research splits.

Raw provider field names are adapter details. Required canonical PBP information: game/play IDs, offense/defense, quarter, play type, EPA, yards gained, dropback/rush indicators, kneel/spike indicators, and no-play status. Required schedule information: stable game ID, season/week/type, teams, kickoff date/time, neutral-site status, final scores/status where known. Inspect and test mappings before building features; fail if required semantic fields cannot be mapped reliably.

The [nflverse availability page](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html) reports that its injury source ended after 2024 and describes changes to depth-chart timestamps. Those feeds are not required for V1. This is a verified source limitation, not an invitation to fabricate replacements.

### Optional market source

[The Odds API historical documentation](https://the-odds-api.com/historical-odds-data/) describes paid historical snapshots, with featured-market history starting in June 2020 and finer snapshot spacing from September 2022. Exact bookmaker/event coverage still requires validation. Its [v4 API documentation](https://the-odds-api.com/liveapi/guides/v4/) defines timestamp semantics. V1 accepts exported canonical CSV; it does not purchase access or implement this API client.

Schedule betting lines may be imported into a separate retrospective benchmark table. Without reliable quote timestamps, bookmaker identities, and settlement rules, call them `untimed_reference`, not opening/closing prices or executable offers. Never use them for a point-in-time ROI claim.

### Source handling

Cache by dataset/season/content hash. Preserve download time, source URL, source version if available, checksum, row count, column schema, and license/attribution reference. Downloads have timeouts, bounded retries, and atomic writes. A failed refresh must not corrupt the last complete cache. Explicit `--offline` prohibits all network access. Raw source data and model binaries are gitignored; commit synthetic fixtures and manifests. Record upstream attribution and review individual dataset/provider redistribution terms before bundling actual source data.

## 6. Time semantics and leakage policy

All internal timestamps are timezone-aware UTC. If schedule times are supplied in Eastern time, use `America/New_York` with date-aware daylight-saving conversion. Never infer a fixed UTC offset. Unresolved kickoff time means the game cannot enter a timestamped backtest.

Default prediction cutoff: `kickoff_utc - 24 hours`, independently for each game. This is a defined research horizon, not a claim to reproduce the sportsbook's first opening line.

Historical free files generally do not prove what exact data revision existed at each old cutoff. Implement two explicit modes:

1. **`historical_reconstruction`**: use current pinned historical files, but only completed source games whose `kickoff_utc + 48 hours <= prediction_cutoff`. The 48-hour lag is a conservative eligibility convention, not an observed publication timestamp. Disclose possible data revisions and upstream EPA model look-ahead.
2. **`recorded_asof`**: additionally require the actual cached version's first observed time to be at or before cutoff. Store immutable snapshots prospectively. Data downloaded today must not masquerade as a historically observed snapshot.

Propagate mode into every report. Date-safe reconstruction is not a fully point-in-time-certified backtest. Past EPA values may be revised or computed with upstream models trained later; label this residual limitation explicitly. An EPA-free ablation is required to measure dependence on those fields, but does not establish full historical certification.

Historical metadata such as eventual kickoff rescheduling may also be revised. A reconstructed kickoff defines the research horizon; recorded-asof forecasts use the then-known schedule. A reschedule must produce a new versioned prediction, never mutate an old forecast.

Every feature row records cutoff, maximum source eligibility timestamp, maximum observed timestamp when applicable, source game IDs/hash, and fallback flags. Features must depend only on eligible preceding games. The target game's PBP, final scores, actual starters, realized weather, win-probability fields, closing lines, and postgame totals are forbidden inputs.

Labels are separated from features and loaded only in fitting/evaluation. A model's training labels and residual-distribution observations must also satisfy the availability policy at its fit time. All imputation/scaling fits occur inside the applicable training fold. No random game split, full-dataset normalization, or full-season feature aggregate joined back to earlier games.

## 7. Canonical data contracts

Use versioned schemas. Identifiers are strings, counts integers, probabilities float64, and timestamps UTC. Missing values stay null until an explicit, logged imputation step.

| Artifact | Key | Required content |
|---|---|---|
| `games.parquet` | game_id | season, week, REG type, home/away franchise IDs, kickoff, neutral flag, status |
| `results.parquet` | game_id | integer nonnegative home/away final scores, final status, eligibility time |
| `team_games.parquet` | game_id + team_id | opponent, offense/defense metric numerators and denominators, final points for/against, source provenance |
| `features.parquet` | game_id + cutoff + perspective | home/away perspective rows, approved features, history counts, missingness, provenance |
| `predictions.parquet` | run_id + game_id + cutoff | score means, distribution reference, moneyline probabilities, fair lines, warnings |
| `odds.parquet` | quote_id | event/game mapping, book, market, selection, line, decimal price, snapshot/update times, rules |
| `paper_bets.parquet` | run_id + quote_id | prediction ID, policy ID, win/loss/push probabilities, EV, stake, settlement, profit |

Normalize historical franchise aliases using an explicit tested mapping; preserve original codes. Do not merge distinct franchises because names resemble each other. Provider-event joins require teams plus compatible scheduled time, with a maintained crosswalk for reschedules. Ambiguous joins fail rather than fuzzy-match silently.

Prediction fields include `mu_home_score`, `mu_away_score`, `mean_margin`, `mean_total`, `p_home_win`, `p_tie`, `p_away_win`, `p_home_win_given_no_tie`, fair decimal/American moneylines under tie-void rules, fair home handicap, fair total, 50%/80%/95% margin/total intervals, tail/clipping diagnostics, and data-quality flags. Include model/schema/config/data hashes. Do not confuse prediction intervals for outcomes with confidence intervals for the estimated fair price.

Canceled/incomplete games have no training label. Ties are valid regular-season labels. Empty slates return a valid empty artifact and message. Missing expected source games produce an auditable exclusion table; never disappear through an inner join unnoticed.

## 8. Feature definitions

### Eligible plays and team-game aggregation

For efficiency features, include scrimmage plays with a valid offense/defense in quarters 1–4 and either dropback or rush indicator; exclude no-play penalties, kneels, spikes, kickoffs, punts, field goals, and conversion attempts. Use explicit adapter tests for scrambles and sacks. Include sacks/scrambles in dropbacks when the source identifies them as such. Designed rush means rush and not dropback. Do not double-count a scramble. Exclude overtime from efficiency features; final-score targets still include overtime.

EPA metrics require finite EPA; yardage metrics require finite yards. Store each metric's denominator separately. Success is `EPA > 0`; zero EPA is not success. Do not apply a win-probability garbage-time filter. This avoids an extra upstream modeled dependency.

| Metric for each team | Definition |
|---|---|
| Offensive EPA/play | Own EPA sum / own valid EPA-play count |
| Defensive EPA/play allowed | Opponent offensive EPA sum / opponent valid EPA-play count; lower is better |
| Offensive/allowed success rate | Success count / corresponding valid EPA-play count |
| Offensive/allowed dropback EPA | EPA sum / valid dropback EPA count |
| Offensive/allowed rush EPA | EPA sum / valid designed-rush EPA count |
| Offensive/allowed explosive rate | Count of dropbacks gaining ≥20 yards or designed rushes gaining ≥10 / valid eligible yardage-play count |
| Offensive plays/game | Eligible own play count per completed team game; a volume proxy, not seconds-per-play pace |
| Points for/against | Final game points; includes defense, special teams, and overtime |

For each metric, use the most recent 16 eligible team games, with game-recency weights `w_j = 2^(-j/8)` for `j=0` most recent. Halve weights of games outside the target season. Do not erase previous-year history at Week 1. Retain numerators/denominators so rate aggregation is weighted by opportunities, not an unweighted average of game rates.

Let `N = sum(w_j * numerator_j)` and `D = sum(w_j * denominator_j)`. Shrink to a prior league rate `r0`:

```text
shrunk_rate = (N + k * r0) / (D + k)
```

Use `k=200` eligible plays for overall EPA/success/explosive rates, `k=100` for dropback/rush EPA, and `k=4` games for play volume and points. These are fixed V1 design values, not tuned on holdout. Derive `r0` from eligible games in the preceding two completed seasons; for warm-up only, use earlier available games. If no prior exists, feature generation reports insufficient warm-up rather than inventing a league average. A new team with no history gets the prior plus a cold-start flag.

### Model row

There are two rows per game: team scoring against opponent. The shared score regression uses:

- Team offensive EPA/play, success, dropback EPA, rush EPA, explosive rate, plays/game, and points for.
- Opponent defensive equivalents for EPA/play, success, dropback EPA, rush EPA, explosive rate, and points allowed.
- Opponent offensive plays/game as a shared-volume predictor.
- `venue_advantage`: +1 for home, −1 for away, 0 for both at neutral venue.
- `rest_difference`: team's days since prior completed game minus opponent's, after individually capping rest to [3, 14]. Season opener/no preceding season game uses 7 days, with a missing-rest indicator for each side. Rest considers only games before cutoff.
- Team and opponent history-game counts capped at 16, and cold-start flags.

Do not add team IDs, season-end rankings, betting variables, projected-QB assumptions, or unapproved interaction features. Higher defensive EPA allowed means weaker defense; do not invert its sign silently. Neutral-site metadata unknown means fail that game's feature validation rather than assume a home advantage.

Keep a generated feature dictionary with exact column names, units, formulas, null policy, source fields, and availability rules. Fit a training-only median imputer plus missingness indicators for unexpected numerical nulls, followed by StandardScaler. All-null required columns fail training. Log null rates; >5% unexpected missing values in any required metric fails the run pending source investigation.

## 9. Models and joint score distribution

### Baseline B0: league score model

Predict pooled training mean team score plus an estimated home/away offset; neutral gets no offset. Estimate only from eligible training games. Attach a residual distribution using the same chronological procedure as the candidate, enabling proper probability comparisons.

### Candidate M1: shared Ridge score regression

One shared pipeline predicts final points for each perspective row. Home/away rows from a game always remain together in a fold. Fit Ridge with intercept and `alpha` in `{1, 10, 100, 1000}`. Each game contributes equal total weight; rows each receive 0.5. No target transformation and no separate independent win classifier. Use the predicted pair as the location of a joint distribution.

Select alpha using mean game-level score MAE across the 2018–2023 chronological development folds. Break ties within 0.01 points in favor of larger alpha. Freeze the choice before the 2024 confirmation and 2025 holdout. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) is the implementation reference; the particular architecture and tuning grid here are project decisions.

### Residual fitting without in-sample optimism

For every forecast season Y, construct expanding-season out-of-fold predictions for seasons 2016 through Y−1, each using eligible training seasons starting 2012 and ending before its prediction season. Use the already fixed alpha for M1. For development comparisons, repeat this procedure per candidate alpha without looking beyond the evaluated fold.

Pool residual pairs `(actual_home - predicted_home, actual_away - predicted_away)` from only the most recent five completed residual seasons. Require at least 500 residual games; otherwise report insufficient history. Estimate the two-dimensional residual mean and sample covariance. Add the residual mean to the predicted score pair, symmetrize covariance, then apply fixed shrinkage `Sigma = 0.9*S + 0.1*diag(S)` and a diagonal numerical floor of `1e-6`. Check positive definiteness. Baseline gets its own residual pool.

This is a homoscedastic bivariate-normal score approximation. The covariance captures empirical home/away score-error association; it is not a possession simulator. Keep all residual-fitting dates and game IDs in the model bundle. For season Y, fit the score regression on 2012…Y−1 and freeze model coefficients/residual parameters for that whole season. Features still update before each game's cutoff. Do not refit inside the season in V1.

### Discrete score support

Construct a deterministic joint PMF over integer scores `h,a >= 0` by integrating the bivariate normal over half-point cells. Score zero absorbs the latent interval `(-infinity, 0.5)`; other score k has interval `[k−0.5, k+0.5)`. This corresponds to rounding a continuous score and flooring at zero. Record the probability that at least one latent score is below −0.5 as `negative_latent_mass` and warn if above 1%.

Start with scores 0…100. Expand the upper boundary in increments of 25 until omitted upper-tail probability is below `1e-8`, with a hard limit of 250. Failure to meet the limit fails pricing. Clamp only numerical negative cell mass in [−1e-10,0]; larger negatives fail. Renormalize after confirming tail tolerance. Use a documented integration tolerance and repeatability test. Do not rely on Monte Carlo to produce final prices.

Derive actual score means, margin `M=H−A`, and total `T=H+A` from this PMF. Distribution-derived means may differ from raw regression locations; save both and name them unambiguously. Nonnegative integer scores guarantee coherent margin/total parity. This distribution permits implausible football scores and smooths key-number behavior; document that limitation prominently.

Emit equal-tail outcome intervals from marginal CDF quantiles (lower quantile uses first integer with CDF ≥ target). Never replace a broken PMF with a silent 50/50 forecast.

## 10. Pricing conventions and exact arithmetic

All calculations use decimal odds internally. Presentation may show American odds. Keep full precision in artifacts.

### Moneyline and ties

Compute `pH=P(M>0)`, `pD=P(M=0)`, and `pA=P(M<0)`. They sum to one. V1 moneyline pricing supports an explicitly configured two-way tie-void settlement rule:

```text
qH = pH / (1 - pD)
qA = pA / (1 - pD)
fair_decimal_home = 1 / qH
fair_decimal_away = 1 / qA
```

Do not assume every imported bookmaker uses this rule. Unknown or different moneyline settlement rules disable comparison and settlement. Three-way moneylines are out of scope. Probabilities exactly 0 or 1 produce documented infinite/degenerate fair-odds output rather than division errors; display bounds only as presentation labels.

### Spread

`home_handicap=s` is the number added to home score. Home −3.5 means `s=-3.5`. Home wins if `M+s>0`, pushes if zero, loses if negative. Away uses the complementary event and opposite handicap.

Output both `mean_home_handicap = -E[M]` and `fair_home_handicap`, because a mean is not automatically an even-probability line. Find the half-point-grid handicap minimizing `abs(P(win|not push)-0.5)` over −80…80; if needed expand to full PMF-supported margin range. Ties choose the line nearest `-E[M]`, then the lower numeric line. Export its actual win/loss/push probabilities and fair prices; it need not be exactly 50/50.

### Total

Over line L wins when `T>L`, pushes at equality, and loses below. Find the half-point-grid line minimizing the analogous conditional probability imbalance over 0…200, expanding to PMF support if necessary. Break ties by proximity to `E[T]`, then lower numeric line.

Use integer half-point units internally for line comparisons so floating-point equality cannot corrupt pushes. Quarter-point and split Asian markets are rejected.

### Odds, vig, and expected value

For American odds A: decimal = `1+A/100` when A>0, else `1+100/abs(A)` when A<0. Reject zero and invalid magnitudes below 100. Decimal must be finite and >1. Convert fair probability q to American: `-100*q/(1-q)` for q≥0.5, otherwise `100*(1-q)/q`; show 0.5 as +100.

For a matched two-sided quote, implied `r_i=1/d_i`; proportional no-vig estimate `q_i=r_i/sum(r)`. Pair only same book, snapshot, event, market, line, and settlement rule. For spread sides, lines must be opposites; totals require identical L. No-vig probabilities at integer lines are conditional on no push, not unconditional event probabilities.

Paper offer EV per unit stake is `p_win*(decimal_odds−1)−p_loss`; pushes/voids return stake and contribute zero. A 60% win probability at −110 with no push has EV `0.60*(100/110)−0.40 = 0.1454545…`. For a synthetic `(win=.50,push=.10,loss=.40)` distribution, fair decimal odds are 1.8 and EV at 1.8 is zero.

Optional illustration only: add a fixed 4% overround to the conditional two-way fair probabilities using `r_i=1.04*q_i`, price `1/r_i`. Reject if any r_i≥1. Label this an illustrative quote, not a risk-managed sportsbook price. Overround is not realized hold.

## 11. Chronological evaluation

### Frozen experiment protocol

1. Validate 2010–2025 source coverage and write exclusions before modeling.
2. Build all historical features with each game's own 24-hour cutoff and the stated availability convention.
3. Run expanding annual development folds: score seasons 2018, 2019, …, 2023; train each on 2012 through preceding season. Build residual distributions as specified above.
4. Select alpha using only development score MAE. Save the decision record.
5. Run 2024 confirmation and report all metrics. If a methodological change is made after seeing 2024, call it development data and document the revised protocol before opening 2025.
6. Freeze config, schema, and experiment plan; record a checksum. Run the 2025 holdout once for the final report. Bug corrections are allowed, but log what changed and label any rerun. Do not repeatedly tune against 2025 and continue calling it untouched.
7. Fit a separate 2026 forecasting bundle on eligible completed seasons through 2025. Do not mix its retrospective predictions into holdout metrics.

The 2025 season is a retrospective project holdout, not proof the researcher had no prior knowledge of its outcomes. Put that caveat in the model card. Current incomplete-season games are excluded from the initial final report.

### Required measures

- Score MAE/RMSE per team-game, plus margin and total MAE/RMSE per game.
- Three-category home/tie/away log loss and multiclass Brier score; document Brier as the sum across categories, averaged over games.
- Binary conditional moneyline Brier/log loss on non-tied games only; report ties separately.
- Home-win and non-tie conditional reliability plots, bin counts, predicted tie rate versus observed tie rate. Use fixed 0.1 probability bins and label bins with <30 observations as sparse.
- Margin/total CRPS from integer marginal PMFs; 50%/80%/95% interval empirical coverage and mean widths.
- Baseline comparisons on exactly matched games; include paired differences.
- Season-level, early-season weeks 1–4, neutral-site, and cold-start slices; show sample sizes and avoid interpreting tiny groups.
- Key-number diagnostics at margins 0, ±3, ±7 and total parity; show predicted versus observed frequencies. These expose distribution shortcomings rather than promise to solve them.
- EPA-free ablation: remove all EPA and success features, reselect alpha on development only, evaluate through the same frozen protocol. All remaining features stay identical.

Use 1,000 paired bootstrap resamples of season-week blocks for 95% percentile uncertainty intervals on aggregate metric differences. Sample blocks with replacement, include all games in each selected block, and recompute metrics weighted by included games. This is a descriptive uncertainty approximation; cross-week team dependence remains. Seed is fixed. No p-value fishing or many-model leaderboard.

Market comparison, when available, reports paired samples and market availability counts. Compare margin expectation with a normalized market home handicap via `market_expected_margin = -home_handicap`; compare total expectation with market total. Label these as price-reference errors, since bookmaker lines are not necessarily conditional means. Market probability scoring requires complete matched odds and correct settlement semantics.

Report poor results. Do not add a success gate requiring M1 to beat B0. The final recommendation may be “baseline preferred; further modeling needed.”

## 12. Optional odds import and paper backtest

CSV schema: `quote_id, provider_event_id, game_id, bookmaker, market, selection, line, decimal_odds, snapshot_at_utc, bookmaker_updated_at_utc, kickoff_at_snapshot_utc, settlement_rule`. Moneyline line is null; selections are home/away or over/under. Book/source identifiers are explicit. `snapshot_at` must represent source observation time, not export time. Keep separate `ingested_at`. Synthetic CSV is labeled synthetic throughout UI and reports.

At forecast cutoff choose the newest complete eligible snapshot from the configured book with `snapshot_at <= cutoff`, `bookmaker_updated_at <= snapshot_at`, and both timestamps no more than 60 minutes old relative to cutoff. Missing update timestamps disqualify a quote from the paper backtest. Require forecast generation/model availability no later than decision time in recorded-asof mode. Historical reconstruction remains explicitly hypothetical.

Use one configured bookmaker, no hindsight best-price selection. Default optional policy: EV≥0.03, one unit flat stake, at most one selection per game across supported markets. If several qualify, choose highest EV, then market order moneyline/spread/total, then selection lexical order. Freeze thresholds before holdout. No Kelly sizing or bankroll compounding. Include losing signals and every exclusion reason.

Settle from final scores under the recorded rule; canceled games void. Show wins/losses/pushes/voids, stake, net profit, ROI on non-void staked units including push stakes, max drawdown of cumulative unit profit, and game counts. Order same-time settlements by game ID for deterministic plots. Provide season-week block bootstrap ROI intervals, resampling bets and recomputing profit/stake. Do not present synthetic returns as evidence.

Optional closing proxy: latest complete pre-kickoff quote from the same book in the final 30 minutes, strictly before kickoff. Call it an observed closing proxy, not guaranteed final close. For spreads, selected-side line improvement is `bet_handicap - close_handicap`; for over, `close_total - bet_total`; for under, `bet_total - close_total`. Positive is favorable. Price CLV uses `bet_decimal * closing_no_vig_selected_probability - 1` only for the same market/line/rules; otherwise price CLV is null. Report line and price changes separately, without pretending they are interchangeable.

If usable market data is absent, comparison/ROI/CLV panels display “unavailable: no eligible timestamped odds.” Core model runs still succeed.

## 13. Repository structure

```text
nfl-origination/
  README.md
  CLAUDE.md
  pyproject.toml
  uv.lock
  .python-version
  .gitignore
  .github/workflows/ci.yml
  configs/{v1,development,holdout,demo}.yaml
  src/nfl_origination/
    __init__.py
    cli.py
    config.py
    schemas.py
    provenance.py
    data/{download,normalize,validate,storage}.py
    features/{aggregate,rolling,asof,builder}.py
    models/{baseline,ridge,distribution,bundle}.py
    pricing/{odds,markets,fair_lines}.py
    evaluation/{splits,metrics,bootstrap,report}.py
    market/{import_csv,asof,compare,settle}.py
    dashboard/app.py
  tests/
    unit/
    integration/
    fixtures/synthetic/
    test_end_to_end_demo.py
  docs/
    architecture.md
    data_dictionary.md
    data_sources.md
    model_card.md
    experiment_protocol.md
    implementation_status.md
    review_packet.md
  data/{raw,normalized,features}/       # ignored, reproducible
  artifacts/{models,runs}/             # ignored
  reports/                            # selected sanitized outputs may be committed
```

`CLAUDE.md` summarizes invariants and links this specification, without conflicting copies of the statistical formulas. Notebooks are optional exploration only; the authoritative pipeline lives in package code. DuckDB exposes read-only views over normalized Parquet and saved artifacts, not a second competing source of truth.

## 14. Module interfaces

Implement typed public interfaces with equivalent behavior; small signature changes are permitted if documented:

```python
ingest(seasons: list[int], cache_dir: Path, offline: bool) -> SourceManifest
validate_sources(manifest: SourceManifest) -> ValidationReport
aggregate_team_games(games, plays, results) -> TeamGameTable
build_features(games, team_games, policy: AsOfPolicy) -> FeatureTable
fit_bundle(features, labels, fit_spec: FitSpec) -> ModelBundle
predict_game(bundle: ModelBundle, game_features) -> ScoreDistribution
price_market(distribution: ScoreDistribution, market: MarketSpec) -> FairPrice
run_backtest(config: ExperimentConfig) -> RunManifest
import_odds(path: Path) -> OddsImportReport
select_quotes(odds, decision_time, policy: QuotePolicy) -> EligibleQuotes
settle(selection: Selection, result: GameResult) -> Settlement
build_report(run: RunManifest) -> Path
```

`ModelBundle` includes preprocessing, fitted coefficients, ordered features, residual parameters, training IDs/cutoff, dependency versions, and provenance. Predictions reject incompatible feature/schema versions. Never load model files supplied by strangers; serialized model loading is for locally generated trusted artifacts.

## 15. CLI contract and reproducible commands

Implement inclusive season ranges (`2010:2025`). Commands emit concise summaries plus machine-readable manifests. Config is the source of defaults; flags override it and resolved config is saved. Unknown config keys fail. Exit codes: 0 success, 2 invalid input/config, 3 missing/ineligible data, 4 failed numerical/model validation. Optional missing market data is not an error for core commands.

```bash
# Install from committed lock after initial project setup.
uv sync --frozen

# Offline review path: synthetic data, small fits, pricing and report.
uv run nfl-origination demo --config configs/demo.yaml --offline

# Real historical research path.
uv run nfl-origination ingest --seasons 2010:2025
uv run nfl-origination validate-data --seasons 2010:2025
uv run nfl-origination build-features --config configs/v1.yaml
uv run nfl-origination backtest --config configs/development.yaml
uv run nfl-origination freeze-protocol --config configs/holdout.yaml
uv run nfl-origination backtest --config configs/holdout.yaml
uv run nfl-origination report --run latest-holdout

# Fit a separate next-season bundle and produce forecasts.
uv run nfl-origination fit --through-season 2025 --config configs/v1.yaml
uv run nfl-origination ingest --seasons 2026
uv run nfl-origination predict --season 2026 --week 3 --as-of 2026-09-16T12:00:00Z

# Optional user-supplied odds; never requires a provider subscription.
uv run nfl-origination import-odds --path data/user/odds.csv
uv run nfl-origination compare --run latest-forecast
uv run nfl-origination paper-backtest --run latest-holdout

# Local artifact browser and quality gates.
uv run streamlit run src/nfl_origination/dashboard/app.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src/nfl_origination
uv run pytest -q
```

`predict --as-of` is an explicit live/snapshot research time. Only future games at that timestamp qualify; use no data after it. If it differs from kickoff−24h, label the forecast `custom_horizon` and do not mix it into the standard-horizon backtest. The date above illustrates syntax, not a promise that all current data is available then. Forecasts need the current season's preceding completed games, so ingest refreshes both schedule and PBP where available. Recorded-asof mode fails rather than backdate newly downloaded data.

`freeze-protocol` writes a checksummed manifest locking alpha, data snapshot hashes, exclusions, code/config, policy, and intended holdout. `latest-*` resolves to a recorded run ID and prints it. `--run <id>` always supports exact reproduction. A holdout command refuses an unfrozen/mismatched protocol; this is an automated integrity gate, not an extra user-approval workflow.

## 16. Dashboard and reporting requirements

Keep the dashboard small and artifact-driven:

1. **Slate:** teams, kickoff/cutoff, fair home handicap/total/moneylines, tie probability, model/version, and missing-history flags. Show unavailable values explicitly.
2. **Game detail:** recent team metrics and source dates, baseline versus candidate, score/margin/total distributions, outcome intervals, and optional matched market quotes with timestamps.
3. **Evaluation:** season table, baseline differences, reliability, interval coverage, exclusions, and key-number diagnostics.
4. **Run audit:** data mode, hashes, fit window, configuration, source links, limitations, and download links to JSON/CSV artifacts.

No “lock,” “guaranteed winner,” or confidence badge derived solely from disagreement with a book. Linear standardized coefficients may be shown as model diagnostics, not causal explanations. A model-market gap is not proof of mispricing. Synthetic/demo mode gets a persistent visible label. Empty/missing optional panels must not crash the app.

Required report: Markdown narrative plus portable PNG charts, metrics JSON, and prediction CSV/Parquet. Include actual results, methodology, samples/exclusions, uncertainty, limitations, and a candid next-step recommendation. An HTML report is optional.

## 17. Tests that protect real failure modes

### Unit tests

- Play filtering: penalty/no-play, kneel, spike, sack, scramble, designed run, missing EPA, and overtime; denominator checks.
- Rolling/shrinkage: hand-calculated weighted rates; correct offseason discount; Week 1 cold-start; no target game inclusion.
- Temporal joins: exact cutoff boundary, late publication/observation, future source game, same-week earlier game, reschedule, and daylight-saving conversion.
- Feature isolation: mutating market columns changes no features/predictions; injecting an unapproved column fails schema validation.
- Leakage sentinel: modifying target/future results or PBP cannot alter an earlier feature row or forecast; modifying legitimately eligible history can.
- Fold grouping: two team rows never split across training/evaluation; future residuals and scaler statistics never enter a prior fold.
- Neutral venue and swapped-team perspectives transform venue/rest features correctly; do not require identical home/away score distributions after swapping because residual means/covariance can be asymmetric.
- PMF: finite, nonnegative, sums to one within 1e-8, stable integration, adequate support, valid covariance, coherent parity and intervals.
- Deterministic 24–21 distribution: home −3 pushes; home −3.5 loses; home −2.5 wins; total 45 pushes; over 44.5 wins; moneyline home wins.
- Deterministic 20–20 distribution: tie probability one; moneyline tie-void settlement voids and conditional odds are undefined with a clear flag.
- American/decimal conversions round-trip within tolerance; +150→2.5, −200→1.5; malformed odds rejected.
- Win/push/loss sum to one; zero-EV fair odds; both sides settle consistently; vig/de-vig with matched pairs only.
- Odds import: stale, future, incomplete pair, mismatched lines/books/rules, duplicate conflict, and ambiguous event join.
- Metrics: known tiny fixtures for log loss, Brier, CRPS, ROI, bootstrap sampling, and paired baselines.

### Integration and acceptance fixtures

Build a deterministic synthetic multi-season fixture generator that supports at least 500 prior residual games. Use small already-aggregated fixtures for end-to-end modeling speed and a separate miniature raw-PBP fixture for ingestion-to-aggregation integration. Do not pretend synthetic NFL-like outcomes are real observed games.

The offline demo must exercise normalization, feature building, fit, distribution, pricing, report, and optional odds settlement. Test raw-download parsing via mocked responses. CI blocks network. Real-source smoke validation is an explicit local command, not an unreliable CI dependency. Rerun identical fixed inputs and compare numerical artifacts (allow 1e-8 probability tolerance and 1e-6 metric tolerance; exclude runtime timestamps).

Tests must verify behavior and mathematical invariants, not only mock calls or reproduce implementation formulas line for line. All core temporal, pricing, and settlement failure paths must have tests; no arbitrary repository-wide coverage percentage replaces these requirements.

## 18. Implementation milestones

| Step | Work and concrete output | Completion evidence |
|---|---|---|
| 1. Scaffold | Package, configs, lockfile, CLI skeleton, CLAUDE.md, synthetic fixtures | Installation and offline CI gates pass |
| 2. Data contracts | Download/cache, mappings, schemas, season audit, provenance | Real 2010–2025 coverage/exclusion report; mocked corruption/retry tests |
| 3. Features | Aggregates, historical/as-of policies, shrinkage, dictionary | Hand-worked fixture and leakage sentinel tests pass |
| 4. Baseline | B0, folds, residual estimation, bundle serialization | Chronological baseline report and bundle round-trip |
| 5. Candidate | Ridge pipeline, restricted alpha search, OOF residual pool | Development selection artifact; no holdout reads |
| 6. Distribution/pricing | Joint PMF, probabilities, fair lines, vig illustration | All pricing, push/tie, integration-tolerance tests pass |
| 7. Evaluation | Metrics, ablation, bootstrap, 2024 confirmation | Paired tables, plots, protocol ready to freeze |
| 8. Optional-market contracts | CSV/as-of selection, comparisons, settlement | Synthetic odds tests; real coverage or explicit unavailable status |
| 9. Dashboard/report | Saved-artifact views, exports, model card | Demo walkthrough and empty-state smoke check |
| 10. Final research run | Freeze protocol, run 2025 once, fit separate forecast bundle | Final review packet, actual results, all acceptance checks |

Dependencies: 1→2→3→4→5; pricing can be developed against fixtures after 1, but integration waits for 4/5. Final evaluation waits for pricing. Dashboard waits for artifact schemas. Complete the baseline before expanding candidate complexity. Milestone completion does not require asking for permission to continue already authorized implementation.

At each milestone record files changed, exact verification commands and outcomes, data/model assumptions, and unresolved blockers. If real data access fails, continue fixture-backed engineering and mark real-data acceptance incomplete; never claim the whole project is validated from synthetic tests alone.

## 19. Acceptance criteria and deliverables

| ID | Acceptance criterion |
|---|---|
| A01 | A fresh checkout installs from lockfile and runs the offline demo with no credentials/network. |
| A02 | Real source coverage for every requested season is audited; excluded/missing games and reasons are exported. |
| A03 | Features and labels are separated; as-of and future-mutation tests pass; data mode appears on every report. |
| A04 | B0, M1, and the EPA-free ablation run under identical chronological splits with grouped game rows. |
| A05 | Residual distribution uses only prior chronological predictions and passes PMF/support/covariance checks. |
| A06 | Moneyline, spreads, totals, ties, pushes, neutral sites, signs, and odds conversions pass exact fixtures. |
| A07 | Frozen 2025 report includes all required metrics, baselines, uncertainty intervals, sample counts, and limitations. |
| A08 | Repeated fixed-input runs reproduce predictions/metrics within documented tolerances. |
| A09 | CLI forecast exports traceable game prices; dashboard displays saved artifacts and handles unavailable odds. |
| A10 | Optional market module rejects ineligible odds and settles synthetic fixtures correctly; absent paid data is disclosed. |
| A11 | Ruff, mypy, pytest, offline CI, and a real-source smoke run pass or their specific blockers are reported. |
| A12 | README, model card, data dictionary, source attribution, experiment protocol, and review packet are complete. |

Deliver:

- Working repository with `uv.lock`, tested CLI and dashboard, configs, and synthetic fixtures.
- Reproducible baseline/candidate/ablation artifacts and frozen experiment manifests.
- Final 2025 evaluation report, charts, predictions, and explicit recommended model (including baseline if appropriate).
- Current-season forecast example when source availability allows; otherwise a clearly labeled historical/demo slate.
- `docs/review_packet.md` linking architecture decisions, test results, real-data exclusions, methodology deviations, and known limitations.
- A concise portfolio README explaining the original-number approach and the difference between price origination, market comparison, and actual market making.

Resume claims must use measured counts and results from the final run. Do not invent performance, trading experience, live deployment, profit, or market-making capability. A suitable eventual claim is “Built a reproducible NFL pregame pricing pipeline with chronological validation, joint outcome probabilities, and tested spread/total settlement,” supplemented only by verified metrics.

## 20. Initial configuration contract

```yaml
schema_version: 1
seed: 42
data:
  seasons: [2010, 2025]  # inclusive endpoints; schema must define this explicitly
  game_type: REG
  mode: historical_reconstruction
  completed_game_lag_hours: 48
forecast:
  cutoff_hours_before_kickoff: 24
features:
  history_games: 16
  half_life_games: 8
  offseason_weight: 0.5
  prior_seasons: 2
  shrinkage_plays: 200
  shrinkage_split_plays: 100
  shrinkage_games: 4
model:
  family: ridge_score
  alpha_candidates: [1, 10, 100, 1000]
  selected_alpha: null  # filled from development decision, never guessed
  train_start_season: 2012
  residual_first_season: 2016
  residual_window_seasons: 5
  min_residual_games: 500
  covariance_diagonal_shrinkage: 0.1
distribution:
  initial_max_score: 100
  score_step: 25
  hard_max_score: 250
  max_omitted_mass: 1.0e-8
evaluation:
  development_seasons: [2018, 2019, 2020, 2021, 2022, 2023]
  confirmation_season: 2024
  holdout_season: 2025
  bootstrap_replicates: 1000
market:
  enabled: false
  bookmaker: null
  max_quote_age_minutes: 60
  min_ev: 0.03
  stake_units: 1
  max_selections_per_game: 1
  moneyline_settlement_rule: two_way_tie_void
```

Development configs may enumerate candidates; fit/forecast configs must resolve one saved selected alpha. Demo config may reduce calendar years and use synthetic years/fixtures but must preserve enough residual games and mathematical tests. No production evaluation thresholds may be weakened to accommodate a broken demo.

## 21. First instruction to Claude Code

Read this specification and inspect the repository before editing. Write a concise implementation checklist mapped to A01–A12, then implement the milestones in order. Start with a functioning offline fixture path and leakage-safe data contracts. Keep the original-number model independent from all market odds. Make routine engineering choices, test them, and record evidence. Do not add deferred features or replace the specified statistical protocol without documenting the needed planning decision. At completion, deliver the repository, real evaluation results where available, and the review packet for the planning assistant to inspect.
