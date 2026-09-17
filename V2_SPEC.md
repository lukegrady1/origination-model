# NFL Origination Model V2 — engineering handoff

Status: implementation specification, not an implementation or performance claim.
Prepared: 2026-09-16. Repository: `/Users/luke/origination-model`.
Audience: Claude Code implementing V2; Luke and the planning assistant reviewing the result.

## 1. Objective and definition of success

Build on V1 to answer: **How well do independently generated NFL prices perform on genuinely
prospective games, against both V1 and contemporaneous sportsbook prices?**

V2 has three required workstreams, in this order:

1. Preserve timestamped source versions and record forecasts before games happen.
2. Collect/import real market quotes and evaluate comparable model/market probabilities.
3. Implement one small, coherent distribution challenger addressing ties and key margins.

Do not promise profitable betting, or make model improvement a condition for completing the
software. A well-tested challenger that fails its promotion criteria is a valid research result.
V1 remains the champion unless evidence supports changing it.

Separate these completion states in every recap:

- **Engineering complete:** contracts, commands, tests, demo, and documentation implemented.
- **Operationally configured:** real source collection, explicit bookmaker/rules configuration,
  and forecast recording have been exercised locally.
- **Prospective evidence pending/available:** actual future games have settled, with coverage
  and sample sizes reported. This cannot be manufactured during an implementation session.
- **Model decision:** retain V1 / challenger ready for shadow collection / evidence insufficient /
  challenger recommended for review. No automatic real-money or model promotion.

## 2. Instructions to the implementing agent

Read `README.md`, `V1_SPEC.md`, `docs/architecture.md`, `docs/experiment_protocol.md`,
`docs/model_card.md`, `docs/implementation_status.md`, and applicable repository instructions.
Then inspect the code; recaps and prior documents describe context, not verified current behavior.

Implement this specification in bounded phases below. Make routine implementation decisions
autonomously and record them in `docs/v2_decisions.md`. Do not stop after scaffolding or a plan.
Do not broaden the model search or weaken time-availability checks to obtain nicer results.

Before changing code:

- Record git status and the baseline check results. Preserve unrelated user changes.
- Inventory existing manifests, model bundles, protocols, and `reports/final/` checksums.
- Record actual schema/feature versions from code; config schema and feature versions are
  separate concepts. Do not assume the `schema_version: 1` YAML means feature version 1.
- Check cached-data availability and current commands. Use existing cached files for research;
  report missing datasets without manufacturing results.

Continue all offline implementation if credentials, real odds, or historical data are absent.
Report the exact missing dependency separately. Do not buy data, create accounts, install a
background scheduler, publish results, or place bets as part of this handoff.

## 3. Preserve the existing system

V1 already supplies normalized nflverse data, availability rules, rolling/shrunk features,
Ridge score models, a rounded bivariate-normal joint score PMF, pricing, chronological evaluation,
CSV odds import, paper settlement, run manifests, and a read-only Streamlit dashboard.

Retain Python 3.12, uv, pandas/NumPy/SciPy/scikit-learn, Parquet/JSON, Typer and Streamlit.
Use the current modules and dependencies. No new database server, frontend framework, task
queue, orchestration service, or model registry service is needed.

### Invariants

- Never overwrite original V1 reports, predictions, protocols, decisions, or fitted bundles.
- Newly reconstructed V1 results must get new run IDs and explicit reproduction labels.
- The original 2025 results came from an imperfectly reconstructible dirty working tree.
  Preserve that caveat. Newly passing tests do not retroactively validate those artifacts.
- 2024 and 2025 outcomes have already been inspected; neither is an untouched V2 test set.
- No market columns enter football features, preprocessing, score fitting, or distribution fitting.
- Keep V1 feature definitions and Ridge alpha 100 fixed for this distribution experiment.
- Unknown availability means ineligible, not available by default.
- Keep all observation-propagation, prior-availability, source-hash/cache, model-bundle contract,
  distribution/model lookup, CLV-sign, and frozen-protocol regression tests.
- Model coefficients, residual parameters, and distribution adjustments are fit using prior
  games only. Predictions and outcomes remain separate artifacts.
- Dashboard refreshes only read artifacts. No implicit ingestion, fitting, or collection.
- All production timestamps are timezone-aware UTC. User-facing displays can also show local time.

Version new schemas explicitly. Support reading legacy V1 runs in the dashboard. Reject old
bundles where a new execution contract is required; do not silently reinterpret or rewrite them.

## 4. Architecture and repository changes

```text
nflverse source fetches                optional odds API / canonical CSV
          |                                       |
immutable blobs + observation receipts + append-only event records
          |                                       |
as-of source-version selection          normalized paired market snapshots
          |                                       |
V1 features -> frozen score bundle -> V1 PMF / V2 adjusted PMF
          |                                       |
          +---------- prospective forecast commit-+
                           |
             later final results + closing proxy
                           |
             versioned evaluation / paper ledger
                           |
                report + read-only dashboard
```

Suggested additions (equivalent small modules are acceptable; document deviations):

```text
V2_SPEC.md
configs/v2_research.yaml
configs/v2_prospective.yaml
configs/v2_demo.yaml
src/nfl_origination/
  data/snapshots.py             # immutable receipts, version lookup, legacy inventory
  market/providers.py          # small provider interface + The Odds API adapter
  market/benchmark.py           # aligned market scoring, coverage, closing proxy
  models/key_number.py          # four-parameter joint-PMF challenger
  prospective/protocol.py       # freeze, epoch contracts, verification
  prospective/ledger.py         # immutable forecast/outcome/event records
  prospective/runner.py         # one-shot due-game collection/forecast cycle
  evaluation/v2.py              # challenger selection and prospective reports
tests/unit/test_snapshots.py
tests/unit/test_key_number.py
tests/unit/test_market_benchmark.py
tests/unit/test_prospective_ledger.py
tests/integration/test_v2_pipeline.py
tests/integration/test_v2_dashboard.py
tests/fixtures/v2/              # explicitly synthetic provider/time/result examples
docs/v2_decisions.md
docs/v2_data_contracts.md
docs/v2_operations.md
docs/v2_model_card.md
docs/dashboard_guide.md
V2_IMPLEMENTATION_RECAP.md
```

Use `artifacts/v2/`, `reports/v2/`, and versioned data locations for new work. Add discovery of
the V2 artifact root to the dashboard without removing legacy/demo roots. Derived tables may
be rebuilt; immutable raw receipts and committed forecast payloads are the source of truth.

## 5. R1 — immutable source history and correct as-of selection

### 5.1 Source records

Current content-addressed raw blobs are useful, but a mutable `entry.json` pointing to the
latest version is insufficient for historical selection. Preserve it for compatibility while
adding immutable per-observation receipts.

Each receipt must include:

| Field | Type / rule |
|---|---|
| `receipt_id` | Unique immutable string |
| `dataset`, `season` | Source identity; nullable season only for all-season schedules |
| `content_sha256`, `blob_path`, `bytes` | Verified payload identity |
| `request_started_at_utc` | Collector clock at request start |
| `observed_at_utc` | Collector clock after the complete response is received |
| `persisted_at_utc` | Successful durable receipt creation time |
| `provider_timestamp_utc` | Optional provider metadata, never a replacement for observation time |
| `source_url_redacted`, `adapter_version` | Provenance without credentials |
| `synthetic`, `provenance_quality` | Explicit booleans/classification |

Store a blob once per hash, but retain each successful observation receipt. A failed/partial
download does not create a usable receipt or alter the last successful state. Use unique temp
paths, atomic publication, and a local writer lock to avoid duplicate/concurrent corruption.

The as-of selector chooses the most recently **observed** valid receipt at or before cutoff,
per dataset/season, with a deterministic receipt-ID tiebreaker. Do not choose a blob simply
because its first-ever observation is old: in A -> B -> A history, the receipt sequence matters.
Later normalization time does not invent earlier source availability.

Use the selected source versions consistently for schedules, PBP, prior rates, rest, training
labels, residual inputs, and feature calculations. Cache keys must include the selected version
identities, effective observation metadata, requested seasons, and existing policy/version fields.

For `recorded_asof`, retain the V1 48-hour completed-game eligibility convention in addition to
observation checks. Keep it explicitly described as a conservative convention, not an assertion
about actual publication time. Never mix a current schedule row into an older as-of feature row.

### 5.2 Legacy cache migration

Provide a read-only inventory plus an explicit, additive migration command.

- Preserve original files and metadata byte-for-byte.
- Import supported existing timestamps with `provenance_quality=legacy_metadata`.
- Old orphaned blobs without timing evidence remain `availability_unknown`; file modification
  time, game date, and discovery time cannot establish earlier availability.
- Unknown historical blobs may be used for explicitly labeled retrospective reconstruction,
  never for strict as-of claims.
- Record a baseline inventory at V2 activation. Files read and hashed now can establish availability
  now for future forecasts, without making any earlier-time claim.
- Legacy migration is idempotent; it must not create new earlier availability on repeated runs.

### 5.3 Practical startup constraint

A model built today can train on old seasons observed today and forecast future games. Do not
require a decade of prospective archives just to begin recording forecasts. Record the exact
training-source manifest and training mode in the bundle. Historical OOF calibration may remain
retrospective; label this separately from the prospective availability of the deployed bundle.

Introduce an explicit compatibility contract for this case rather than disabling V1 bundle
validation: all training inputs and fitted parameters must exist before the forecast cutoff;
live feature inputs use strict observed availability. Record both `training_evidence_mode` and
`forecast_evidence_mode`. No legacy bundle silently acquires this contract.

## 6. R2 — odds collection, identity, and source contracts

### 6.1 Required paths

Keep canonical CSV import. Add one optional live adapter for The Odds API, using the existing
HTTP dependency. No second provider, scraping, or historical data purchase is required.

The provider documents NFL odds under sport key `americanfootball_nfl`, with `h2h`, `spreads`,
and `totals` markets and decimal output. Use its documented quota headers and error responses.
Reference: [official V4 API documentation](https://the-odds-api.com/liveapi/guides/v4/), checked
2026-09-16. Recheck the adapter contract against those official docs when implementing.

Requirements:

- API key only from `ODDS_API_KEY`; never persist it in URLs, manifests, errors, fixtures, or logs.
- Request decimal prices and an explicitly configured bookmaker list. No silent bookmaker choice.
- One bounded collection operation, not an infinite polling loop. Configure request timeout,
  bounded retries, quota reserve, and maximum requests per invocation.
- Persist response bytes, receipt timestamps, non-secret request parameters, and quota metadata.
- Handle timeout, invalid JSON, empty payload, missing book, authentication failure, quota
  exhaustion, and 429 responses with structured statuses. No synthetic fallback in real mode.
- Tests use synthetic/mocked responses and no credentials; label those fixtures synthetic.

### 6.2 Extended canonical quote contract

Preserve V1 fields (`quote_id`, provider event ID, game ID, bookmaker, market, selection, line,
decimal odds, snapshot time, bookmaker update time, kickoff-at-snapshot, settlement rule).
Add provider, receipt ID, raw hash, local observed time, provenance mode, and a `pair_id`.

- Spread lines must be opposite; total lines identical; moneyline lines null.
- Decimal odds finite and greater than 1; supported line grid in integer half-points.
- A complete pair belongs to one event, bookmaker, market, line pair, receipt/snapshot, and
  settlement contract. Multiple alternate lines must not make an otherwise valid main pair
  appear incomplete. Main-market selection must be explicit; quarantine ambiguous alternatives.
- Pair sides may have separate update timestamps; preserve both and validate freshness on both.
- Stable keys make identical imports idempotent; conflicting duplicates are rejected with reasons.
- For live responses, snapshot collection time cannot be inferred from a book's update time.
- For historical CSVs, preserve supplied snapshot time and today's local import time separately.
  Late imports cannot qualify as prospective decisions, even if the CSV contains old timestamps.
- Synthetic status propagates through every derived artifact and report.

Require configured, documented settlement rules per bookmaker/market. Price data alone does
not establish whether ties void or how postponements settle. Missing rules disable paper
settlement for that market; clearly label any analytical-only comparison assumptions.

### 6.3 Event mapping and kickoff changes

Use canonical franchise aliases plus provider event ID mappings. Team names and kickoff
proximity can propose a match, but ambiguous matches must be quarantined rather than guessed.
Persist explicit crosswalks and their observed/effective times. Never join solely on team names.

Preserve kickoff-at-forecast and kickoff-at-quote. Append reschedule events when later observed;
do not rewrite earlier records or backdate a new forecast. For V2 paper results, a changed
kickoff after commitment is `rescheduled_review_required` and excluded from automatic monetary
settlement unless a previously frozen rule explicitly covers it. Report exclusions and coverage.
This conservative policy is not a claim about a bookmaker's actual rescheduling terms.

## 7. R3 — prospective protocol and forecast ledger

### 7.1 Freeze before collecting scored forecasts

A prospective protocol is an immutable **epoch** containing:

- Activation time assigned by the system; no user-supplied backdating in real mode.
- Exact champion and challenger bundle hashes and their creation times.
- Code digest covering all prediction, ingestion, timing, matching, evaluation, settlement, and
  CLI orchestration paths; dependency-lock digest and full resolved configuration.
- Training-source manifest hash, feature/contract versions, calibration history, selected
  hyperparameters, seed, evidence modes, and champion/challenger roles.
- Game scope: NFL regular season only; target horizon 24 hours before kickoff.
- Timing tolerance, market policy, selection rule, settlement assumptions, and review thresholds.
- Planned evaluation population: all eligible games after activation; predeclared review after
  at least 100 matched settled games and eight distinct season-week blocks, or end-of-season
  descriptive report if less. These are review thresholds, not proof of adequate statistical power.

Use a separate prospective freeze from V1's fixed-source historical holdout protocol. Future
input snapshots cannot be hashed before they exist: freeze the selection policy and training
bundle, then hash the selected sources in each forecast record. New live data must not invalidate
the epoch. Changes to code, model, scoring rules, or policy require a new epoch and explanation.
Never pool epochs silently or repeatedly test until a favorable interval appears.

### 7.2 One-shot operation and timing

Implement `prospective-tick`, which reads the active epoch, collects permitted snapshots,
identifies due games, writes forecasts, and exits. Document how an external scheduler could call
it every five minutes, but do not install or start one automatically.

Default target window: kickoff minus 24 hours, plus/minus five minutes. A forecast is eligible
only if its information cutoff AND successful commitment time fall within that window and
strictly before kickoff. The information cutoff is the real current time after required input
collection, frozen at forecast start. Actual commitment time is recorded after durable publication.
Display the actual horizon; do not pretend every record is exactly 24 hours before kickoff.

- No required source receipt or bundle creation time may exceed the information cutoff.
- Forecast computation uses pinned inputs selected at that cutoff, even if data arrives later.
- If the window is missed, record `missed_forecast_window`; no retrospective replacement in
  the prospective sample. An optional later forecast is explicitly off-policy/descriptive.
- No games already started or known final. A later correction revealing an already-started game
  marks it ineligible in evaluation via an appended event, without deleting the original.
- Use an injected clock for tests only; real commands cannot accept arbitrary creation times.
- Prefer creating both champion/challenger records atomically for the game; if one fails, retain
  the failure reason and do not silently substitute another model. Forecast validity and paired
  evaluation eligibility are separate statuses.
- A missing odds quote must not prevent a valid football forecast. Market coverage is separate.

### 7.3 Immutable records and crash recovery

Uniqueness: `(protocol_id, game_id, horizon_policy_id, model_bundle_hash)` for on-policy forecasts.
Re-running a tick returns the existing committed record. It never produces duplicate paper bets.

Each forecast must reference/store:

| Field group | Required content |
|---|---|
| Identity | forecast ID, protocol ID, game ID, season/week, team IDs, model role/ID/hash |
| Times | target cutoff, actual information cutoff, created/committed times, kickoff as known |
| Inputs | selected receipt IDs/hashes, feature-row hash and saved feature rows |
| Outputs | score means, full joint PMF or lossless reproducible PMF artifact, marginals, prices |
| Provenance | code/config/lock hashes, evidence modes, synthetic flag, quality flags |
| Market decision | exact paired quote IDs, probabilities, EV, decision status/reason, if eligible |
| Integrity | payload hash, manifest hash, atomic commit marker |

Store outcomes and corrections separately, linked by game ID and result version. Never add
actual scores to the immutable forecast payload. A report combines the two using an explicit
results-as-of timestamp. Pending games remain pending, not zero-valued losses or automatic voids.

Use a writer lock and commit markers to survive interruption. An uncommitted partial run is not
eligible. On recovery, reuse a fully committed record or record failure; never backdate completion
from an earlier attempt. A local hash proves content consistency, not external notarization;
describe the ledger as locally auditable, not tamper-proof or independently certified.

### 7.4 Paper selections are frozen before outcomes

For each model, save all eligible market probabilities and one deterministic proposed selection
per game at most: greatest EV >= 0.03, one flat unit, ties broken by fixed market/selection order.
Persist even `no_bet` and `market_unavailable` decisions. Use V1's formula including refunds:

`EV = p_win * (decimal_odds - 1) - p_loss`.

Record the chosen quote's observed price, not a no-vig price. Champion and challenger paper
ledgers are separate comparisons, not a combined betting portfolio. Never choose bets after
seeing actual outcomes or closing lines. No Kelly sizing, bankroll optimization, or bet execution.

## 8. R4 — market benchmarks and settlement

### 8.1 Matching rules

At the information cutoff, select the newest complete eligible pair from the configured reference
book. Both local observation and provider snapshot must be <= cutoff for prospective use;
book update must be <= snapshot and nonmissing. Both snapshot and book update must be no more
than 60 minutes old. No cross-book, cross-line, or cross-time splicing.

Keep exclusions per market, not just a game-level boolean. Example reasons: no quote, stale
snapshot, stale update, unknown rule, incomplete pair, ambiguous join, late local observation,
conflicting prices, model unavailable, or invalid forecast timing.

For two decimal prices d1/d2, use proportional de-vigging:

`r1 = 1/d1; r2 = 1/d2; q1 = r1/(r1+r2); q2 = 1-q1`.

Record `overround = r1+r2-1` and the de-vig method. These are inferred market probabilities,
not known true probabilities. Do not invent a market joint score distribution from three lines.

### 8.2 Comparable evaluation

For the SAME matched games, cutoff, line, rule, and model IDs, compute:

- Moneyline binary log loss and Brier: non-tied games only; both model and market probabilities
  conditional on no tie under the supported tie-void contract.
- Spread binary log loss and Brier: evaluate home cover at the book's actual handicap; exclude
  pushes from this conditional metric, and report push counts separately.
- Total binary log loss and Brier: over at the book's actual total; exclude pushes likewise.
- Keep model three-way log loss, tie probability, and push calibration as separate diagnostics.
  Two-way odds alone cannot supply market tie/push probabilities.
- Report model-minus-market paired differences, 95% season-week block bootstrap intervals,
  2,000 replicates, fixed seed, exact sample counts, and unique game counts.
- Score only one orientation per market (home/over); do not double-count complementary sides.
- Clip only inside log-loss evaluation at documented epsilon 1e-12; retain original predictions
  and counts of clipped values. Do not alter prices to hide extreme probabilities.
- Suppress confidence intervals below eight blocks with `insufficient_blocks`; still show
  descriptive estimates and sample sizes. Separate this from the 100-game planned-review gate.

Report all-game model metrics and matched-subset model metrics side by side so missing odds
cannot silently change the evaluation population. Coverage denominators include all scheduled
in-scope games, forecast attempts, valid forecasts, matched markets, and settled matched games.

Line differences remain descriptive price-reference differences, not proof of error or edge.
Do not label sportsbook spread/total as the true conditional expected margin/total.

### 8.3 Closing proxy and CLV

Use the newest complete pair observed strictly before kickoff within the final 30 minutes,
from the same book and compatible settlement contract. Enforce update/observation validity
and freshness here too; do not merely reuse a pair-completeness test. If no eligible close was
collected, closing value is unavailable. No substitution of a post-kickoff or another-book quote.

Line CLV, positive favorable to the selected side:

- Spread: `bet_selected_side_handicap - close_selected_side_handicap`.
- Over: `closing_total - bet_total`.
- Under: `bet_total - closing_total`.

Price CLV only when the line and settlement contract match:
`bet_decimal_odds * closing_no_vig_probability_for_selection - 1`.
For push-capable lines, label this a conditional non-push price comparison, not unconditional EV.
No price CLV across different spread/total lines. Report sample counts separately for each
market and CLV type; do not average spread and total line movements into one ambiguous number.

### 8.4 Outcome accounting

Support pending, win, loss, push, void, and review-required statuses. Pending does not enter
settled ROI. A push contributes zero profit and its stake to the non-void settled denominator;
a void contributes neither. Final scores include overtime, consistent with the configured market.
Retain all result revisions and produce a new report version when a correction changes settlement.

Report fixed-unit profit, settled non-void stake, ROI, chronological maximum drawdown, signal
counts, and coverage. Bootstrap ROI as a ratio of sums by season-week, preserving all decisions
in each block and reporting undefined zero-stake resamples. Show no confidence interval when
the block threshold fails. Never present theoretical EV or positive CLV as realized profit.

## 9. R5 — bounded joint-distribution challenger

### 9.1 Design choice

Keep the V1 score regression and rounded-normal PMF as the baseline. Add an experimental
four-parameter exponential adjustment of its **joint** score probabilities. This is a proposed
model to test, not a proven fix or a complete football scoring simulator.

For nonnegative integer scores h and a, define M=h-a and T=h+a:

```text
f(h,a) = [ I(M=0), I(abs(M)=3), I(abs(M)=7), I(T is even) ]
P_theta(h,a | game) = P_V1(h,a | game) * exp(theta dot f(h,a)) / Z_game(theta)
```

The shared +/-3 and +/-7 adjustments avoid a gratuitous home/away asymmetry. Four parameters
limit search scope. This can improve aggregate tie/key-number/parity behavior, but does not
remove all impossible scores or model overtime possessions. State those remaining limitations.

Derive **all** probabilities, means, marginals, fair lines, and intervals from the adjusted joint
PMF. Never change tie probability independently and leave spreads/totals inconsistent. The
adjustment can change expected scores; report both pre-adjustment regression outputs and final
distribution means distinctly. Do not force the means back afterward with an undocumented step.

### 9.2 Fitting and numerical rules

Fit theta by minimizing calibration-game average negative log probability of the actual joint
score plus `lambda * sum(theta**2)/2`. Use deterministic optimization, analytic or checked
gradients, log-sum-exp normalization, zero initialization, and bounds [-3, 3] on each coefficient.

Predeclared search: lambda in `[0.01, 0.1, 1.0]`, plus the identity distribution `theta=0`.
Do not add features, widen bounds, add candidate families, or search more hyperparameters after
seeing results. Such a change is a separate research protocol, not finishing this V2 task.

For each evaluation season s:

1. Score model trained only on seasons before s, retaining V1 fitting conventions.
2. Base residual distribution estimated only from earlier chronological OOF residuals.
3. Calibration examples come from up to the three latest seasons before s with valid
   chronological OOF **base distributions**; require at least 250 eligible calibration games.
4. For a calibration game in season c, its base distribution must itself use a score model fit
   before c and residual parameters fit from seasons before c. Its outcome cannot have been
   used to build its own base distribution. Save the nested provenance.
5. Fit each lambda's theta using that earlier calibration pool, then evaluate season s once.

If the minimum prior residual/calibration history is unavailable, record an exclusion. Do not
relax sample thresholds or use same-season outcomes. A base distribution built from residuals
including its calibration target is leakage even if the final evaluation season is later.

Optimization failure, invalid probabilities, or insufficient history means candidate unavailable
with a reason. Do not silently label the identity model as a successfully fit challenger.
Report bound hits, solver status, iterations, gradient norm, pool seasons/game hashes, objective,
and fitted coefficients in the bundle. The identity candidate is explicitly named and valid.

Use the V1 expandable score grid with its tail limit. For adjusted distributions, bound omitted
mass conservatively using the minimum/maximum adjustment weights and base omitted-mass bound;
expand until the adjusted bound is <=1e-8, or fail at the existing hard support limit. Store
both base and adjusted tail diagnostics. No dropping outlier actual scores to improve fit.

### 9.3 Research evaluation and selection

Use 2019–2023 as V2 development folds, subject to the prior-history eligibility checks above.
Treat them as reused research data. Reconstruct corrected V1 and B0 alongside the challenger
on exactly the same games, in new V2-labeled runs. Do not compare new challenger outputs solely
with the preserved original dirty-tree V1 report.

Primary candidate selection metric: pooled margin CRPS, lower is better. For lambda scores
within 1e-6, prefer stronger regularization; if identity ties, prefer identity.

For a non-identity candidate to be designated ready for prospective shadow comparison, require
on development matched games:

- Strictly lower margin CRPS than the reconstructed V1.
- Total CRPS and three-way log loss no more than 1% worse than V1.
- Smaller mean absolute predicted-versus-observed frequency gap across M=0, |M|=3, |M|=7.
- No numerical-integrity failure or unexplained candidate-only exclusions.

These are engineering/research screening choices, not guarantees or significance thresholds.
Report full paired uncertainty and every candidate, including rejected ones. If no candidate
passes, record `retain_v1`; the implemented challenger and honest negative result still ship.

After fixing the choice, evaluate 2024 and 2025 as **previously inspected retrospective checks**.
Do not use them to retune. Record deterioration and retain V1 as champion. Future post-activation
forecasts provide the new evidence; no season label alone makes a test prospective.

Include score/margin/total MAE, margin/total CRPS, three-way log loss/Brier, reliability with counts,
50/80/95% margin AND total coverage and average interval widths, tie rate, +/-3, +/-7, combined
absolute key margins, and total parity. Cover early-season and neutral-site subsets with counts.
Sparse events such as ties must be accompanied by observed counts, not just a percentage.

### 9.4 Prospective model lifecycle

Fit final bundles using completed prior seasons only (initial 2026 collection: through 2025).
Use eligible prior OOF calibration distributions and the fixed selected lambda. Record actual
creation time. Freeze parameters for the epoch; rolling input features may update as games
become eligible. No automatic weekly parameter retuning in V2.

At the planned review, compare V1/challenger on paired prospective games and compare both with
the market where quotes exist. Produce an evidence packet and recommendation, not automatic
promotion. Any later model/policy changes start a new epoch. An unsuccessful challenger does
not prevent ongoing V1 prospective collection and market benchmarking.

## 10. R6 — dashboard and understandable reporting

Extend the existing read-only dashboard. Keep the existing pages and add a Prospective page;
avoid a visual redesign or unrelated frontend work.

### Required user-facing changes

- Human-readable labels and tooltips for raw field names. Percent formatting for probabilities;
  signed spreads/moneylines; clear points versus unitless probability-loss metrics.
- Persistent evidence badge: synthetic, retrospective reconstruction, legacy observation
  metadata, or prospective local recording. Never display a generic green “verified” badge.
- Show selected protocol, model roles, cutoff, commitment time, actual horizon, source freshness,
  and latest successful collection. Distinguish “no odds” from “no model prediction.”
- Slate: V1/challenger fair lines, reference-book lines, matched status, key warnings. Do not
  use a “recommended bet” label; use “paper signal under frozen policy.”
- Game detail: side-by-side base/adjusted distributions, key-margin and tie probabilities,
  interval explanations, selected feature provenance, matched quote timestamps and rules.
- Evaluation: V1/B0/challenger metrics on identical games; market comparison by market; all-game
  versus matched-game coverage; paired intervals and sparse-sample explanations.
- Prospective: eligible/missed/pending/settled counts, cohort/epoch filter, collection failures,
  missing-source reasons, frozen paper decisions, drawdown/CLV with denominators, next review gate.
- Audit: snapshot/forecast IDs, content hashes, bundle/calibration lineage, protocol digest,
  result revision, exclusion exports, integrity verification status and limitations.
- Never merge synthetic and real records or epochs by default. Never silently change run selection.
- No future actuals in forecast-only views. Empty/missing optional artifacts get useful messages.
- Extend `docs/dashboard_guide.md` to explain what each page measures and what it cannot prove.

## 11. R7 — CLI contract and configuration

The following NEW commands/options are requirements to implement, not claims that they exist
today. Retain existing V1 commands. Prefer extending compatible commands over duplicate engines.
Every command needs help text, machine-readable output, deterministic exit/status behavior, and
tests. Mutating real commands require an explicit config; offline demo must be credential-free.

### Core commands

```bash
# Inspect/migrate local snapshot metadata without rewriting V1 blobs.
uv run nfl-origination snapshot-inventory --config configs/v2_prospective.yaml
uv run nfl-origination migrate-snapshots --config configs/v2_prospective.yaml

# Synthetic end-to-end lifecycle, with controlled test clock and no network.
uv run nfl-origination demo-v2 --config configs/v2_demo.yaml --offline

# Retrospective distribution experiment, reading cached real data.
uv run nfl-origination research-v2 --config configs/v2_research.yaml --offline

# Collect one real odds snapshot; needs ODDS_API_KEY and configured book/rules.
uv run nfl-origination collect-odds --config configs/v2_prospective.yaml

# Fit compatible V2-era bundles, then freeze an epoch.
uv run nfl-origination fit --through-season 2025 --config configs/v2_prospective.yaml
uv run nfl-origination freeze-prospective --config configs/v2_prospective.yaml

# Execute one bounded cycle using the frozen epoch.
uv run nfl-origination prospective-tick --config configs/v2_prospective.yaml

# Append final results/corrections; evaluate committed forecasts only.
uv run nfl-origination settle-prospective --config configs/v2_prospective.yaml
uv run nfl-origination report-prospective --config configs/v2_prospective.yaml
uv run nfl-origination verify-ledger --config configs/v2_prospective.yaml

uv run streamlit run src/nfl_origination/dashboard/app.py
```

Examples use the initial 2026 deployment; do not hard-code that season into business logic.
Research writes a decision record. Fit resolves its exact selected settings; if research is
unavailable it may explicitly fit V1-only collection, never invent challenger parameters.
Freeze prints/stores the exact protocol ID in a local active-epoch pointer; operational commands
also accept `--protocol-id` to avoid ambiguous selection. Switching the pointer is an explicit
action, not a side effect of inspecting an old run.

`prospective-tick` may collect closing snapshots for approaching kickoffs even when no forecast
is due. Enforce request budgets and report what was attempted. Final-result retrieval must be
explicit and distinguish new observations from cached settlement. `--offline` replay commands
never perform network calls. Replay cannot produce real prospective eligibility.

Provide strict configuration fields for:

| Group | Required settings / defaults |
|---|---|
| Storage | Isolated V2 artifacts/reports and receipt/ledger roots |
| Evidence | Research reconstruction vs observed prospective; synthetic explicit |
| Models | Champion ID/hash, challenger decision reference, fixed score alpha 100 |
| Challenger | Four features, lambdas above, bounds, min 250 games, max three calibration seasons |
| Horizon | 24 hours, +/-5 minute commitment window, strict before-kickoff |
| Sources | 48-hour eligibility lag, required datasets, adapter/observation policy |
| Market | Enabled false initially, provider, explicit book, settlement rules, max age 60 minutes |
| Collection | Timeout, <=3 attempts/request, request budget, quota reserve; no embedded secrets |
| Closing proxy | Same book, final 30 minutes, strictly pre-kickoff |
| Paper | EV >=0.03, one unit, <=1 selection/game/model, deterministic tie order |
| Evaluation | 2,000 bootstrap replicates, fixed seed, min eight blocks, planned 100-game review |

Default configuration must safely run football-only collection without odds credentials. Enabling
markets without a book or rules must fail configuration validation with a useful message.
Do not silently fill these fields with a guessed user's preferred sportsbook.

## 12. Implementation phases and acceptance gates

| Phase | Work | Exit evidence |
|---|---|---|
| A | Baseline inventory; schema contracts; source receipts and as-of lookup | R1 tests pass; original artifact hashes unchanged |
| B | CSV extension; mocked live adapter; pairing/mapping/freshness | Real-shaped synthetic fixture roundtrip, secret-redaction and rejection tests |
| C | Prospective protocol, ledger, clock/commit rules, one-shot tick | Two synthetic games committed; duplicate/restart/late cases handled |
| D | Settlement, matched market benchmarks, CLV, coverage | Known outcomes produce hand-checked probabilities, returns, and exclusions |
| E | Joint adjustment, nested chronological calibration, selection report | Identity invariance, no-leakage and numerical tests; candidate decision recorded |
| F | Wire candidate bundles into prospective flow; dashboard/report/docs | Offline full lifecycle and legacy dashboard both pass |
| G | Full verification and handoff | Commands, evidence, deviations, limitations, and external dependencies in recap |

Finish and test each phase before moving on. Do not require user approval between ordinary
phases. Start prospective engineering before extended model research so a disappointing
challenger does not strand the useful collection system.

## 13. Required tests

Use focused tests of observable contracts and real failure cases. No arbitrary test-count goal.
All automated tests run offline; integration tests use temporary paths, an injected clock, small
synthetic seasons, and deterministic HTTP fixtures. Existing V1 tests remain meaningful.

### Temporal and provenance

- A observed before cutoff, B after: choose A. A -> B -> A: choose by observation sequence.
- A revision received after kickoff never changes an already committed forecast or its inputs.
- Identical repeat fetch preserves content identity and earlier receipts without backdating.
- Missing/partially missing observation times remain ineligible through offense/opponent/prior
  aggregation. Selected historical schedule supplies rest and kickoff metadata.
- Late source receipt, late model creation, late commitment, and missed window each reject the
  prospective claim with distinct reasons. Future `--as-of` cannot bypass the real clock.
- Source selection and effective observation changes invalidate relevant feature caches.
- Legacy migration cannot create an earlier observation timestamp; original bytes unchanged.
- Live snapshots may evolve under the epoch; changed algorithm/config/bundle/lock fails verify.

### Ledger and recovery

- Duplicate tick/import is idempotent; conflicting duplicate rejected; two concurrent writers
  cannot commit two forecasts or bets for one uniqueness key.
- Interrupted write leaves no eligible partial forecast. Recovery cannot backdate commitment.
- Results settle existing forecasts without modifying them; corrected result creates a new
  version, and old reports remain reproducible.
- Pending, canceled, rescheduled, and incomplete-final-data cases do not become ordinary losses.
- Tampered payload or missing blob produces integrity failure and exclusion from scored reports.
- Synthetic/replay records cannot enter real prospective reports; epochs not silently pooled.

### Market arithmetic and matching

- Whole-point and half-point spread/total, both sides, favorite/underdog, tie and push cases.
- Hand-computed example: d=1.91, p_win=.55, p_loss=.40, p_push=.05 gives EV=.1005 units.
- Away spread CLV: bet +3.5, close +2.5 => +1.0. Home -3.5, close -4.5 => +1.0.
- Over 44.5 vs close 46 => +1.5; under 47.5 vs close 46 => +1.5.
- Same-line price CLV available; different-line price CLV unavailable.
- Complementary pairs, alternatives, mixed books/times, stale/missing update, late CSV observation,
  post-kickoff close, ambiguous event and unknown settlement rule all have explicit outcomes.
- Paired scores use the same game IDs, one orientation and exact line; no double counting.
- No-push/no-tie conditional metrics exclude refunded outcomes; full model diagnostics retain them.
- No quotes yields null market metrics and correct coverage, not zero loss or zero ROI.
- Small block sample suppresses interval; zero-stake bootstrap samples counted as undefined.

### Distribution and leakage

- theta=0 reproduces V1 PMF, marginals, prices, and intervals within 1e-8 probability tolerance.
- Nonnegative finite mass sums to one; home win + tie + away win =1; settlement outcomes sum to1.
- Home/away swapping transposes the adjusted joint PMF for correspondingly swapped base inputs.
- Margins/totals computed from joint PMF agree with pricing; quantile intervals are nested.
- Negative tie coefficient reduces tie probability on a controlled PMF; shared |3| adjustment
  affects both signs as specified. These mechanics tests do not assert real-world improvement.
- Adjusted omitted-mass bound enforced, including extreme parameter/base-distribution fixtures.
- Changing an evaluation outcome does not change its earlier forecast, calibration pool, fitted
  parameters or chosen bundle. Each calibration base PMF excludes its own target from fitting.
- Insufficient history and optimizer failure cannot silently fall back under a challenger label.
- Market-column contamination is rejected at model/feature boundaries.

### End-to-end and UI

- Offline demo: receipts -> bundle/protocol -> forecast/quote commit -> closing receipt -> final
  outcome -> settlement -> report -> integrity verify. Include one missed game and one missing book.
- Run twice to verify deterministic numerical outputs and idempotent records with fixed test clock.
- Streamlit smoke: old V1, new research, prospective pending, settled, missing-market, and synthetic
  runs render without exceptions and with correct evidence labels.
- Dashboard refresh performs no network call or fit; artifact loading respects model and protocol IDs.

## 14. Verification commands and required evidence

Run from repository root after implementation:

```bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy src/nfl_origination
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run pytest -q
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run nfl-origination demo-v2 --config configs/v2_demo.yaml --offline
```

Verify all documented new commands through `--help` and the applicable fixture workflow. The
network-blocking environment must cover the new adapter too. Record actual command, exit code,
test count, duration, run/protocol IDs, and artifact paths. Do not claim live integration tested
based only on mocked HTTP. No fresh full-suite run is required after documentation-only edits.

If cached historical data supports it, execute the bounded V2 research experiment and export
all candidates. If unavailable, finish software tests and report `real_research_not_run` with
the exact prerequisite and command. Neither a missing API key nor future games is a reason to
leave offline software unfinished or to substitute synthetic statistical conclusions.

## 15. Deliverables and final acceptance checklist

- [ ] Additive V2 implementation with legacy behavior/readability preserved.
- [ ] Strict versioned contracts and migration/inventory report.
- [ ] Immutable snapshot history with tested as-of selection.
- [ ] Optional provider adapter, extended CSV path, identity/rule/freshness validation.
- [ ] Frozen prospective epoch, safe tick command, immutable forecast and decision ledger.
- [ ] Versioned results/settlement, aligned market benchmarks, closing proxy and CLV.
- [ ] Joint-distribution challenger with nested chronological calibration and documented decision.
- [ ] Full offline demo, regression tests, dashboard smoke coverage, passing project checks.
- [ ] Updated README quickstart, operations guide, data contracts, model card, dashboard guide.
- [ ] Human-readable research report and separate prospective report with coverage and evidence labels.
- [ ] Original V1 artifacts/protocols verified unchanged.
- [ ] `V2_IMPLEMENTATION_RECAP.md` containing summary, changed modules, requirement-to-test mapping,
      exact commands/results, model selection table, deviations, known limitations, and next actions.

The recap must answer plainly:

1. What can Luke run now without credentials?
2. What needs a configured bookmaker, supported rules, and API key?
3. Is collection actually running, or has only a one-shot command been implemented/tested?
4. How many REAL forecasts were committed before kickoff, and how many have settled?
5. Did the challenger improve the declared metrics, or did we retain V1?
6. Which results are retrospective, synthetic, prospective, unavailable, or inconclusive?
7. What exact evidence is still needed before saying the model competes with sportsbook prices?

## 16. Explicit non-goals

- Real-money wagering, account login, bet submission, execution/fill assumptions, or guaranteed profit.
- Full sportsbook operation: exposure limits, customer profiling, liability-based line moves.
- Live/in-play markets, playoffs, player props, parlays, correlated same-game pricing.
- Injury feeds, quarterback projections, weather, depth charts, new football feature families.
- Market odds as predictive model inputs or market/model blending.
- Drive/possession simulation, explicit overtime simulator, neural networks, gradient boosting,
  Bayesian hierarchy, or an open-ended model tournament.
- Multi-book best-price optimization, multi-provider collection, historical odds purchases.
- Production hosting, cloud deployment, scheduled-task installation, external notifications.
- Retroactive certification of historical nflverse data or local timestamps as independent proof.
- Rewriting V1 reports, retuning on 2025 while calling it untouched, or claiming synthetic ROI.

Deferred work can be listed in the recap, but is not part of completing V2.

## 17. Source references and interpretation

- Existing repository code and V1 documents are the compatibility reference.
- [nflverse project](https://nflverse.nflverse.com/) and
  [nflverse data releases](https://github.com/nflverse/nflverse-data) remain football-data sources.
- [The Odds API V4 documentation](https://the-odds-api.com/liveapi/guides/v4/) is the optional
  adapter contract; do not infer historical access, pricing, or settlement terms from examples.

The statistical challenger, screening thresholds, timing tolerance, and review population above
are deliberate V2 design decisions. They are not claims endorsed by a data provider or validated
results. The purpose of this implementation is to make those decisions testable and auditable.
