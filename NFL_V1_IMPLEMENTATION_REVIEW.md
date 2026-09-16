# NFL origination V1 — implementation review

Reviewed September 16, 2026. Repository: `/Users/luke/origination-model`. Reviewed HEAD: `32b16ea`; `RECAP.md` was untracked. This was a read-only review of the project. No source, configuration, reports, or holdout records were changed.

## Verdict

Claude Code did a good job implementing a substantial first version. Keep this foundation. The implementation follows much of the specification, includes actual research artifacts, and does not pretend synthetic betting results establish profitability. The modular structure is appropriate for the task.

However, I would not mark all acceptance criteria complete yet. I confirmed seven issues below, including two important correctness problems: incomplete point-in-time safeguards and a market comparison that can use a different model's distribution. Passing the current tests does not cover these cases.

The existing historical-reconstruction results remain useful research evidence. These findings do not, by themselves, show that the reported 2025 numerical results are wrong. They do limit claims about point-in-time correctness, experiment provenance, and forecast/market functionality.

## What I checked

- Read `RECAP.md`, the repository specification, working agreement, review packet, model card, experiment protocol, and saved report metadata.
- Inspected data normalization, aggregation, feature building, training splits, residual fitting, distribution/pricing, evaluation, forecast orchestration, protocol verification, and optional market paths.
- Independently ran the 100 unit tests with the project's Python environment, network blocking enabled, bytecode disabled, and pytest cache disabled. All passed.
- Ran small in-memory reproductions of the defects described below. They did not write into the repository.
- Recomputed primary metrics directly from the saved 272 holdout predictions: score MAE **7.4387830285**, three-way log loss **0.6706428143**. These match the report.

I did not rerun the full model backtest, open the holdout again through the experiment runner, or independently run the entire integration/end-to-end suite. Claude Code's full final test run remains outside this review's verification. The recap reports an earlier 109-test pass; that is distinct from my independently observed 100-unit-test pass.

## Prioritized findings

### R1 — P1: `recorded_asof` does not protect every feature input

**Locations:** [league prior construction](/Users/luke/origination-model/src/nfl_origination/features/builder.py:179), [aggregate observation timestamp](/Users/luke/origination-model/src/nfl_origination/features/aggregate.py:119), [label construction](/Users/luke/origination-model/src/nfl_origination/features/builder.py:250).

Team history is filtered by observation time, but league priors are calculated from all rows in the selected previous seasons. The prior cache is keyed only by season, not the observation cutoff. Those prior inputs are also omitted from the feature row's source IDs and maximum observation timestamp.

**Reproduction:** I marked every synthetic historical row as first observed in 2030 and generated a 2012 forecast in `recorded_asof` mode. Both teams correctly had zero eligible history, yet `require_forecastable` accepted the features. Changing the unavailable prior-season points changed `team_points_for` from **22.953125 to 122.953125**, while the reported maximum observation time remained null. Unavailable information therefore directly changes a supposedly as-of forecast.

A second timestamp problem compounds this: aggregation attaches the schedule observation time and discards the PBP observation time. With schedule observed in 2010 and PBP first observed in 2030, the resulting PBP-derived team-game features were stamped **2010**. Labels similarly use kickoff plus the fixed lag without enforcing actual observed availability.

**Change:** Apply eligibility to every dependency: priors, rolling metrics, schedule metadata, results/labels, and residual inputs. Preserve the maximum observation time across the source versions actually used, not just the schedule timestamp. Include prior dependencies in provenance. Select historically eligible cached versions, and fail clearly when sufficient observed history does not exist. Audit rest-feature metadata as part of this same boundary.

**Regression tests:** Entirely late-observed history must not produce an accepted observed-as-of forecast; mutating late-observed prior data must not change one; late PBP with an early schedule must remain ineligible; late-observed labels must not enter fitting. Test the end-to-end feature/fit path, not just `AsOfPolicy.eligible_mask`.

**Impact on existing report:** The published run is labeled `historical_reconstruction`, so this does not establish leakage in its ordinary annual historical splits. It does invalidate the broader claim that recorded-as-of protection is complete. Reopen A03 and the associated part of A09.

### R2 — P1: Market comparison can use B0 probabilities under the M1 label

**Locations:** [comparison CLI loading](/Users/luke/origination-model/src/nfl_origination/cli.py:351), [distribution lookup](/Users/luke/origination-model/src/nfl_origination/market/compare.py:18).

Forecast runs save a combined `distributions.parquet` containing all models. The comparison command prefers this combined file, while the lookup selects only by game ID and takes its first row. The first row is B0, even when the selected prediction is M1.

**Reproduction using saved artifacts:** For `2026_02_DET_BUF`, the prediction model is `M1_ridge_alpha100`, but lookup returns the `B0_league_baseline` distribution. At a home handicap of −3.5, the displayed model cover probability would be **0.4616716**, whereas M1's own distribution gives **0.5624190**. That is a material difference of about ten percentage points.

**Change:** Join distributions using at least `(game_id, model_id)` and, where multiple forecast versions are possible, cutoff/prediction ID as well. Require exactly one match. Prefer per-model artifacts when appropriate, but do not rely on file order as a correctness guarantee. Validate the loaded PMF.

**Regression tests:** A multi-model forecast comparison must equal the selected model's direct pricing result. Reordering distribution rows must not change the answer. Missing/ambiguous model matches must fail explicitly.

**Impact:** Optional market comparison is incorrect for this forecast artifact layout. This does not alter the existing reported historical score metrics; no real odds comparison has been claimed. Reopen A10 and part of A09.

### R3 — P2: The frozen protocol records Git metadata but does not verify code

**Locations:** [protocol checksum payload](/Users/luke/origination-model/src/nfl_origination/protocol.py:43), [verification](/Users/luke/origination-model/src/nfl_origination/protocol.py:105).

The protected payload includes configuration, data hashes, and exclusions, but no code digest, dependency-lock digest, or feature/schema implementation identity. `git_commit` and `git_dirty` are metadata outside the checked payload. Altering fitting or pricing code therefore need not invalidate the protocol.

**Reproduction:** Verification passed when the Git-identity function was replaced with a different commit/dirty identity; verification never called it. The saved holdout and freeze artifacts both report `git_dirty=true`, so their recorded commit alone cannot reconstruct the exact working code used at freeze time.

This is not evidence of deliberate holdout misuse. It means the advertised integrity gate does not enforce the full specification.

**Change:** Freeze a deterministic digest of the relevant source tree, schema/feature definitions, and `uv.lock`, or require a clean, immutable committed state and verify it. Prefer a scoped source digest so report generation does not itself invalidate the lock. Save a retrievable code snapshot when a dirty tree is allowed. Record incompatible reruns as new, explicitly labeled experiments.

**Regression tests:** A numerical implementation change with unchanged YAML/data must be rejected; a dependency-lock change must be detected; harmless report output changes should not silently redefine the research implementation.

**Existing holdout handling:** Preserve the original artifacts. Do not fabricate a clean historical provenance record or silently regenerate them. If code recovery is impossible, disclose that limitation and record any corrected reproduction separately. Reopen the provenance portion of A07/A08.

### R4 — P2: Forecasting does not verify bundle/configuration compatibility

**Locations:** [forecast feature building and bundle use](/Users/luke/origination-model/src/nfl_origination/experiment.py:953), [single-game prediction validation](/Users/luke/origination-model/src/nfl_origination/models/bundle.py:182).

Forecast features come from the current configuration, while coefficients and residual parameters come from whichever saved bundles are in the season directory. The path does not check that feature settings, availability policy, or data mode match the fitted bundle. A stored `config_hash` is not compared. Changing rolling windows or shrinkage can therefore feed a differently defined variable into the same trained coefficients without an error.

**Reproduction:** A saved `historical_reconstruction` bundle accepted feature rows labeled `recorded_asof`. No compatibility error was raised. In the slate path the reported mode comes from the current configuration, creating a risk of relabeling a model with a stronger provenance claim than its training supports.

**Change:** Store the resolved feature/data-policy contract in each bundle. Compare a semantic compatibility hash during inference, while allowing irrelevant differences such as output paths. Record training provenance separately from forecast-input provenance; do not equate prospective inputs with prospectively observed historical training. In strict recorded-as-of use, enforce model availability before decision time as well.

**Regression tests:** Changed history length/shrinkage must fail until a compatible bundle is supplied; a data-mode change must be rejected or explicitly reported as mixed provenance; changing only an output directory should remain allowed.

### R5 — P2: Future information cutoffs are labeled as standard-horizon forecasts

**Locations:** [slate target selection](/Users/luke/origination-model/src/nfl_origination/experiment.py:896), [future-cutoff handling](/Users/luke/origination-model/src/nfl_origination/experiment.py:963).

Without `--as-of`, the forecast command uses each game's future kickoff-minus-24-hour cutoff even when generating the prediction earlier. A warning flag acknowledges this, but the forecast is still labeled `kickoff_minus_24h` and downstream odds selection uses that future cutoff. A prediction made now could consequently be compared against prices that become available later.

**Saved-artifact evidence:** The Week 2 example was created at **2026-09-16 17:53:05 UTC**. Its first cutoff is **2026-09-17 00:15 UTC**, and all **16** game cutoffs are later than artifact creation. They are labeled standard-horizon predictions despite using the then-current cache.

Additionally, when `--as-of` is omitted, the target selector does not exclude already-started games. A historical week can be presented through the upcoming-slate interface without an explicit reconstruction request.

**Change:** Separate “forecast using information available now” from “historically reconstruct kickoff-minus-24h.” For an upcoming game whose standard cutoff has not arrived, use actual generation time and label it `custom_horizon`, or defer the standard forecast. Exclude started games from ordinary live forecasts. Prevent paper decisions and quote selection from occurring before model availability or after the claimed observed information time.

**Regression tests:** A forecast generated before its standard cutoff must not acquire tomorrow's quote timestamp; default forecasts must exclude completed games; explicit historical reconstruction must remain available under a distinct mode.

### R6 — P2: Away-spread line CLV has the wrong sign

**Location:** [CLV formula](/Users/luke/origination-model/src/nfl_origination/market/settle.py:34).

The stored handicap is already relative to the selected team. For both home and away spreads, favorable line improvement is `bet_handicap - close_handicap`. The away branch incorrectly reverses this.

**Reproduction:** Betting away +3.5 before it closes +3 is a favorable **+0.5-point** difference. The function returns **−0.5**.

**Change:** Use the selected-side difference for both spread selections. Preserve the separate over/under formulas.

**Regression tests:** Cover home and away favorites and underdogs, favorable and unfavorable changes. Verify the complete paper-backtest output, not just the helper.

### R7 — P2: The recommendation overstates statistical support

**Location:** [model card recommendation](/Users/luke/origination-model/docs/model_card.md:54); the same claim appears in `RECAP.md`.

It says bootstrap intervals exclude zero for every headline error/probability metric across phases. Its own holdout total-MAE interval is **[−0.69509, +0.01773]**, which includes zero. Several other headline measures are reported without such intervals.

**Change:** State specifically which measured differences have intervals below zero. For total MAE, say the point estimate improved but the interval includes no improvement. M1 can still be the preferred research model; that does not require claiming conclusive improvement on every metric.

Keep claims about the EPA-free ablation similarly measured: its closeness is evidence that the displayed performance is not heavily dependent on EPA features, not proof all historical data-revision concerns have disappeared.

## What was done well

- Clear separation between football model inputs and market prices, with explicit feature allowlists.
- Chronological annual training and grouped home/away rows rather than random splitting.
- Residual estimation from earlier out-of-fold predictions, rather than optimistic in-sample errors.
- Opportunity-weighted rolling features, explicit shrinkage, baseline comparisons, and an EPA-free ablation.
- Deterministic coherent score distributions and meaningful tie/push/odds fixtures.
- Saved manifests and actual reports with sample sizes, uncertainty intervals, key-number diagnostics, and candid limitations.
- A locally usable CLI/dashboard and an offline synthetic path without making paid data a prerequisite.

I would preserve these choices. The problem is missing enforcement at boundaries, not a need to rebuild the project.

## Model limitations versus implementation defects

The tie/key-number problem is real: the holdout report puts about **2.8%** probability on ties versus approximately **0.37%** observed, and about **2.8%** on home winning by three versus **8.5%** observed. That matters for moneyline refunds and spread pushes.

However, the rounded bivariate-normal distribution was deliberately specified in the original plan. Claude Code implemented and exposed that limitation rather than secretly introducing it. I share responsibility for this modeling choice as the author of the V1 specification. Do not characterize it as a failure to follow instructions.

After correctness fixes, the most relevant modeling experiment is a distribution that better represents NFL scoring margins and ties, evaluated under a newly documented protocol. Keep V1 as a baseline. Do not repeatedly optimize this replacement against the already viewed 2025 holdout and retain the label “untouched holdout.”

Beating B0 is a useful achievement, but B0 is a simple league baseline. Without actual market comparisons, the project has not established sportsbook-level pricing or a betting edge. The absence of real market data is honestly disclosed and was allowed by the specification.

## Complexity assessment and continuation criteria

Requirement-to-complexity assessment: approximately **3/10**, a reviewer judgment rather than a measured score. The local modular architecture fits a single-maintainer research project. I found no reason to add infrastructure or rewrite it into a more elaborate system. Monthly maintenance debt cannot responsibly be estimated from this review alone.

The useful decision question is: can another reviewer reproduce a price and verify precisely which inputs and model produced it? Fix that before adding features.

For this research project, the appropriate stop/continue rules are:

- Withhold point-in-time-certified claims while R1/R4 remain unresolved; withhold market-comparison outputs as evidence while R2 remains unresolved.
- Preserve and label experiments whose code/data provenance cannot be reconstructed. Do not delete their evidence or silently overwrite results.
- Continue V1 once the listed regression cases and existing full suite pass. Luke remains the acceptance decision-maker; the planning assistant can review the resulting evidence.
- Require a concrete research question and evaluation plan before adding model families, data feeds, or dependencies. Keep a baseline that a new experiment must justify replacing.
- Do not add cloud services, live trading, or general plugin infrastructure to address these bugs. They are local correctness issues.

Production uptime, deployment staffing, and automatic shutdown machinery do not apply to the present local research scope.

## Recommended repair order and handoff to Claude Code

1. Finish and save the currently running final test results.
2. Add failing regression tests for R1 and R2, then fix them. These protect what the data and probability labels mean.
3. Add code/lockfile provenance enforcement and bundle compatibility checks (R3/R4), preserving the original holdout artifacts.
4. Correct forecast time semantics and away-spread CLV (R5/R6).
5. Correct the model-card/recap claims (R7), revise the acceptance-status table, and document which changes affect numerical research outputs.
6. Run the full suite including integration/demo and the new tests. Produce a follow-up review packet linking each finding to its fix and test.
7. Only after reviewing those changes, decide whether a separately labeled historical reproduction is necessary. Do not rerun or retune the existing holdout simply to clear an acceptance checkbox.

Claude Code should keep this a bounded V1 correction pass. Do not add injury feeds, weather, new model families, a key-number model, or a new interface during these repairs. Those are later planning decisions.
