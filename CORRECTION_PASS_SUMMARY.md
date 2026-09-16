# Correction pass summary for the planning assistant

Repository: `/Users/luke/origination-model`. Reviewed HEAD was `32b16ea`; the correction pass is
commit `38c2bea` (2026-09-16). Scope was bounded to the seven findings in
`NFL_V1_IMPLEMENTATION_REVIEW.md`. No new features, data feeds, model families or interfaces
were added. Architecture unchanged.

## Verification state

| Gate | Result |
|---|---|
| `ruff check .`, `ruff format --check .` | clean (78 files) |
| `mypy src/nfl_origination` | clean (41 files) |
| `pytest` with `NFL_ORIGINATION_BLOCK_NETWORK=1` | 132 passed (109 prior + 23 new regression tests in `tests/unit/test_review_regressions.py`), including integration, dashboard AppTest and end-to-end demo |
| Holdout gate check | `backtest --config configs/holdout.yaml --rerun-reason "gate check"` refused; protocol log still records exactly one holdout run |
| Frozen research artifacts | development, confirmation and holdout runs untouched and unregenerated |

## Finding by finding

**R1 (P1) recorded_asof did not protect every input.** Fixed.
- League priors are now computed from prior-season rows eligible at the game's cutoff (policy
  eligibility incl. observation time), cached per season only when every prior row is eligible.
  Prior game IDs (`prior_games_hash`), their max eligibility time and max observation time are in
  each feature row's provenance.
- Team-game rows carry `schedule_first_observed_utc`, `pbp_first_observed_utc`, and
  `first_observed_utc` = the later of the two (reviewer's early-schedule/late-PBP case now
  ineligible).
- Rest uses only schedule rows observed at or before the cutoff in recorded mode.
- A feature row whose own schedule row was not observed at the cutoff is flagged
  `unobserved_inputs`; such rows never train or forecast (`require_forecastable`, `usable_rows`).
- Labels in recorded mode become available at the later of `kickoff + 48h` and observation.
- `chronological_fits` and `residual_pool` drop labels unavailable at fit time (season's first
  cutoff, or wall-clock time for the forecast bundle fit), record the dropped count, and fail
  clearly when nothing remains.
- Tests: entirely late-observed history is rejected and mutation-invariant; a late-observed prior
  season drops out of the prior and provenance reflects it; late PBP + early schedule ineligible;
  late labels excluded from fitting and residuals (end-to-end feature/fit path); a
  `recorded_asof` backtest on data without observation times fails with `MissingDataError`.
- Not done (out of bounded scope): selecting historically eligible *cached versions* — only the
  current cache entry's first-observed time exists; `history.jsonl` records past entries but no
  version selection is implemented. `recorded_asof` is enforced, not exercised, on real data.

**R2 (P1) market comparison could use B0's distribution under the M1 label.** Fixed.
- `load_distribution(dists, game_id, model_id)` requires exactly one (game, model) match,
  validates the PMF, and errors on ambiguity or a missing `model_id` column. Comparison and paper
  backtest pass the prediction's own model; CLI prefers per-model distribution files.
- Tests: reordered rows give the same answer; comparison equals the selected model's direct
  pricing; ambiguous/missing model matches fail.

**R3 (P2) protocol recorded git metadata but did not verify code.** Fixed with a disclosed
limitation.
- Checksummed payload now includes `code_digest` (SHA-256 over `src/nfl_origination/**/*.py`
  excluding `dashboard/`, `cli.py`, `evaluation/report.py`) and `lock_digest` (`uv.lock`).
- Protocols are per `run.label` (`frozen_protocol_<label>.json`); the original keeps its file.
- A protocol lacking digests is refused outright for reruns. The original V1 protocol
  (`a31f98d0…`) is therefore unverifiable against the current implementation; it and the single
  holdout run are preserved unchanged, the dirty-tree limitation is stated in the model card,
  protocol doc and review packet, and any rerun must be a new labeled protocol.
- Tests: presentation-code edits leave the digest unchanged; a fitting-code edit or lockfile
  change is rejected; legacy protocol refused; new label gets its own file.

**R4 (P2) forecasting did not verify bundle/configuration compatibility.** Fixed.
- Bundles store `feature_contract` (data mode, lag, cutoff hours, features config, feature set,
  feature/schema versions) and `contract_hash`; `check_bundle_compatible` runs before any slate
  and `predict_game` rejects rows from a different data mode or with unobserved inputs. In
  `recorded_asof` the bundle must have been created before the decision time.
- The 2026 bundles were refit (`fit-20260916T182932Z-c541fc`) and carry contracts; this does not
  touch holdout metrics.
- Tests: changed history length/shrinkage rejected; data-mode change rejected; output-directory
  change allowed; contract-less bundle rejected.

**R5 (P2) future cutoffs labeled as standard-horizon forecasts.** Fixed.
- Three explicit horizons: live (default; information time = generation time; started games
  excluded; labeled `custom_horizon`), `--as-of` (must be in the past), and
  `--reconstruct-standard-horizon` (kickoff − 24h; refused until every cutoff has passed; allows
  completed games; labeled `kickoff_minus_24h` with a reconstruction note).
- Forecast rows record `model_created_utc`, `training_data_mode` and `horizon`; comparison and
  paper backtest exclude decisions dated before the model existed.
- The mislabeled week 2 slate was removed from `reports/final/`; replaced by
  `forecast_2026_week3_live` and `forecast_2026_week1_reconstruction`.
- Tests: live cutoff equals generation time and excludes started games; reconstruction refused
  before cutoffs pass and labeled correctly after; future `--as-of` rejected; pre-model quotes
  excluded.

**R6 (P2) away-spread line CLV sign.** Fixed: both spread sides use `bet − close`; totals
unchanged. Tests cover home/away favorites and underdogs and the full paper-backtest path
(away +3.5 closing +3 → +0.5).

**R7 (P2) overstated statistical support.** Fixed in `docs/model_card.md`, `README.md`,
`RECAP.md`, `docs/review_packet.md`. Wording now: holdout intervals exclude zero for score MAE,
margin MAE, 3-way log loss, 3-way Brier, margin CRPS and (narrowly, upper bound −0.018) total
CRPS; total MAE improves in point estimate (−0.355) with interval [−0.695, +0.018]; confirmation
total metrics also include zero; RMSE, binary log loss/Brier, coverage and tie rates carry no
intervals. EPA-free closeness is described as evidence of low EPA dependence, not resolution of
data-revision concerns.

## Acceptance status after the pass

- A03, A09, A10: reopened items resolved (with A03 noting recorded_asof is enforced, not
  exercised, on real data).
- A07, A08: met with the provenance caveat above (original freeze not code-verifiable).
- All others unchanged. Full table in `docs/review_packet.md`, which also has a
  finding → fix → test map.

## Impact on numerical research outputs

None for the committed development/confirmation/holdout artifacts: in
`historical_reconstruction`, every prior-season row is eligible at every in-season cutoff and
every label is available at each season's first cutoff, so the new availability filters select
exactly the rows those runs used. The 2026 forecast bundles and slates were regenerated.

## Open decisions for planning

1. Whether a separately labeled historical reproduction of the holdout is wanted now that
   code/lock digests exist. Not required to clear a checkbox; the original stays as evidence
   with its disclosed limitation.
2. Cached-version selection for `recorded_asof` (multiple snapshots per dataset/season) and
   when to start accumulating prospective snapshots.
3. The key-number/tie modeling experiment (next model family), under a new documented protocol
   and a new `run.label`, keeping V1 as the baseline.
4. Real timestamped odds remain absent; market comparison/ROI/CLV are mechanics-only on
   synthetic fixtures.

## Follow-up pass (F1, F2) — same day

**F1 (P2) stale feature caches bypassed the corrected builder.** Fixed.
- `FEATURE_VERSION` = "2", `SCHEMA_VERSION` = 2, so the cache key and bundle versions change.
- Every cache hit is validated against the current features schema, feature version and data
  mode; an invalid cache is moved aside (`features_<key>.stale-<ts>.parquet` plus an
  `.invalidated.json` reason) and features are rebuilt. Nothing historical is rewritten.
- `insufficient_warmup`, `unobserved_inputs` and the current feature version are mandatory at
  `usable_rows`, `require_forecastable`, `fit_score_model` and `predict_game`; missing fields
  raise `ModelValidationError` instead of being treated as permission to continue.
- Old bundles (schema 1) are refused by `ModelBundle.load`; the 2026 bundles were refit under
  version 2 (`fit-20260916T185415Z-b671ea`) and the live week 3 / week 1 reconstruction slates
  regenerated (`reports/final/`).
- Tests: seeded old-format cache under the same key is rebuilt through `prepare_dataset` (the
  normal loading path); a wrong-mode cache is rebuilt; missing safety fields and stale feature
  versions cannot pass the boundary.

**F2 (P2) missing PBP observation timestamps treated as known.** Fixed.
- Observation columns are normalized to tz-aware UTC first (naive or unparseable values raise a
  domain error). The combined team-game observation time is set only when both the schedule and
  PBP times are known; otherwise it is `NaT`, and `AsOfPolicy.eligible_mask` treats `NaT` as
  ineligible. A missing PBP observation column now means unknown, not known.
- Tests: all four known/unknown combinations; missing column; deliberate dtype errors; an
  aggregation-to-feature mutation test showing unknown-observation rows never become source
  games and cannot move an accepted recorded-as-of forecast.

Verification after the follow-up: ruff/format/mypy clean; 142 tests pass with network blocked.
Frozen research artifacts unchanged. Acceptance A03 updated in `docs/review_packet.md`; the
deferred cached-version selection remains disclosed and `recorded_asof` continues to fail
clearly when no eligible source version exists.
