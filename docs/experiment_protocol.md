# Experiment protocol (frozen)

Protocol checksum: `a31f98d0cee0830b01a27def818c08281f750fc7b07132b1bf36d7f273408f5b`
(`artifacts/protocol/frozen_protocol.json`, copied to `reports/final/frozen_protocol.json`).

## Splits

| Role | Seasons |
|---|---|
| Feature warm-up | 2010–2011 |
| Initial model history | 2012–2017 |
| Chronological development folds | 2018, 2019, 2020, 2021, 2022, 2023 |
| Confirmation | 2024 |
| Locked retrospective holdout | 2025 (run once) |
| Forecast bundle | fit on 2012–2025 for the 2026 season |

Only completed regular-season games with valid final scores are scored. Each game's cutoff is
`kickoff_utc − 24h`; a source game is eligible when `kickoff_utc + 48h ≤ cutoff` (data mode
`historical_reconstruction`).

## Steps as executed

1. `validate-data --seasons 2010:2025`: coverage audited (one 2022 game absent from source).
2. Features built at each game's own cutoff (8,350 perspective rows; 2010 rows flagged
   `insufficient_warmup` and never used for training).
3. Development backtest `development-20260916T173324Z-b71c52`: B0, M1 with alpha in
   {1, 10, 100, 1000}, and the EPA-free ablation with the same grid, each with expanding annual
   fits and residual pools from the five most recent out-of-fold seasons (≥ 512 games).
4. Alpha selection on pooled development score MAE (ties within 0.01 → larger alpha):
   full feature set → **alpha = 100** (MAE 7.5660 vs 7.5657 at alpha 10 and 7.5664 at alpha 1;
   alpha 1000 at 7.5788 is outside the tie band); EPA-free → **alpha = 100**.
   Decision record: `reports/final/development/decision_record.json`.
5. Confirmation 2024 `confirmation-20260916T173933Z-48f95d`: no methodological change made
   after seeing 2024.
6. `freeze-protocol` at 2026-09-16T17:50:27Z, then the holdout `holdout-20260916T175029Z-4385c4`
   ran exactly once. The protocol log has one holdout entry.
7. `fit --through-season 2025` produced the separate 2026 bundles under
   `artifacts/models/forecast_2026/`; their predictions are not mixed into holdout metrics.

## Post-freeze code changes (logged, no holdout rerun)

Immediately after the freeze:

- `experiment.predict_slate`: forecast outputs are now written per model
  (`predictions_<model_id>.parquet`) because the slate schema key is (run, game, cutoff); the
  first forecast attempt failed validation and its empty run directory was removed.
- Run directories are removed when a run fails before writing a manifest.

Correctness pass after the 2026-09-16 implementation review (findings R1–R7):

- R1: league priors, rest metadata, labels and residual pools are now filtered by the as-of
  policy; team-game rows carry both schedule and PBP observation times; feature rows carry
  `unobserved_inputs` and `prior_games_hash`; fits drop labels unavailable at fit time.
  In `historical_reconstruction` every prior-season row is eligible at every in-season cutoff
  and every label is available at each season's first cutoff, so these filters select exactly
  the rows the frozen runs used. The saved holdout artifacts were not regenerated.
- R2: saved distributions are looked up by (game, model) and validated.
- R3: the frozen protocol now includes a scoped source-code digest and the `uv.lock` digest.
  The original V1 protocol (`frozen_protocol.json`) predates this and is therefore
  **unverifiable against the current implementation**; the verifier refuses any holdout run
  against it. It is preserved unchanged with its single recorded holdout run. Any future rerun
  must freeze a new protocol under a new `run.label` and is a separately labeled experiment.
- R4: bundles carry a feature/policy contract that is checked before forecasting; the 2026
  bundles were refit (`fit-20260916T182932Z-c541fc`), which does not touch holdout metrics.
- R5: forecasts have three explicit horizons (live = generation time, `--as-of`, and
  `--reconstruct-standard-horizon`); the earlier "week 2 standard horizon" slate that used
  future cutoffs was removed from `reports/final/` and replaced by a live week 3 slate and a
  week 1 reconstruction.
- R6: away-spread line CLV sign corrected (synthetic-demo mechanics only; no real odds).
- R7: model-card and recap wording now states exactly which intervals exclude zero.

No backtest numerics changed for the frozen data mode; the development, confirmation and
holdout artifacts are the originals.

## Measures reported

Score/margin/total MAE and RMSE; 3-way log loss and Brier (sum over classes, averaged over
games); binary conditional log loss/Brier on non-tied games with ties counted separately;
reliability in fixed 0.1 bins with sparse (<30) flags; predicted vs observed tie rate; margin and
total CRPS from the integer PMFs; 50/80/95% interval coverage and widths; paired differences vs
B0 with 1,000 season-week block-bootstrap resamples (seed 42); season, weeks 1–4, neutral-site and
cold-start slices; key-number diagnostics at margins 0, ±3, ±7 and total parity; EPA-free
ablation under the identical protocol.
