# V2 model card — key-number challenger and prospective protocol

## Completion states (see V2_SPEC.md §1)

- **Engineering complete:** yes — contracts, commands, tests, demo and documentation implemented.
- **Operationally configured:** partially — receipts migrated, V2 bundles fit through 2025,
  epoch `epoch-0560a48399d2ec29` frozen 2026-09-17T16:14:16Z, one tick and verification run
  locally, football-only. No bookmaker, settlement rules or API key configured; no odds collected.
- **Prospective evidence:** pending — 0 real forecasts committed (the first tick found no due game;
  the Thursday 2026 Week 2 window had already passed and was recorded as missed). No settled games.
- **Model decision:** `challenger_ready_for_shadow_collection` from retrospective research; V1
  remains champion. Nothing here is prospective evidence.

## Champion (V1) and challenger

Champion: `M1_ridge_alpha100` (V1 Ridge score regression, rounded bivariate-normal PMF), refit
through 2025 under the dual-mode contract (training = retrospective reconstruction of completed
seasons observed 2026-09-17; forecasting = strict observed availability).

Challenger `KN_lambda0.1`: four-parameter exponential tilt of the champion's joint PMF with
features [tie, |M|=3, |M|=7, T even], theta fit by L-BFGS-B from zero with bounds ±3 on the
three latest prior out-of-fold seasons (2023–2025, 1,343 games for the 2026 bundle). Fitted
theta for the 2023 research fold at lambda 0.1: [−0.180, +0.583, +0.157, +0.047].

## Retrospective research (run `v2research-20260917T161131Z-e7a346`, real cached data)

Development seasons 2019–2023, 1,327 identical games; all candidates use the same reconstructed
V1 base distributions. Reused research data; 2024–2025 previously inspected.

| Candidate | Margin CRPS | Total CRPS | 3-way log loss | 80% margin cover | Pred tie rate | P(|M|=3) pred / obs | Key gap |
|---|---|---|---|---|---|---|---|
| B0 (reconstructed) | 8.0189 | 7.8937 | 0.7317 | 0.800 | 0.028 | 0.028 / 0.069 | 0.0475 |
| V1 (reconstructed) | 7.4016 | 7.6776 | 0.6802 | 0.806 | 0.028 | 0.028 / 0.069 | 0.0473 |
| identity (θ=0) | 7.4016 | 7.6776 | 0.6802 | 0.806 | 0.028 | 0.028 / 0.069 | 0.0473 |
| λ=0.01 | 7.4060 | 7.6777 | 0.6683 | 0.785 | 0.013 | 0.069 / 0.069 | 0.0055 |
| **λ=0.1 (selected)** | **7.3974** | 7.6776 | 0.6755 | 0.798 | 0.023 | 0.046 / 0.069 | 0.0292 |
| λ=1 | 7.4006 | 7.6776 | 0.6793 | 0.803 | 0.027 | 0.031 / 0.069 | 0.0446 |

Observed ties: 5 of 1,327 (0.38%). Paired block-bootstrap differences vs V1 (2,000 replicates):

| Candidate − V1 | Margin CRPS | Total CRPS | 3-way log loss |
|---|---|---|---|
| λ=0.01 | +0.0044 [−0.0175, +0.0283] | +0.0001 [−0.0005, +0.0006] | −0.0119 [−0.0147, −0.0088] |
| λ=0.1 | −0.0042 [−0.0133, +0.0056] | 0.0000 [−0.0002, +0.0003] | −0.0046 [−0.0055, −0.0037] |
| λ=1 | −0.0010 [−0.0022, +0.0002] | 0.0000 | −0.0009 [−0.0010, −0.0007] |

Screening for λ=0.1: margin CRPS strictly lower (yes, by 0.004 points; interval includes zero),
total CRPS and log loss within 1% (yes), key-frequency gap smaller (0.029 vs 0.047), no
numerical failures or candidate-only exclusions. It therefore meets the predeclared engineering
screen and is designated ready for **shadow** comparison. The margin-CRPS gain is small and not
statistically distinguishable from zero; the clear gains are in log loss and key-number/tie
frequencies. λ=0.01 fixes key numbers best but worsens margin CRPS and 80% coverage.

Retrospective checks (2024–2025, 544 games, not used for tuning): V1 margin CRPS 7.3038 vs λ=0.1
7.3035; log loss 0.6615 vs 0.6568; key gap 0.050 vs 0.034. No deterioration; V1 retained as
champion regardless.

## Limitations

- The tilt reshapes tie/key-margin/parity mass; it does not remove impossible scores or model
  overtime, and it can move expected scores (both regression and distribution means are stored).
- Training evidence is retrospective; only post-activation forecasts are prospective.
- No real market data: the market benchmark, CLV and paper ledger are exercised on synthetic
  fixtures only.
- The V1 2025 holdout artifacts came from an imperfectly reconstructible dirty tree; the newly
  passing tests do not retroactively validate them.
