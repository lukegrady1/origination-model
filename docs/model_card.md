# Model card — NFL pregame origination model V1

**Intended use.** Research-only estimation of NFL regular-season game outcomes and derived fair
moneylines, spreads and totals at a cutoff of kickoff minus 24 hours, for reviewers evaluating
football feature engineering, chronological validation and coherent pricing. Not for real-money
wagering, automated execution, or production trading.

**Models.** B0: pooled training-mean team score plus an estimated home/away offset. M1: one
shared Ridge regression (alpha 100, selected on development seasons) predicting points for each
team-perspective row from recency-weighted, opportunity-weighted, shrunk EPA/success/explosive
rates, play volume, points, venue, rest and history counts. Both use a homoscedastic bivariate
normal residual model (mean plus 0.9·S + 0.1·diag(S) covariance from the five most recent
out-of-fold seasons), rounded to integer scores and floored at zero. Ablation: M1 without EPA and
success features (alpha 100).

**Data.** nflverse schedules and play-by-play 2010–2025 (2010–2011 warm-up, training from 2012).
Historical reconstruction mode with a 48-hour eligibility convention. No injuries, starters,
weather, depth charts, or market inputs.

## Results (real data, frozen protocol)

Holdout 2025, 272 games, run `holdout-20260916T175029Z-4385c4`, run once after freezing:

| Metric | B0 | M1 (alpha 100) | M1 EPA-free | M1 − B0 (95% block bootstrap) |
|---|---|---|---|---|
| Score MAE (team-game) | 7.877 | **7.439** | 7.464 | −0.438 [−0.666, −0.218] |
| Score RMSE | 9.880 | 9.344 | 9.351 | |
| Margin MAE | 11.064 | **10.149** | 10.191 | −0.915 [−1.367, −0.474] |
| Total MAE | 10.971 | **10.616** | 10.650 | −0.355 [−0.695, +0.018] |
| 3-way log loss | 0.7295 | **0.6706** | 0.6704 | −0.059 [−0.086, −0.032] |
| 3-way Brier (sum over classes) | 0.502 | 0.448 | 0.447 | −0.054 [−0.079, −0.030] |
| Binary log loss, non-tied (271 games) | 0.6906 | 0.6314 | 0.6310 | |
| CRPS margin | 7.987 | **7.287** | 7.308 | −0.700 [−1.001, −0.390] |
| CRPS total | 7.797 | 7.589 | 7.588 | −0.207 [−0.387, −0.018] |
| Margin coverage 50 / 80 / 95 % | 0.566 / 0.776 / 0.949 | 0.563 / 0.801 / 0.956 | 0.548 / 0.813 / 0.952 | |
| Margin 80 % mean width (points) | 36.0 | 33.5 | 33.7 | |
| Predicted vs observed tie rate | 0.028 vs 0.0037 (1 tie) | 0.028 vs 0.0037 | 0.028 vs 0.0037 | |

Confirmation 2024 (272 games): M1 score MAE 7.352 vs B0 7.828 (−0.476 [−0.736, −0.235]);
3-way log loss 0.652 vs 0.721. Development 2018–2023 (1,583 games): M1 7.566 vs B0 8.041
(−0.475 [−0.589, −0.365]); 3-way log loss 0.677 vs 0.730.

Slices (holdout, M1): weeks 1–4 score MAE 7.16 (64 games); neutral site 6.64 (7 games, too few
to interpret); no cold-start games in 2025.

Reliability (holdout, 3-way home-win probability, 0.1 bins): bins 0.2–0.7 are close to the
diagonal (n = 16–72 each); bins above 0.7 (n = 24 and 11) and below 0.2 (n = 1) are sparse.

Key numbers (holdout, M1): predicted P(margin = 3) 0.028 vs observed 0.085; P(margin = −3) 0.027
vs 0.066; P(margin = ±7) 0.026/0.024 vs 0.048/0.048; P(tie) 0.028 vs 0.004; P(total even) 0.500
vs 0.445. The rounded normal badly under-weights 3 and 7 and over-weights ties. This is the
distribution's known structural weakness, not a calibration bug.

**Recommendation.** M1 (alpha 100) is preferred over B0 on every headline error and probability
metric with bootstrap intervals excluding zero on the holdout, confirmation and development
seasons. The EPA-free ablation is within noise of M1 (holdout score MAE 7.464 vs 7.439), so the
improvement over B0 does not depend on possibly-revised EPA fields. Further modeling is required
before any real pregame use: a key-number-aware score distribution, personnel/injury and weather
inputs, and a certified point-in-time data path.

## Limitations and caveats

- Historical reconstruction is not a point-in-time-certified backtest; the 48-hour lag is a
  convention and upstream EPA models may have been trained later.
- The 2025 season is a retrospective project holdout. It is not proof that the researcher had no
  prior knowledge of its outcomes.
- The score distribution permits implausible scores, smooths key numbers and predicts ties about
  seven times too often.
- No market comparison, ROI, or CLV was run on real odds: no timestamped odds were supplied.
  Synthetic odds exercise the mechanics only.
- Standardized coefficients are diagnostics, not causal explanations. In the 2026 bundle the
  largest are `team_points_for` (+1.31), `opp_points_against` (+1.12), `venue_advantage` (+0.99)
  and `team_off_dropback_epa` (+0.72) points per standard deviation.

**Provenance.** Protocol checksum `a31f98d0…`, config hash and source SHA-256 hashes are in
`reports/final/holdout/manifest.json`. Tested on Apple M5 (10 cores, 24 GB), macOS 26.6,
Python 3.12.14: ingest of 17 seasons 86 s; features 7 s; development backtest 72 s;
confirmation/holdout 5 s each; offline demo 37 s; full test suite ≈ 3 min.
