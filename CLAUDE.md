# CLAUDE.md — working agreement for this repository

The planning authority is `V1_SPEC.md`. It owns scope and statistical design; this file only
summarizes invariants so they are not violated by routine edits. Do not copy or restate the
spec's formulas here; link to the spec section instead.

## Invariants (see V1_SPEC.md sections 4, 6, 8–11)

- `models/` and `features/` never import `market/`. Market-derived columns are quarantined
  (`schemas.assert_no_market_columns`) and can never be model inputs.
- Training inputs are exactly the feature allowlist in `schemas.FEATURE_SETS`. Adding a feature
  is a planning decision, not an implementation detail.
- Every feature row depends only on source games with `kickoff_utc + 48h <= cutoff_utc`
  (plus `first_observed <= cutoff` in `recorded_asof`). Labels are a separate table.
- Chronological only: season Y models are fit on 2012…Y−1 and frozen for the season; residual
  distributions come from prior out-of-fold seasons; scalers/imputers fit inside the fold.
- The joint score PMF is deterministic quadrature over a rounded, zero-floored bivariate normal.
  Never replace a failed PMF with a 50/50 forecast and never use Monte Carlo for final prices.
- Lines are handled in integer half-point units; quarter lines are rejected.
- The holdout (2025) runs only against a frozen protocol (`artifacts/protocol/`). Reruns need
  `--rerun-reason` and are labeled. Do not tune against 2025.
- Report poor results. There is no success gate requiring M1 to beat B0 and no ROI claim.

## Commands

```bash
uv sync --frozen
uv run nfl-origination demo --config configs/demo.yaml --offline
uv run ruff check . && uv run ruff format --check . && uv run mypy src/nfl_origination && uv run pytest -q
```

See `docs/implementation_status.md` for what has been run and what is still open, and
`docs/review_packet.md` for the review entry point.
