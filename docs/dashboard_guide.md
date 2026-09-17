# Dashboard guide

`uv run streamlit run src/nfl_origination/dashboard/app.py`. Read-only: refreshing never
ingests, fits or collects. Roots: `artifacts/`, `artifacts/demo/`, `artifacts/v2/research/`,
`artifacts/v2/demo/research/` for runs; `artifacts/v2/`, `artifacts/v2/demo/` for epochs
(override with `NFL_ORIGINATION_ARTIFACTS` / `NFL_ORIGINATION_PROSPECTIVE`).

Every page shows an evidence badge: **SYNTHETIC**, **RETROSPECTIVE RECONSTRUCTION**, **LEGACY
OBSERVATION METADATA** or **PROSPECTIVE LOCAL RECORDING**. There is no "verified" badge.

| Page | What it measures | What it cannot prove |
|---|---|---|
| Slate | Fair spread/total/moneylines, tie probability and warnings per game from a saved run | Nothing about market edge; "unavailable" values are not filled |
| Game detail | Base vs candidate distributions, key-margin and tie probabilities, intervals, feature provenance, matched quotes with timestamps | A model-market gap is a price-reference difference, not mispricing |
| Evaluation | V1/B0/challenger metrics on identical games (research runs show the candidate table, decision, retrospective checks and paired intervals) | Retrospective metrics are not prospective evidence; 2024–2025 were inspected |
| Prospective | Epoch, roles, cutoff/commitment/actual horizon, source freshness, eligible/missed/pending/settled counts, collection failures, frozen paper signals, market benchmark, CLV with counts, review gate | Local hashes are not external notarization; small samples show no intervals; synthetic epochs are never merged with real ones |
| Run audit | IDs, hashes, config, integrity status, downloads | Provenance of the original V1 holdout tree |

"No odds" (market_unavailable) is distinct from "no model prediction" (missing forecast).
"Paper signal under frozen policy" is never a recommendation.
