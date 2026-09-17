# V2 data contracts

Artifact schema/feature versions come from code (`schemas.SCHEMA_VERSION = 2`,
`FEATURE_VERSION = "2"`); the YAML `schema_version: 1` is the *config* contract and unrelated.

## Source observation receipts (`data/raw/receipts/`)

| Field | Rule |
|---|---|
| `receipt_id` | unique, immutable; encodes quality, dataset, season, observed time, sha prefix |
| `dataset`, `season` | `schedules`/`pbp` (season null only for the all-season schedule); synthetic datasets in demos |
| `content_sha256`, `blob_path`, `bytes` | verified payload identity; one blob per hash |
| `request_started_at_utc`, `observed_at_utc`, `persisted_at_utc` | collector clock at request start, after the complete response, after durable receipt creation |
| `provider_timestamp_utc` | optional; never a substitute for observation time |
| `source_url_redacted`, `adapter_version` | provenance without credentials (`apiKey=REDACTED`) |
| `synthetic`, `provenance_quality` | `observed` / `legacy_metadata` / `availability_unknown` / `synthetic` |

As-of selection: most recently observed usable receipt with `observed_at <= cutoff`, tiebreak by
receipt ID. `availability_unknown` receipts are never selected. Feature cache keys include the
selected receipt IDs and their observation times.

## Extended canonical quote

V1 columns plus `provider`, `receipt_id`, `raw_hash`, `observed_at_utc` (local observation),
`provenance_mode` (`historical_csv` / `live_collected` / `synthetic`), `market_role`
(`main` / `alternate`), `pair_id` (book, game, market, unordered line pair, snapshot, role).
Rules: spread lines opposite, total lines identical, moneyline lines null; decimal odds finite
and > 1; half-point grid; identical imports idempotent; conflicting duplicates rejected; late local
observation cannot be prospective; settlement rules must be configured per bookmaker/market.

## Provider receipts (`data/odds/receipts/`, blobs in `data/odds/blobs/`)

`receipt_id`, `provider`, `status` (ok, timeout, http_error, invalid_json, empty_payload,
missing_book, auth_failed, quota_exhausted, rate_limited, quota_reserve, budget_exhausted,
no_api_key), request/observed/persisted times, redacted non-secret parameters, HTTP status,
content hash, blob path, bytes, quota headers, error text (redacted). Event crosswalk rows in
`data/odds/event_crosswalk.jsonl` carry the match status (`matched`, `ambiguous_event_join`,
`unknown_team_name`, `kickoff_mismatch`, `no_schedule_match`) and observation time.

## Epoch (`artifacts/v2/prospective/epochs/<protocol_id>.json`)

Activation time (system), champion/challenger bundle refs (path, sha256, creation time, evidence
modes, training-source manifest hash, contract hash, calibration provenance), code digest (all
package code except the dashboard), lockfile digest, resolved config and hash, feature/schema
versions, seed, scope, horizon, market/paper policy, review thresholds, planned population,
payload hash. Immutable; a changed environment fails `verify-ledger`.

## Ledger (`artifacts/v2/prospective/<protocol_id>/`)

- `forecasts/<id>.json` payload (identity, times, inputs incl. receipts/hashes/feature rows,
  outputs incl. marginal PMFs and joint-PMF hash, provenance, quality flags, status) and
  `<id>.pmf.npz` (lossless joint PMF), `<id>.manifest.json` (hashes, commitment time, actual
  horizon, eligibility), `<id>.committed` marker.
- `decisions/<id>.json` (+ marker): every eligible market probability, EV, the single proposed
  selection or `no_bet` / `market_unavailable` with reason and exclusions.
- `outcomes/<game_id>.v<N>.json`: result versions (`final`, `pending`, `canceled`,
  `rescheduled_review_required`) with source receipt/hash and observation time; never merged
  into forecasts.
- `events.jsonl`: `tick`, `missed_forecast_window`, `forecast_failure`, `collection_failure`,
  `reschedule_observed`.
- `reports/prospective_<asof>.{json,md}` and `paper_ledger_<asof>.csv`.

Uniqueness key: `(protocol_id, game_id, horizon_policy_id, model_bundle_hash)`.
Eligibility: information cutoff and commitment within ±5 minutes of kickoff−24h and strictly
before kickoff. Hashes give local content consistency only.

## Challenger bundle (`family = key_number_adjusted`)

`model_params.base` (the wrapped V1 bundle verbatim), `theta` (4), `lambda`, `fit` (status,
iterations, gradient norm, objective, bound hits, pool seasons, game-ids hash), `calibration`
(nested provenance per calibration season: training seasons, residual seasons, residual ids
hash, counts; decision reference; fit time). Contract and execution contract copied from the
base.
