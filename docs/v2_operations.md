# V2 operations guide

## What runs without credentials (offline)

```bash
uv sync --frozen
NFL_ORIGINATION_BLOCK_NETWORK=1 uv run nfl-origination demo-v2 --config configs/v2_demo.yaml --offline
uv run nfl-origination snapshot-inventory --config configs/v2_prospective.yaml
uv run nfl-origination migrate-snapshots --config configs/v2_prospective.yaml
uv run nfl-origination research-v2 --config configs/v2_research.yaml --offline      # cached 2010–2025 data
uv run nfl-origination fit --through-season 2025 --config configs/v2_prospective.yaml --offline
uv run nfl-origination freeze-prospective --config configs/v2_prospective.yaml --label v2_2026
uv run nfl-origination prospective-tick --config configs/v2_prospective.yaml --offline
uv run nfl-origination verify-ledger --config configs/v2_prospective.yaml
uv run nfl-origination report-prospective --config configs/v2_prospective.yaml
uv run nfl-origination settle-prospective --config configs/v2_prospective.yaml       # cache only
uv run streamlit run src/nfl_origination/dashboard/app.py
```

Football-only collection (no odds) is the default configuration and needs no key.

## What needs a configured bookmaker, rules and an API key

Set `v2.market.enabled: true`, `v2.market.bookmaker` (reference book), `v2.market.bookmakers`
and `v2.market.settlement_rules[<book>][moneyline|spread|total]` in
`configs/v2_prospective.yaml`; export `ODDS_API_KEY`. Enabling the market without a book or
rules fails config validation. Then:

```bash
export ODDS_API_KEY=...            # never written to receipts, manifests, errors or logs
uv run nfl-origination collect-odds --config configs/v2_prospective.yaml
uv run nfl-origination prospective-tick --config configs/v2_prospective.yaml       # collects once if a game is due or closing
uv run nfl-origination settle-prospective --config configs/v2_prospective.yaml --refresh   # fetches new schedule/PBP observations
```

Each invocation is bounded by `v2.collection` (timeout, ≤3 attempts per request, request budget
per invocation, quota reserve from the last `x-requests-remaining` header).

## Scheduling

Nothing is installed automatically. To collect prospectively, an external scheduler must call
the tick every five minutes, e.g. a cron entry:

```
*/5 * * * * cd /path/to/origination-model && ODDS_API_KEY=... uv run nfl-origination prospective-tick --config configs/v2_prospective.yaml >> logs/tick.log 2>&1
```

A tick is idempotent: rerunning returns existing committed records and never duplicates paper
decisions. A missed window is recorded as `missed_forecast_window` and is never replaced in the
prospective sample. Any change to code, `uv.lock`, configuration or bundles makes the epoch fail
verification; freeze a new epoch (new `--label`) and explain the change.

## Timing semantics

Target window: kickoff − 24h ± 5 min. The information cutoff is the real time after input
collection, frozen at forecast start; commitment time is recorded after durable publication;
both must be inside the window and strictly before kickoff for the record to be eligible.
`--as-of` cannot bypass the real clock (real commands accept no creation times).

## Settlement and reporting

`settle-prospective` appends result versions from the newest observed schedule; with
`--refresh` it first fetches new observations and reports which sources changed. A changed
kickoff after commitment becomes `rescheduled_review_required` and is excluded from automatic
paper settlement. `report-prospective` combines forecasts, decisions and outcome versions as of
now, with coverage denominators, all-game vs matched-game metrics, CLV counts per market and the
planned review gate (≥100 matched settled games and 8 season-week blocks; a review threshold, not
a power claim).
