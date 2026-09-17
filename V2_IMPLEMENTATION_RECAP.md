# V2 implementation recap

Prepared 2026-09-17 against `V2_SPEC.md`. Repository `/Users/luke/origination-model`.
Baseline before V2 work: HEAD `c54f6da`, 148 tests passing, original V1 artifacts inventoried in
`docs/v2_baseline_inventory.json` (62 files/manifests re-verified unchanged after V2).

## Completion states

| State | Status |
|---|---|
| Engineering complete | **Yes.** Receipts + as-of selection, extended quote contract + mocked provider adapter, epochs + immutable ledger + one-shot tick + settlement + verification, aligned market benchmarks + CLV + paper accounting, key-number challenger with nested calibration and decision record, dashboard Prospective page, docs, offline demo, tests. |
| Operationally configured | **Partially.** Legacy cache migrated to receipts (36 receipts, 0 orphans); V2 bundles fit through 2025 under the dual-mode contract; epoch `epoch-0560a48399d2ec29` frozen at 2026-09-17T16:14:16Z (champion `M1_ridge_alpha100`, challenger `KN_lambda0.1`); one tick, `verify-ledger`, `report-prospective` and `settle-prospective` executed locally. **No bookmaker, settlement rules or `ODDS_API_KEY` configured; no odds collected.** |
| Prospective evidence | **Pending.** 0 real forecasts committed before kickoff, 0 settled. The first real tick (16:14Z on 2026-09-17) found no game inside its ±5-minute window; the 2026 Week 2 Thursday game's window (cutoff 2026-09-17T00:15Z) had already passed and is recorded as `missed_forecast_window`. |
| Model decision | `challenger_ready_for_shadow_collection` (retrospective research). **V1 remains champion.** |

## Plain answers

1. **What can Luke run now without credentials?** Everything in `docs/v2_operations.md` "offline":
   `demo-v2`, `snapshot-inventory`, `migrate-snapshots`, `research-v2`, `fit` (v2 config),
   `freeze-prospective`, `prospective-tick --offline`, `verify-ledger`, `report-prospective`,
   `settle-prospective` (cache only), the dashboard, and all V1 commands.
2. **What needs a configured bookmaker, supported rules and API key?** `collect-odds`, market
   collection inside `prospective-tick`, `settle-prospective --refresh` (network for new source
   observations), and any market benchmark/CLV/paper result on real data. Enabling the market
   without an explicit book and rules fails configuration validation.
3. **Is collection actually running?** No. Only the one-shot command exists and was run once by
   hand. No scheduler is installed; `docs/v2_operations.md` shows the cron line to add.
4. **How many REAL forecasts were committed before kickoff, and how many settled?** 0 and 0.
5. **Did the challenger improve the declared metrics, or did we retain V1?** On development
   seasons 2019–2023 (1,327 identical games) λ=0.1 passed the predeclared screen: margin CRPS
   7.3974 vs 7.4016 (difference −0.0042, 95% block-bootstrap interval [−0.0133, +0.0056]), total
   CRPS unchanged within 1%, 3-way log loss 0.6755 vs 0.6802 (interval below zero), key-frequency
   gap 0.029 vs 0.047. It is designated ready for shadow collection only; V1 is retained as
   champion. The margin-CRPS gain is not distinguishable from zero.
6. **Which results are retrospective, synthetic, prospective, unavailable or inconclusive?**
   Research metrics: retrospective (reused data; 2024–2025 previously inspected). Demo lifecycle,
   paper ledger and market benchmark numbers: synthetic. Real prospective forecasts, market
   comparison, CLV and ROI: unavailable/pending. Challenger margin-CRPS improvement: inconclusive.
7. **What exact evidence is still needed before saying the model competes with sportsbook
   prices?** A configured reference book with documented settlement rules and a live key; ticks
   running every five minutes through the season; ≥100 matched settled games across ≥8
   season-week blocks under one epoch; aligned model-vs-market log loss/Brier intervals below
   zero on those games; CLV reported with counts per market; integrity verification passing.
   None of that exists yet.

## Changed modules

| Module | Change |
|---|---|
| `config.py` | `V2Config` section (storage, evidence, models, challenger, horizon, sources, market, collection, closing proxy, paper, evaluation) with market validation; run kinds `v2_research`, `v2_prospective`, `v2_demo` |
| `data/snapshots.py` (new) | receipts, re-entrant writer lock, as-of selection, `manifest_as_of`, inventory, migration |
| `data/download.py` | records observed receipts (request/observed/persisted times); `SourceEntry.receipt_id`/`provenance_quality` |
| `experiment.py` | `prepare_dataset(source_cutoff=…)`, receipt IDs in cache keys, pinned synthetic manifests, injectable fit time/manifest/out dir/contract in `fit_forecast_bundles` |
| `models/bundle.py` | `execution_contract` dual-mode compatibility (declared only) |
| `market/import_csv.py`, `market/asof.py` | extended quote columns, `pair_id`, main/alternate pairing, ambiguity, late local observation, closing-proxy validity/freshness |
| `market/providers.py` (new) | The Odds API v4 adapter, budget/quota reserve, receipts, statuses, event crosswalk |
| `market/benchmark.py` (new) | de-vig, aligned scoring, CLV, paper settlement/summary |
| `prospective/` (new) | `clock.py`, `protocol.py` (epochs), `ledger.py`, `runner.py` (tick/settle/verify/fit), `demo.py` |
| `models/key_number.py` (new) | tilt, class-mass fitting, tail bound, challenger bundle, prediction |
| `evaluation/v2.py` (new) | prospective report; nested calibration research and selection |
| `cli.py` | 10 new commands; `fit` routes v2 configs |
| `dashboard/app.py` | V2 roots, evidence badges, labels, Prospective and research pages |
| configs | `v2_research.yaml`, `v2_prospective.yaml`, `v2_demo.yaml` |
| tests | `test_snapshots.py` (9), `test_providers.py` (13), `test_prospective_ledger.py` (5), `test_key_number.py` (8), `test_v2_pipeline.py` (5), `test_v2_dashboard.py` (2) |
| docs | `v2_decisions.md`, `v2_data_contracts.md`, `v2_operations.md`, `v2_model_card.md`, `dashboard_guide.md`, `v2_baseline_inventory.json`, README section |

## Requirement → test mapping

| Requirement | Test(s) |
|---|---|
| A→B→A observation sequence; identical repeat fetch; unknown availability ineligible; missing blob | `test_snapshots.py::test_asof_selection_follows_observation_sequence`, `::test_unknown_availability_never_selected`, `::test_missing_blob_is_not_selectable` |
| Failed download creates no receipt; blob stored once | `test_snapshots.py::test_ingest_writes_observed_receipts_and_failed_downloads_do_not` |
| Legacy migration additive/idempotent, no earlier availability, bytes unchanged | `test_snapshots.py::test_legacy_migration_is_additive_and_idempotent` |
| Writer lock | `test_snapshots.py::test_writer_lock_serializes_concurrent_writers` |
| Secret redaction; structured statuses; timeout; missing book; no-key no-fallback; budget/reserve | `test_providers.py::test_ok_response_persists_receipt_without_secret`, `::test_structured_failure_statuses`, `::test_timeout_and_missing_book`, `::test_no_api_key_has_no_synthetic_fallback`, `::test_request_budget_and_quota_reserve` |
| Event ambiguity quarantine; missing settlement rule | `test_providers.py::test_event_mapping_quarantines_ambiguity_and_rules` |
| Alternates, ambiguous main line, late local observation, historical CSV never prospective | `test_providers.py::test_pairing_rules_alternates_ambiguity_and_local_observation` |
| Idempotent identical import; conflicting duplicates | `test_providers.py::test_identical_import_is_idempotent_and_conflicts_rejected` |
| Two concurrent writers cannot commit two forecasts; conflicting duplicate; timing ineligibility; PMF hash | `test_prospective_ledger.py` |
| Two synthetic games committed, one missed, one missing book; determinism; idempotent tick; tampering/missing blob; partial record ineligible; outcome versions never modify forecasts; epoch drift | `test_v2_pipeline.py` |
| θ=0 reproduces V1 within 1e-8; coherence; swap transposes; tie/shared-|3| mechanics; gradient check; deterministic bounded fit; insufficient history; tail bound; zero-mass rejection | `test_key_number.py` |
| Dashboard renders legacy, research, prospective (pending/settled/missing market) with evidence labels; empty state | `test_v2_dashboard.py` |
| V1 invariants retained | all prior tests (148) unchanged and passing |

## Commands run and results

| Command | Result |
|---|---|
| `uv sync --frozen` | ok (lockfile unchanged) |
| `uv run ruff check .` / `ruff format --check .` | clean (98 files) |
| `uv run mypy src/nfl_origination` | clean (52 files) |
| `NFL_ORIGINATION_BLOCK_NETWORK=1 uv run pytest -q -p no:cacheprovider` | 190 passed, 0 failed, exit 0 (148 prior + 42 new), about 7 minutes wall clock including two synthetic V2 lifecycles and the dashboard AppTest |
| `demo-v2 --config configs/v2_demo.yaml --offline` | ok, ~31 s; epoch `epoch-69558ff3e12f4332`, 16 forecasts (8 games × champion/challenger), 14 bet / 2 market_unavailable decisions, 1 missed window, integrity ok |
| `migrate-snapshots` | 36 receipts created (18 legacy_metadata, 18 observed baseline), 18 duplicates skipped, 0 orphans; report `artifacts/v2/snapshots/migration_2026-09-17T145055.668085Z.json` |
| `research-v2 --config configs/v2_research.yaml --offline` | run `v2research-20260917T161131Z-e7a346`, ~90 s, 0 exclusions, decision `challenger_ready_for_shadow_collection` (λ=0.1) |
| `fit --through-season 2025 --config configs/v2_prospective.yaml --offline` | bundles under `artifacts/v2/models/forecast_2026/` incl. `KN_lambda0.1.json` (calibration seasons 2023–2025, 1,343 games) |
| `freeze-prospective --label v2_2026` | `epoch-0560a48399d2ec29`, activated 2026-09-17T16:14:16Z, code digest `a8e08c5e…` |
| `prospective-tick --offline` | 4,447 in-scope games, 0 due, 1 missed recorded (`2026_02_DET_BUF`), 0 forecasts |
| `verify-ledger` | status ok, epoch ok |
| `report-prospective` | coverage: 256 scheduled in-scope games after activation, 0 committed, review gate `pending_evidence` |
| `settle-prospective` | no new result versions (cache only) |
| all new commands `--help` | ok |

## Deviations from the spec

- Cached-version replay is limited to receipts created by V2 (legacy metadata imported as
  `legacy_metadata`; no reconstruction of pre-V2 snapshots). Disclosed in `docs/v2_decisions.md`.
- The V1 holdout code digest (excludes CLI/report) and the V2 epoch digest (all code except the
  dashboard) are different scopes, by design of the two specs.
- The demo's challenger is fit from a synthetic research decision so both roles are exercised;
  when research retains V1 the demo labels the challenger as a synthetic shadow.
- Epoch scope seasons are taken from `data.seasons` (2010–2026), so the in-scope population is
  every REG game in the cache; the coverage denominator counts games kicking off after activation.

## Known limitations

- No real odds; market benchmark, CLV and paper results exist only on synthetic fixtures.
- 0 prospective forecasts; the first eligible windows are the 2026 Week 2 Sunday games
  (cutoff 2026-09-19T17:00Z) and the tick must be running then.
- The challenger's margin-CRPS gain is within noise; its clear effects are on tie/key-margin
  frequencies and log loss.
- The original V1 2025 holdout provenance limitation remains.
- Prospective ledger hashes are local consistency checks, not independent certification.

## Next actions

1. Configure a reference bookmaker with documented settlement rules and export `ODDS_API_KEY`.
2. Add the five-minute cron line for `prospective-tick`; run `settle-prospective --refresh`
   after each week; `verify-ledger` and `report-prospective` weekly.
3. Do not change code/config/bundles under the active epoch; any change → new labeled epoch.
4. Review after ≥100 matched settled games and 8 blocks; produce the evidence packet.
