# V2 implementation decisions

Routine implementation decisions made while implementing `V2_SPEC.md`, with the reason for each.
Statistical design (challenger family, lambda grid, bounds, thresholds, horizon) follows the spec
unchanged.

| Area | Decision | Why |
|---|---|---|
| Config | V2 settings live in a `v2:` section of the existing strict config (`V2Config`), not a parallel config system | keeps unknown-key strictness and one loader; V1 commands ignore `v2:` |
| Receipts | `data/raw/receipts/<dataset>/<season>/<receipt_id>.json`, blobs unchanged at their content-addressed paths; `entry.json`/`history.jsonl` preserved | additive, byte-for-byte preservation of V1 metadata |
| Receipt IDs | `obs|legacy|unknown|syn-<dataset>-<season>-<full-precision observed time>-<sha12>` | deterministic, idempotent migration; sub-second precision after a same-second collision surfaced in tests |
| Writer lock | `flock` on `receipts/.writer.lock`, re-entrant within a process via an RLock | nested writes (ingest inside a locked migration) deadlocked without re-entrancy |
| As-of selection | most recently observed usable receipt with `observed_at <= cutoff`; ties by receipt ID; `availability_unknown` never selectable; missing blob never selectable | A→B→A follows the observation sequence; unknown means ineligible |
| Baseline inventory | migration also writes an `observed` receipt at migration time for every blob that re-hashes correctly | establishes availability *from now*, never earlier (spec 5.2) |
| Cached-version selection scope | only receipts of the current cache layout are selectable; there is no reconstruction of pre-V2 snapshots | disclosed limitation; V1 metadata is `legacy_metadata` |
| Synthetic sources | synthetic demo data is stored as three all-season blobs (`synthetic_games`, `synthetic_results`, `synthetic_team_games`) with `synthetic` receipts | lets the demo exercise receipts → as-of selection without a synthetic PBP generator |
| Dual-mode bundle contract | `execution_contract` on bundles: `training_evidence_mode=retrospective_reconstruction`, `forecast_evidence_mode=observed_prospective`; compatibility check allows only the data-mode difference when declared | spec 5.3; V1 bundles without the contract are refused in the prospective flow |
| Bundle creation time | equals the injected fit time (system clock in real commands) | epochs require bundle creation ≤ activation; demos use a fixed clock |
| Epoch identity | `epoch-<sha16>` of the payload; activation time assigned by the system clock; active pointer file under `artifacts/v2/prospective/active_epoch.json` | immutable; switching the pointer is an explicit `freeze-prospective` |
| Epoch code digest | every package `.py` except `dashboard/` (V1's holdout digest excludes CLI and report rendering; V2 includes them) | spec 7.1 lists CLI orchestration and evaluation paths |
| Forecast uniqueness | `fc-<sha20>` of (protocol, game, horizon policy, bundle hash) | deterministic idempotency across tick reruns and restarts |
| Joint PMF storage | lossless `.npz` blob next to the JSON payload, hash inside the payload and manifest | spec asks for full joint PMF or a lossless reproducible artifact; JSON would be ~200 KB per forecast |
| Commit order | PMF blob → payload → manifest → marker; commitment time taken after payload/PMF are durable | an interrupted write leaves no marker and is ineligible |
| Eligibility | requires information cutoff and commitment both within ±window of the target and strictly before kickoff; stored on the manifest with the actual horizon | spec 7.2; displayed horizon is the real one |
| Missed window | one `missed_forecast_window` event per game, recorded on the first tick that observes the window closed before kickoff without a committed champion forecast | no retrospective replacement in the prospective sample |
| Market collection in a tick | at most one bounded provider call per tick, only when a game is due or within the closing window | request budget and quota reserve enforced by `CollectionBudget` |
| Odds quotes store | canonical quotes per receipt under `data/odds/quotes/<receipt>.parquet` after passing the V1 import validator | identical rejection reasons for live and CSV paths |
| Main vs alternate lines | `market_role` column; adapter marks provider `spreads/totals/h2h` as main; two complete main pairs at one snapshot are `ambiguous_main_line` | spec 6.2 |
| Historical CSV observation | `observed_at_utc` = import time unless the row carries its own collector time; `provenance_mode=historical_csv` | late imports cannot be prospective |
| Closing proxy validity | update non-missing and ≤ snapshot and within the age policy; local observation strictly before kickoff only when `require_local_observation` | spec 8.3; historical CSV comparisons stay analytical |
| Model availability | forecast rows carry `model_created_utc`; comparison/paper paths exclude decisions dated before it | spec 7.2/8.1 |
| Aligned market scoring | model conditional probability = `p_win/(1-p_push)` for the home/over side at the book line; refunds excluded and counted; intervals suppressed below 8 blocks | spec 8.2 |
| Key-frequency gap | mean of `|mean predicted − observed frequency|` over M=0, |M|=3 (both signs pooled), |M|=7 | aggregate frequency gap as specified; a per-game version was replaced after it proved uninformative |
| Challenger fit | class masses (≤16 feature classes) instead of full grids; L-BFGS-B, analytic gradient, zero start, bounds ±3, `ftol=1e-12`, `gtol=1e-8` | exact for a tilt that depends on (h,a) only through f; deterministic |
| Challenger support | adjusted omitted-mass bound `omitted·w_max/(w_min(1−omitted))`; base support expanded until ≤ 1e-8 or hard limit failure | spec 9.2 |
| Selection ties | within 1e-6 pooled margin CRPS prefer stronger regularization; identity wins ties | spec 9.3 |
| Demo challenger | the demo runs the bounded research on synthetic seasons and fits the selected candidate as a labeled shadow so both roles are exercised | spec 13 (end-to-end with both roles) |
| Dashboard | Prospective page added; V2 research runs render their own metrics page; evidence badge on every page; presentation strings exempt from the line-length lint | keeps existing pages, no redesign |
| Deferred | multi-version historical cached-source replay, second providers, scheduler installation | non-goals / deferred per spec |
