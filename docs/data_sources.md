# Data sources and attribution

## Required free sources (used)

| Source | Asset | Use |
|---|---|---|
| nflverse-data releases, `schedules/games.csv` (maintained by Lee Sharpe) | one file, all seasons | games, results, kickoff times, neutral sites; betting columns quarantined as `untimed_reference` |
| nflverse-data releases, `pbp/play_by_play_<season>.parquet` (nflfastR) | one file per season 2010–2026 | EPA, yards, dropback/rush flags, kneel/spike/no-play/2pt flags, quarter |

URLs are resolved in `data/download.py` and recorded in every cache entry and run manifest.
Attribution: data from nflverse (https://github.com/nflverse/nflverse-data). Review the
upstream license and each dataset's redistribution terms before bundling raw files; this
repository does not commit raw data.

## Field mapping verified on downloaded files (2010–2026)

| Canonical field | nflverse column | Verified semantics |
|---|---|---|
| dropback indicator | `pass` | 1 for pass attempts, sacks and scrambles (scrambles have `play_type=run`, `pass=1`, `rush=0`) |
| designed rush | `rush` | 1 for designed rushes only; never 1 together with `pass` on real plays |
| kneel / spike / two-point | `qb_kneel`, `qb_spike`, `two_point_attempt` | excluded from efficiency plays |
| no play | `play_type == "no_play"` | penalties and reviews; excluded |
| quarter | `qtr` | 5 = overtime, excluded from efficiency, included in final scores |
| EPA / yards | `epa`, `yards_gained` | finite on every real scrimmage play except one 2010 and one 2019 play (skipped from EPA denominators) |
| kickoff | `gameday` + `gametime` (Eastern) | converted with `America/New_York`, DST-aware; no null times in 2010–2025 REG |
| neutral site | `location == "Neutral"` | 68 REG games 2010–2026 |
| franchise aliases | `STL→LA`, `SD→LAC`, `OAK→LV` | explicit mapping; originals preserved |

Placeholder rows such as `*** play under review ***` carry `pass=0, rush=0` and are ignored;
the semantic invariant checks (sack ⇒ dropback, scramble ⇒ dropback and not rush) are applied
to real scrimmage plays only and pass on every season.

## Coverage audit (real source, `validate-data --seasons 2010:2025`)

| seasons | scheduled | final | with PBP | note |
|---|---|---|---|---|
| 2010–2020 | 256 each | 256 | 256 | complete |
| 2021, 2023, 2024, 2025 | 272 each | 272 | 272 | complete |
| 2022 | 271 | 271 | 271 | the canceled Week 17 BUF–CIN game is absent from the source; reported in `coverage_report.json` notes and every backtest report |

Total plays 2010–2025 after normalization: 737,117 regular-season plays; 4,175 completed games.

## Known upstream limitations

- The nflverse availability page reports the injury feed ended after 2024 and depth-chart
  timestamps changed; neither feed is used.
- Historical files are current revisions. EPA values may have been recomputed with models
  trained later; the EPA-free ablation measures dependence on those fields but does not certify
  point-in-time availability.
- Schedule betting fields (`spread_line`, `total_line`, moneylines) have no quote timestamps or
  bookmaker identity. They are stored as `untimed_reference` and are never used for features,
  calibration, selection, or ROI.

## Optional market source (not purchased)

The Odds API historical snapshots (paid) could be exported to the canonical CSV in
`V1_SPEC.md` section 12 and imported with `import-odds`. No subscription was used; all market
tests run on `tests/fixtures/synthetic/odds_synthetic.csv`, which is labeled synthetic in every
artifact, report and dashboard panel.
