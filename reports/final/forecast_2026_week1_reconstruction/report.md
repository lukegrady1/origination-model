# NFL Origination Model — forecast run `forecast-20260916T185418Z-1704bd`

Created 2026-09-16T18:54:18.584647Z · data mode `historical_reconstruction` · forecast policy `kickoff_minus_24h` · git `38c2bea2cbbb88b292bb3c1e10f945a0fc96bee9` (dirty) · config hash `6a0abc279a98`

Forecast slate `2026_week1` · model `M1_ridge_alpha100` · policy `kickoff_minus_24h`

## Slate

| game | kickoff (UTC) | cutoff (UTC) | fair home handicap | fair total | home ML | away ML | P(tie) | flags |
|---|---|---|---|---|---|---|---|---|
| NE @ SEA | 2026-09-10 00:20 | 2026-09-09 00:20 | 0.0000 | 48.0000 | 100.5648 | -100.5648 | 0.0304 | rest_missing |
| SF @ LA | 2026-09-11 00:35 | 2026-09-10 00:35 | -2.5000 | 52.5000 | -137.4890 | 137.4890 | 0.0299 | rest_missing |
| ATL @ PIT | 2026-09-13 17:00 | 2026-09-12 17:00 | -3.0000 | 45.0000 | -144.3153 | 144.3153 | 0.0298 | rest_missing |
| BAL @ IND | 2026-09-13 17:00 | 2026-09-12 17:00 | -1.0000 | 47.5000 | -114.0462 | 114.0462 | 0.0304 | rest_missing |
| BUF @ HOU | 2026-09-13 17:00 | 2026-09-12 17:00 | 0.0000 | 46.5000 | -102.2808 | 102.2808 | 0.0305 | rest_missing |
| CHI @ CAR | 2026-09-13 17:00 | 2026-09-12 17:00 | 1.5000 | 45.5000 | 117.3895 | -117.3895 | 0.0303 | rest_missing |
| CLE @ JAX | 2026-09-13 17:00 | 2026-09-12 17:00 | -10.5000 | 41.0000 | -390.0995 | 390.0995 | 0.0222 | rest_missing |
| NO @ DET | 2026-09-13 17:00 | 2026-09-12 17:00 | -4.5000 | 46.0000 | -175.9084 | 175.9084 | 0.0288 | rest_missing |
| NYJ @ TEN | 2026-09-13 17:00 | 2026-09-12 17:00 | -4.5000 | 43.0000 | -175.2736 | 175.2736 | 0.0289 | rest_missing |
| TB @ CIN | 2026-09-13 17:00 | 2026-09-12 17:00 | -2.5000 | 48.0000 | -137.5084 | 137.5084 | 0.0299 | rest_missing |
| ARI @ LAC | 2026-09-13 20:25 | 2026-09-12 20:25 | -5.5000 | 44.5000 | -205.8736 | 205.8736 | 0.0278 | rest_missing |
| GB @ MIN | 2026-09-13 20:25 | 2026-09-12 20:25 | -0.5000 | 42.5000 | -104.9368 | 104.9368 | 0.0306 | rest_missing |
| MIA @ LV | 2026-09-13 20:25 | 2026-09-12 20:25 | 2.5000 | 40.5000 | 138.0937 | -138.0937 | 0.0302 | rest_missing |
| WAS @ PHI | 2026-09-13 20:25 | 2026-09-12 20:25 | -6.0000 | 44.5000 | -207.9499 | 207.9499 | 0.0277 | rest_missing |
| DAL @ NYG | 2026-09-14 00:20 | 2026-09-13 00:20 | -2.5000 | 52.0000 | -134.4931 | 134.4931 | 0.0299 | rest_missing |
| DEN @ KC | 2026-09-15 00:15 | 2026-09-14 00:15 | 1.0000 | 41.5000 | 110.8984 | -110.8984 | 0.0306 | rest_missing |

## Market comparison and paper backtest

unavailable: no eligible timestamped odds

## Limitations

- Historical reconstruction uses current pinned nflverse files with a 48-hour eligibility convention; it is not a point-in-time-certified backtest. Upstream EPA values may have been revised or computed with models trained later (see the EPA-free ablation).
- No injuries, projected starters, weather, or depth charts are used. This limits real pregame usefulness and is disclosed rather than hidden.
- Scores come from a rounded, zero-floored bivariate normal. It permits implausible scores and smooths NFL key numbers (3, 7) and total parity; key-number diagnostics expose this.
- The 2025 holdout is a retrospective project holdout, not proof of no prior knowledge of its outcomes.
- Bookmaker lines are price references, not conditional means. A model-market gap is not proof of mispricing, and no positive ROI is claimed.

## Provenance

```json
{
 "run_id": "forecast-20260916T185418Z-1704bd",
 "config_hash": "6a0abc279a98c4481f6d0fbd5dbb2858a7b99998cfafc54cd7f01f4a97015593",
 "source_file_hashes": {
  "pbp/2010": "d2ef9c407319719910ad8de8c4c1b88f0ea945ffb3f78f48ef32fa3ed3a7573b",
  "pbp/2011": "c54753b8137b88dab260489b1462cbf1309c3d11741eb8d10e5f1b0a9b8795b1",
  "pbp/2012": "447b90789104aac70427ec9d3b51df7247234029d0ecabd269279369b1cc6278",
  "pbp/2013": "f454605a1983b3946baf0ffadf3a0b9f5e04735c4b84732e0483f2049a780d00",
  "pbp/2014": "54c01408d8f0ff9e013a4ea51428b91d610101f72545d082e932f5710e2fb4f7",
  "pbp/2015": "01b5ae7d06633a2e66b418404461a3d4ff055ad2f3a87d453cebcc8d3fda7ed9",
  "pbp/2016": "95eba04e2145e3c1c8ca502f2a3a76cfb0a5990680c3fb480f02a74a45f54a3b",
  "pbp/2017": "84eacd963c1fdd45965f6222c62e9329a7f3412f029d92c1ac5e34a1bb4d3710",
  "pbp/2018": "2e6f2dce7c7ebd46e985cabe0c17eb72b39a77f98cb4478409294f50b5820150",
  "pbp/2019": "60c3067017db2d28a78f66a79b657268be8578d9a5288e6a827efdcd7fe42540",
  "pbp/2020": "8889f5d8782b5c5dce3c3644acdc0322b228a7e9cb13fa36cbac250e260e9f60",
  "pbp/2021": "333ad34378e5339d5172717cc83378e908daf02c8699416ab3e17c2ec10f78d8",
  "pbp/2022": "931121d8897779d7944e2a293e92ed8799c8e5cceef84096ac42339003fedc09",
  "pbp/2023": "bd3484731408def6b0ec93225bba2bd7b2c65769ca707a2b9444d891abdc6776",
  "pbp/2024": "3fd2896bc0b911b615142d2f1fabae54a4bbba5ab7b73b28187b118ef8af6a3b",
  "pbp/2025": "c6ecedd6d678cc37ed316b23ef84ee1ec6abb69c514bb11868a7ebd5a367df29",
  "pbp/2026": "e7567f9931b02ddafca608ff91ca8f6443a864a1a0d18aef4abe3f4fc49562bc",
  "schedules/all": "fa6684321ed9d08ee7496dfa9a4d345e9445cba7e8842c42f383ad524dc442d0"
 },
 "output_hashes": {
  "predictions.parquet": "6cba27c2537cb7a1625695ecf4dc455e77cfa4fb2eb76bd40565503b24ee54f4"
 },
 "dependency_versions": {
  "duckdb": "1.5.5",
  "matplotlib": "3.11.2",
  "nfl_origination": "0.1.0",
  "numpy": "2.5.3",
  "pandas": "3.0.5",
  "pyarrow": "25.0.1",
  "pydantic": "2.13.5",
  "python": "3.12.14",
  "scikit-learn": "1.9.1",
  "scipy": "1.18.1",
  "streamlit": "1.64.0",
  "typer": "0.27.2"
 },
 "notes": [
  "horizon=standard_reconstruction",
  "historical reconstruction of the standard kickoff-24h horizon; not a live forecast",
  "training_data_mode=historical_reconstruction; forecast_input_mode=historical_reconstruction"
 ]
}
```
