# NFL Origination Model — forecast run `forecast-20260916T175305Z-6f5afe`

Created 2026-09-16T17:53:05.934629Z · data mode `historical_reconstruction` · forecast policy `kickoff_minus_24h` · git `518434a9d3a2e56d284c8a8bc701b181e3da819d` (dirty) · config hash `6a0abc279a98`

Forecast slate `2026_week2` · model `M1_ridge_alpha100` · policy `kickoff_minus_24h`

## Slate

| game | kickoff (UTC) | cutoff (UTC) | fair home handicap | fair total | home ML | away ML | P(tie) | flags |
|---|---|---|---|---|---|---|---|---|
| DET @ BUF | 2026-09-18 00:15 | 2026-09-17 00:15 | -5.5000 | 52.0000 | -201.9089 | 201.9089 | 0.0278 | cutoff_in_future_features_from_current_cache |
| CAR @ ATL | 2026-09-20 17:00 | 2026-09-19 17:00 | -3.0000 | 46.0000 | -145.7239 | 145.7239 | 0.0297 | cutoff_in_future_features_from_current_cache |
| CIN @ HOU | 2026-09-20 17:00 | 2026-09-19 17:00 | -3.5000 | 49.5000 | -152.5752 | 152.5752 | 0.0294 | cutoff_in_future_features_from_current_cache |
| CLE @ TB | 2026-09-20 17:00 | 2026-09-19 17:00 | -6.0000 | 41.5000 | -212.8493 | 212.8493 | 0.0277 | cutoff_in_future_features_from_current_cache |
| GB @ NYJ | 2026-09-20 17:00 | 2026-09-19 17:00 | 2.5000 | 45.0000 | 136.8467 | -136.8467 | 0.0299 | cutoff_in_future_features_from_current_cache |
| MIN @ CHI | 2026-09-20 17:00 | 2026-09-19 17:00 | -6.0000 | 49.0000 | -220.3905 | 220.3905 | 0.0272 | cutoff_in_future_features_from_current_cache |
| NO @ BAL | 2026-09-20 17:00 | 2026-09-19 17:00 | -5.0000 | 46.5000 | -187.9505 | 187.9505 | 0.0283 | cutoff_in_future_features_from_current_cache |
| PHI @ TEN | 2026-09-20 17:00 | 2026-09-19 17:00 | 4.5000 | 42.0000 | 179.9211 | -179.9211 | 0.0287 | cutoff_in_future_features_from_current_cache |
| PIT @ NE | 2026-09-20 17:00 | 2026-09-19 17:00 | -7.5000 | 45.0000 | -268.1526 | 268.1526 | 0.0256 | cutoff_in_future_features_from_current_cache |
| JAX @ DEN | 2026-09-20 20:05 | 2026-09-19 20:05 | 4.0000 | 45.0000 | 169.7723 | -169.7723 | 0.0290 | cutoff_in_future_features_from_current_cache |
| LV @ LAC | 2026-09-20 20:05 | 2026-09-19 20:05 | -5.5000 | 38.0000 | -203.9613 | 203.9613 | 0.0283 | cutoff_in_future_features_from_current_cache |
| MIA @ SF | 2026-09-20 20:25 | 2026-09-19 20:25 | -9.5000 | 46.0000 | -350.1653 | 350.1653 | 0.0231 | cutoff_in_future_features_from_current_cache |
| SEA @ ARI | 2026-09-20 20:25 | 2026-09-19 20:25 | 4.5000 | 45.0000 | 173.0515 | -173.0515 | 0.0289 | cutoff_in_future_features_from_current_cache |
| WAS @ DAL | 2026-09-20 20:25 | 2026-09-19 20:25 | -4.0000 | 51.0000 | -162.5306 | 162.5306 | 0.0291 | cutoff_in_future_features_from_current_cache |
| IND @ KC | 2026-09-21 00:20 | 2026-09-20 00:20 | -6.0000 | 44.0000 | -208.4444 | 208.4444 | 0.0277 | cutoff_in_future_features_from_current_cache |
| NYG @ LA | 2026-09-22 00:15 | 2026-09-21 00:15 | -4.0000 | 50.5000 | -164.4984 | 164.4984 | 0.0291 | cutoff_in_future_features_from_current_cache |

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
 "run_id": "forecast-20260916T175305Z-6f5afe",
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
  "predictions.parquet": "a78ea5de642a54296acef0e070688c613552d9057f061cc5a77faa6b86e8bbd6"
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
 "notes": []
}
```
