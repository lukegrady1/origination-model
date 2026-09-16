# Data dictionary

Schema version 2, feature version 2 (v2 adds `unobserved_inputs`, `prior_games_hash`, and
as-of availability for priors, rest, labels and observation times; the frozen 2025 research
artifacts were produced under version 1 and are preserved as such). Identifiers are strings, counts integers, probabilities
float64, timestamps tz-aware UTC. Missing values stay null until the logged training-only median
imputation inside `models/ridge.py`.

## Artifacts

| Artifact | Key | Content |
|---|---|---|
| `games.parquet` | game_id | season, week, REG, canonical/original franchise codes, `kickoff_utc`, `neutral_site`, status (`final`/`not_final`), source dataset |
| `results.parquet` | game_id | integer nonnegative final scores, overtime flag, `eligible_from_utc = kickoff + 48h` |
| `untimed_reference.parquet` | game_id | schedule betting fields quarantined as `untimed_reference`; never joined to features |
| `team_games.parquet` | game_id + team_id | opponent, home/neutral flags, points for/against, offensive and allowed numerators/denominators, `has_pbp`, source hash |
| `labels.parquet` | game_id + perspective | points and `label_available_utc` |
| `features_<key>.parquet` | game_id + cutoff_utc + perspective | approved features, history counts, cold-start/rest flags, `max_source_eligible_utc`, `max_observed_utc`, source game IDs and hash, `insufficient_warmup`, data mode, feature version |
| `predictions*.parquet` | run_id + game_id + cutoff_utc | regression locations, distribution means, margin/total means, home/tie/away probabilities, conditional home probability, fair decimal/American moneylines, mean and fair handicaps with win/push/loss, fair total with over/push/under, 50/80/95 intervals, tail diagnostics, warnings, flags, hashes; scored runs add actuals, CRPS and per-game metric columns |
| `distributions_<model>.parquet` | game_id | integer margin and total PMFs (list columns) used by market comparison and settlement |
| `odds.parquet` | quote_id | canonical quote with `snapshot_at_utc`, `bookmaker_updated_at_utc`, `kickoff_at_snapshot_utc`, `ingested_at_utc`, `synthetic` |
| `paper_bets.parquet` | run_id + quote_id | prediction/policy IDs, win/push/loss probabilities, EV, stake, settlement, profit, line/price CLV |

## Team-game metric columns

`off_eligible_plays`, `off_epa_plays`/`off_epa_sum`, `off_success_count`,
`off_dropback_epa_plays`/`off_dropback_epa_sum`, `off_rush_epa_plays`/`off_rush_epa_sum`,
`off_yard_plays`/`off_explosive_count`, and the `def_*` mirrors taken from the opponent's offense
in the same game. Eligible plays: Q1–Q4 scrimmage plays with both teams known and either the
dropback or designed-rush flag, excluding no-plays, kneels, spikes, two-point tries and special
teams. Success is `EPA > 0`. Explosive: dropback ≥ 20 yards or designed rush ≥ 10 yards.

## Generated feature dictionary

| column | unit | formula | null_policy | source | availability |
|---|---|---|---|---|---|
| team_off_epa_per_play | EPA per play | shrunk (k=200 plays) recency-weighted (half-life 8 games, offseason x0.5, last 16): sum(off EPA)/count(valid EPA plays) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_off_success_rate | rate | shrunk (k=200 plays) recency-weighted (half-life 8 games, offseason x0.5, last 16): count(EPA>0)/count(valid EPA plays) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_off_dropback_epa | EPA per dropback | shrunk (k=100) recency-weighted (half-life 8 games, offseason x0.5, last 16): sum(dropback EPA)/count(valid dropback EPA plays) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_off_rush_epa | EPA per designed rush | shrunk (k=100) recency-weighted (half-life 8 games, offseason x0.5, last 16): sum(rush EPA)/count(valid rush EPA plays) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_off_explosive_rate | rate | shrunk (k=200) recency-weighted (half-life 8 games, offseason x0.5, last 16): count(dropback>=20yd or rush>=10yd)/count(valid yardage plays) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_plays_per_game | plays | shrunk (k=4 games) recency-weighted (half-life 8 games, offseason x0.5, last 16): eligible plays per completed team game | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| team_points_for | points | shrunk (k=4 games) recency-weighted (half-life 8 games, offseason x0.5, last 16): final points scored incl. OT/ST/defense | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; team perspective (offense) | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_def_epa_allowed | EPA per play | shrunk (k=200) recency-weighted (half-life 8 games, offseason x0.5, last 16): opponent offensive EPA sum/opp valid plays (higher = weaker) | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_def_success_allowed | rate | shrunk (k=200) recency-weighted (half-life 8 games, offseason x0.5, last 16): opponent success count/opp valid plays | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_def_dropback_epa_allowed | EPA per dropback | shrunk (k=100) recency-weighted (half-life 8 games, offseason x0.5, last 16): opponent dropback EPA/opp dropbacks | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_def_rush_epa_allowed | EPA per rush | shrunk (k=100) recency-weighted (half-life 8 games, offseason x0.5, last 16): opponent rush EPA/opp designed rushes | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_def_explosive_allowed | rate | shrunk (k=200) recency-weighted (half-life 8 games, offseason x0.5, last 16): opponent explosive count/opp yardage plays | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_points_against | points | shrunk (k=4 games) recency-weighted (half-life 8 games, offseason x0.5, last 16): final points allowed | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| opp_plays_per_game | plays | shrunk (k=4 games) recency-weighted (half-life 8 games, offseason x0.5, last 16): eligible plays per completed team game | prior r0 when no history (cold start); null only if prior unavailable | nflverse PBP epa/yards_gained/pass/rush and schedule scores; opponent perspective | source games with kickoff+48h<=cutoff (recorded_asof also requires observed<=cutoff) |
| venue_advantage | indicator | +1 home, -1 away, 0 neutral | never null; unknown neutral status fails normalization | schedule.location | schedule metadata |
| rest_difference | days | clip(team rest,3,14) - clip(opp rest,3,14); season opener uses 7 | never null; missing rest flagged | schedule kickoff times before cutoff | prior game kickoff < cutoff, same season |
| team_rest_missing | indicator | 1 if no same-season prior game before cutoff | never null | schedule | cutoff |
| opp_rest_missing | indicator | as above for opponent | never null | schedule | cutoff |
| team_history_games | games | min(eligible completed games, 16) | never null | team_games | kickoff+48h<=cutoff (and observed<=cutoff in recorded_asof) |
| opp_history_games | games | as above for opponent | never null | team_games | as above |
| team_cold_start | indicator | 1 if zero eligible history games (prior only) | never null | team_games | as above |
| opp_cold_start | indicator | as above for opponent | never null | team_games | as above |

Prediction fields distinguish `mu_*` (regression location) from `dist_mean_*` (mean of the
discrete distribution after the residual-mean shift and zero flooring). Intervals are outcome
prediction intervals, not confidence intervals for the fair price.
