"""Deterministic synthetic multi-season fixtures (spec section 17).

Synthetic teams, schedules, aggregated team-game metrics and scores are generated from a seed.
They exercise the pipeline mechanics and are labeled synthetic everywhere; they are not NFL
observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_origination.features.aggregate import DEFENSE_NUMERIC, OFFENSE_NUMERIC
from nfl_origination.provenance import hash_frame

SYNTHETIC_LABEL = "SYNTHETIC FIXTURE — not real NFL data"


@dataclass
class SyntheticDataset:
    games: pd.DataFrame
    results: pd.DataFrame
    team_games: pd.DataFrame
    label: str = SYNTHETIC_LABEL


def _schedule(rng: np.random.Generator, season: int, teams: list[str], weeks: int) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp(f"{season}-09-10T17:00:00Z")
    for week in range(1, weeks + 1):
        order = rng.permutation(len(teams))
        pairs = order.reshape(-1, 2)
        week_start = base + pd.Timedelta(days=7 * (week - 1))
        for g, (i, j) in enumerate(pairs):
            home, away = teams[i], teams[j]
            if g == 0:
                kickoff = week_start - pd.Timedelta(days=3, hours=-3)  # Thursday night
            elif g == len(pairs) - 1:
                kickoff = week_start + pd.Timedelta(days=1, hours=3)  # Monday night
            else:
                kickoff = week_start + pd.Timedelta(hours=int(rng.integers(0, 2)) * 3)
            neutral = week == 5 and g == 3
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "home_team": home,
                    "away_team": away,
                    "home_team_original": home,
                    "away_team_original": away,
                    "kickoff_utc": kickoff,
                    "neutral_site": neutral,
                    "status": "final",
                    "source_dataset": "synthetic",
                }
            )
    return pd.DataFrame(rows)


def generate_synthetic_dataset(
    seed: int = 7,
    seasons: tuple[int, int] = (2010, 2020),
    n_teams: int = 32,
    weeks: int = 17,
    lag_hours: float = 48.0,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(1, n_teams + 1)]
    offense = rng.normal(0.0, 0.05, n_teams)
    defense = rng.normal(0.0, 0.04, n_teams)
    games_frames, tg_frames = [], []
    for season in range(seasons[0], seasons[1] + 1):
        offense = 0.8 * offense + rng.normal(0.0, 0.03, n_teams)
        defense = 0.8 * defense + rng.normal(0.0, 0.025, n_teams)
        strength = {t: (offense[i], defense[i]) for i, t in enumerate(teams)}
        sched = _schedule(rng, season, teams, weeks)
        n = len(sched)
        rows = []
        for g in sched.itertuples(index=False):
            hv = 0.0 if g.neutral_site else 1.0
            for team, opp, is_home in (
                (g.home_team, g.away_team, True),
                (g.away_team, g.home_team, False),
            ):
                o, d_opp = strength[team][0], strength[opp][1]
                plays = int(np.clip(rng.normal(62, 6), 40, 90))
                epa_rate = 0.0 + o - d_opp + rng.normal(0, 0.12)
                dropbacks = int(np.clip(rng.binomial(plays, 0.6), 10, plays - 5))
                rushes = plays - dropbacks
                success = int(
                    np.clip(rng.binomial(plays, np.clip(0.43 + 1.2 * epa_rate, 0.2, 0.7)), 0, plays)
                )
                explosive = int(
                    np.clip(
                        rng.binomial(plays, np.clip(0.09 + 0.3 * epa_rate, 0.02, 0.25)), 0, plays
                    )
                )
                pts_latent = 22.5 + 45.0 * epa_rate + (2.0 * hv if is_home else -2.0 * hv)
                points = int(max(0, round(rng.normal(pts_latent, 6.5))))
                rows.append(
                    {
                        "game_id": g.game_id,
                        "team_id": team,
                        "opponent_id": opp,
                        "is_home": is_home,
                        "off_eligible_plays": float(plays),
                        "off_epa_plays": float(plays),
                        "off_epa_sum": float(plays * epa_rate),
                        "off_success_count": float(success),
                        "off_dropback_epa_plays": float(dropbacks),
                        "off_dropback_epa_sum": float(dropbacks * (epa_rate + 0.05)),
                        "off_rush_epa_plays": float(rushes),
                        "off_rush_epa_sum": float(rushes * (epa_rate - 0.08)),
                        "off_yard_plays": float(plays),
                        "off_explosive_count": float(explosive),
                        "points_for": points,
                    }
                )
        tg = pd.DataFrame(rows)
        pts = tg.pivot(index="game_id", columns="is_home", values="points_for")
        sched = sched.merge(
            pts.rename(columns={True: "home_score", False: "away_score"}).reset_index(),
            on="game_id",
        )
        games_frames.append(sched)
        opp = tg[["game_id", "team_id", *OFFENSE_NUMERIC, "points_for"]].rename(
            columns={"team_id": "opponent_id", "points_for": "points_against"}
        )
        opp = opp.rename(
            columns={
                c: c.replace("off_", "def_", 1)
                for c in OFFENSE_NUMERIC
                if c != "off_eligible_plays"
            }
        ).drop(columns=["off_eligible_plays"])
        tg = tg.merge(opp, on=["game_id", "opponent_id"])
        tg = tg.merge(
            sched[["game_id", "season", "week", "kickoff_utc", "neutral_site"]], on="game_id"
        )
        tg_frames.append(tg)
        del n
    games = pd.concat(games_frames, ignore_index=True)
    team_games = pd.concat(tg_frames, ignore_index=True)
    results = pd.DataFrame(
        {
            "game_id": games["game_id"],
            "home_score": games["home_score"].astype(np.int64),
            "away_score": games["away_score"].astype(np.int64),
            "status": "final",
            "eligible_from_utc": games["kickoff_utc"] + pd.Timedelta(hours=lag_hours),
            "overtime": False,
        }
    )
    games = games.drop(columns=["home_score", "away_score"])
    games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], utc=True)
    games["status"] = games["status"].astype(str)
    team_games["kickoff_utc"] = pd.to_datetime(team_games["kickoff_utc"], utc=True)
    team_games["game_count"] = np.int64(1)
    team_games["points_for"] = team_games["points_for"].astype(np.int64)
    team_games["points_against"] = team_games["points_against"].astype(np.int64)
    team_games["has_pbp"] = True
    for c in OFFENSE_NUMERIC + DEFENSE_NUMERIC:
        team_games[c] = team_games[c].astype(float)
    team_games = team_games.sort_values(["kickoff_utc", "game_id", "team_id"]).reset_index(
        drop=True
    )
    team_games["source_hash"] = hash_frame(team_games[["game_id", "team_id", "points_for"]])
    return SyntheticDataset(games=games, results=results, team_games=team_games)
