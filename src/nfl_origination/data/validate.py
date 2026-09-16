"""Season coverage audit and auditable exclusion table (spec sections 5, 7, 11 step 1)."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.data.normalize import NormalizedData, expected_regular_season_games

EXCLUSION_COLUMNS = ["game_id", "season", "week", "reason", "detail"]


class SeasonCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int
    scheduled_games: int
    expected_games: int
    final_games: int
    games_with_pbp: int
    plays: int
    excluded_games: int
    complete: bool


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seasons: list[SeasonCoverage]
    exclusions: list[dict[str, Any]] = Field(default_factory=list)
    all_complete: bool
    notes: list[str] = Field(default_factory=list)

    def exclusions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.exclusions, columns=EXCLUSION_COLUMNS)


def validate_normalized(
    data: NormalizedData, seasons: list[int], *, require_pbp: bool = True
) -> ValidationReport:
    """Audit per-season coverage. Missing games never vanish silently; each gets a reason."""
    games, results, plays = data.games, data.results, data.plays
    exclusions = data.exclusions.to_dict("records") if len(data.exclusions) else []
    pbp_games = set(plays["game_id"].unique()) if len(plays) else set()
    final_ids = set(results["game_id"])
    rows: list[SeasonCoverage] = []
    notes: list[str] = []
    for season in seasons:
        g = games[games["season"] == season]
        expected = expected_regular_season_games(season)
        season_final = g[g["game_id"].isin(final_ids)]
        n_pbp = int(season_final["game_id"].isin(pbp_games).sum())
        season_plays = int((plays["season"] == season).sum()) if len(plays) else 0
        for _, row in g[~g["game_id"].isin(final_ids)].iterrows():
            exclusions.append(
                {
                    "game_id": row["game_id"],
                    "season": season,
                    "week": int(row["week"]),
                    "reason": "no_final_score",
                    "detail": "not a training label or scored game",
                }
            )
        if require_pbp:
            for _, row in season_final[~season_final["game_id"].isin(pbp_games)].iterrows():
                exclusions.append(
                    {
                        "game_id": row["game_id"],
                        "season": season,
                        "week": int(row["week"]),
                        "reason": "pbp_missing",
                        "detail": "final score known but no PBP rows; efficiency metrics null",
                    }
                )
        excluded_ids = {e["game_id"] for e in exclusions if e["season"] == season}
        scheduled_excl = (
            len(data.exclusions[data.exclusions["season"] == season]) if len(data.exclusions) else 0
        )
        complete = (
            (len(g) + scheduled_excl == expected)
            and len(season_final) == len(g)
            and (n_pbp == len(season_final) or not require_pbp)
        )
        if len(g) + scheduled_excl != expected:
            notes.append(
                f"season {season}: {len(g) + scheduled_excl} scheduled games, expected {expected}"
            )
        rows.append(
            SeasonCoverage(
                season=season,
                scheduled_games=int(len(g) + scheduled_excl),
                expected_games=expected,
                final_games=len(season_final),
                games_with_pbp=n_pbp,
                plays=season_plays,
                excluded_games=len(excluded_ids),
                complete=bool(complete),
            )
        )
    return ValidationReport(
        seasons=rows,
        exclusions=exclusions,
        all_complete=all(r.complete for r in rows),
        notes=notes,
    )
