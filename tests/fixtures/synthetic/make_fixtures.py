"""Regenerate the committed synthetic fixtures deterministically.

Run: uv run python tests/fixtures/synthetic/make_fixtures.py
Produces mini_schedule.csv, mini_pbp.csv (nflverse-shaped) and odds_synthetic.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from nfl_origination.synthetic import generate_synthetic_dataset  # noqa: E402

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "qtr",
    "play_type",
    "epa",
    "yards_gained",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "two_point_attempt",
    "qb_scramble",
    "sack",
    "desc",
]


def mini_schedule() -> pd.DataFrame:
    rows = [
        # season, game_type, week, gameday, weekday, gametime, away, away_score,
        # home, home_score, location, overtime
        (
            "2023_01_AAA_BBB",
            2023,
            "REG",
            1,
            "2023-09-10",
            "Sunday",
            "13:00",
            "AAA",
            20,
            "BBB",
            27,
            "Home",
            0,
        ),
        (
            "2023_01_CCC_DDD",
            2023,
            "REG",
            1,
            "2023-09-10",
            "Sunday",
            "16:25",
            "CCC",
            24,
            "DDD",
            24,
            "Home",
            1,
        ),
        (
            "2023_02_BBB_CCC",
            2023,
            "REG",
            2,
            "2023-09-17",
            "Sunday",
            "13:00",
            "BBB",
            17,
            "CCC",
            31,
            "Neutral",
            0,
        ),
        (
            "2023_02_DDD_AAA",
            2023,
            "REG",
            2,
            "2023-09-18",
            "Monday",
            "20:15",
            "DDD",
            10,
            "AAA",
            13,
            "Home",
            0,
        ),
        (
            "2023_03_AAA_CCC",
            2023,
            "REG",
            3,
            "2023-11-05",
            "Sunday",
            "13:00",
            "AAA",
            21,
            "CCC",
            28,
            "Home",
            0,
        ),  # DST end day
        (
            "2023_19_AAA_BBB",
            2023,
            "WC",
            19,
            "2024-01-13",
            "Saturday",
            "16:30",
            "AAA",
            7,
            "BBB",
            30,
            "Home",
            0,
        ),  # playoff, excluded
    ]
    cols = [
        "game_id",
        "season",
        "game_type",
        "week",
        "gameday",
        "weekday",
        "gametime",
        "away_team",
        "away_score",
        "home_team",
        "home_score",
        "location",
        "overtime",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["spread_line"] = [3.0, -1.5, 2.5, 7.0, 3.5, 9.5]
    df["total_line"] = [44.5, 47.0, 41.5, 38.5, 45.5, 40.0]
    df["home_moneyline"] = [-150, 120, -130, -300, -170, -450]
    df["away_moneyline"] = [130, -140, 110, 250, 145, 350]
    return df


def mini_pbp() -> pd.DataFrame:
    """Hand-written plays covering every filtering rule (values are synthetic)."""
    plays = []
    pid = 0

    def add(
        game,
        off,
        deff,
        qtr,
        ptype,
        epa,
        yds,
        pas,
        rush,
        kneel=0,
        spike=0,
        two=0,
        scr=0,
        sack=0,
        desc="",
    ):
        nonlocal pid
        pid += 1
        season = int(game[:4])
        week = int(game[5:7])
        stype = "REG" if week < 19 else "POST"
        plays.append(
            [
                game,
                pid,
                season,
                week,
                stype,
                off,
                deff,
                qtr,
                ptype,
                epa,
                yds,
                pas,
                rush,
                kneel,
                spike,
                two,
                scr,
                sack,
                desc,
            ]
        )

    g = "2023_01_AAA_BBB"
    add(g, "AAA", "BBB", 1, "pass", 0.5, 12, 1, 0, desc="completed pass")
    add(g, "AAA", "BBB", 1, "run", -0.3, 2, 0, 1, desc="designed run")
    add(g, "AAA", "BBB", 1, "pass", -1.2, -7, 1, 0, sack=1, desc="sack counts as dropback")
    add(
        g,
        "AAA",
        "BBB",
        2,
        "run",
        0.9,
        22,
        1,
        0,
        scr=1,
        desc="scramble: dropback, not rush, explosive (>=20)",
    )
    add(g, "AAA", "BBB", 2, "no_play", 0.4, 0, 1, 0, desc="penalty no play excluded")
    add(g, "AAA", "BBB", 2, "qb_kneel", -0.1, -1, 0, 0, kneel=1, desc="kneel excluded")
    add(g, "AAA", "BBB", 2, "qb_spike", -0.2, 0, 0, 0, spike=1, desc="spike excluded")
    add(g, "AAA", "BBB", 3, "pass", 0.7, 25, 1, 0, two=1, desc="two point attempt excluded")
    add(
        g,
        "AAA",
        "BBB",
        3,
        "run",
        np.nan,
        11,
        0,
        1,
        desc="missing EPA: yards count, EPA does not; explosive run (>=10)",
    )
    add(g, "AAA", "BBB", 3, "punt", 0.1, 0, 0, 0, desc="punt excluded")
    add(g, "AAA", "BBB", 4, "field_goal", 0.3, 0, 0, 0, desc="field goal excluded")
    add(g, "AAA", "BBB", 5, "pass", 2.0, 40, 1, 0, desc="overtime excluded from efficiency")
    add(g, "BBB", "AAA", 1, "pass", 0.0, 5, 1, 0, desc="zero EPA is not success")
    add(g, "BBB", "AAA", 2, "run", 0.2, 4, 0, 1, desc="")
    add(g, "BBB", "AAA", 3, "pass", 1.5, 30, 1, 0, desc="explosive dropback")
    add(g, "BBB", "AAA", 4, "run", -0.5, -2, 0, 1, desc="")
    add(g, None, None, 1, "kickoff", 0.0, 0, 0, 0, desc="kickoff no posteam")
    for game, home, away in (
        ("2023_01_CCC_DDD", "DDD", "CCC"),
        ("2023_02_BBB_CCC", "CCC", "BBB"),
        ("2023_02_DDD_AAA", "AAA", "DDD"),
        ("2023_03_AAA_CCC", "CCC", "AAA"),
        ("2023_19_AAA_BBB", "BBB", "AAA"),
    ):
        rng = np.random.default_rng(abs(hash(game)) % (2**32))
        for off, deff in ((home, away), (away, home)):
            for _ in range(12):
                is_pass = rng.random() < 0.6
                add(
                    game,
                    off,
                    deff,
                    int(rng.integers(1, 5)),
                    "pass" if is_pass else "run",
                    float(rng.normal(0, 1)),
                    float(rng.normal(6 if is_pass else 4, 8)),
                    int(is_pass),
                    int(not is_pass),
                )
    return pd.DataFrame(plays, columns=PBP_COLUMNS)


def synthetic_odds() -> pd.DataFrame:
    """Synthetic quotes for the demo's confirmation/holdout seasons, plus deliberate rejects."""
    syn = generate_synthetic_dataset(seed=42, seasons=(2010, 2020))
    games = syn.games.merge(syn.results[["game_id", "home_score", "away_score"]], on="game_id")
    games = games[games["season"].isin([2019, 2020])]
    rng = np.random.default_rng(2024)
    rows = []
    qid = 0

    def quote(
        game,
        market,
        selection,
        line,
        dec,
        snap,
        upd,
        book="synthetic_book",
        rule="two_way_tie_void",
        event=None,
    ):
        nonlocal qid
        qid += 1
        rows.append(
            {
                "quote_id": f"q{qid:06d}",
                "provider_event_id": event or f"ev-{game.game_id}",
                "game_id": game.game_id,
                "bookmaker": book,
                "market": market,
                "selection": selection,
                "line": "" if line is None else line,
                "decimal_odds": round(dec, 4),
                "snapshot_at_utc": snap.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "bookmaker_updated_at_utc": ""
                if upd is None
                else upd.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "kickoff_at_snapshot_utc": game.kickoff_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "settlement_rule": rule,
            }
        )

    for g in games.itertuples(index=False):
        kickoff = pd.Timestamp(g.kickoff_utc)
        cutoff = kickoff - pd.Timedelta(hours=24)
        true_margin = g.home_score - g.away_score
        # partially informed synthetic book: shrunk realized margin plus noise (mechanics only)
        book_margin = 0.5 * true_margin + rng.normal(1.5, 6.0)
        spread = round(float(np.round(-book_margin * 2) / 2), 1)
        if spread == int(spread) and rng.random() < 0.5:
            spread += 0.5
        total = round(float(np.round((45 + rng.normal(0, 3)) * 2) / 2), 1)
        for snap_offset in (pd.Timedelta(minutes=45), pd.Timedelta(minutes=10)):
            snap = cutoff - snap_offset
            upd = snap - pd.Timedelta(minutes=3)
            juice = 1.0 + 100.0 / 110.0
            quote(g, "spread", "home", spread, juice, snap, upd)
            quote(g, "spread", "away", -spread, juice, snap, upd)
            quote(g, "total", "over", total, juice, snap, upd)
            quote(g, "total", "under", total, juice, snap, upd)
            p_home = 1 / (1 + np.exp(spread / 7.0))
            quote(g, "moneyline", "home", None, 1 / min(0.97, p_home * 1.04), snap, upd)
            quote(g, "moneyline", "away", None, 1 / min(0.97, (1 - p_home) * 1.04), snap, upd)
        # closing proxy 20 minutes before kickoff with a moved line
        snap = kickoff - pd.Timedelta(minutes=20)
        moved = spread + float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        quote(g, "spread", "home", moved, 1.9091, snap, snap - pd.Timedelta(minutes=1))
        quote(g, "spread", "away", -moved, 1.9091, snap, snap - pd.Timedelta(minutes=1))
        quote(g, "total", "over", total, 1.9091, snap, snap - pd.Timedelta(minutes=1))
        quote(g, "total", "under", total, 1.9091, snap, snap - pd.Timedelta(minutes=1))
    # deliberate rejects / ineligibles on the first game
    g = games.iloc[0]
    cutoff = pd.Timestamp(g.kickoff_utc) - pd.Timedelta(hours=24)
    quote(
        g,
        "spread",
        "home",
        -3.25,
        1.91,
        cutoff - pd.Timedelta(minutes=5),
        cutoff - pd.Timedelta(minutes=6),
    )  # quarter line
    quote(
        g,
        "spread",
        "home",
        -3.0,
        1.91,
        cutoff + pd.Timedelta(minutes=5),
        cutoff + pd.Timedelta(minutes=4),
    )  # future snapshot
    quote(
        g,
        "total",
        "over",
        44.5,
        1.91,
        cutoff - pd.Timedelta(hours=5),
        cutoff - pd.Timedelta(hours=5),
    )  # stale
    quote(
        g, "moneyline", "home", None, 1.8, cutoff - pd.Timedelta(minutes=5), None, rule="three_way"
    )  # unsupported rule
    quote(
        g, "moneyline", "away", None, 2.1, cutoff - pd.Timedelta(minutes=5), None
    )  # missing update timestamp
    quote(
        g,
        "spread",
        "home",
        -3.0,
        0.95,
        cutoff - pd.Timedelta(minutes=5),
        cutoff - pd.Timedelta(minutes=6),
    )  # bad decimal
    quote(
        g,
        "spread",
        "home",
        -3.0,
        1.91,
        cutoff - pd.Timedelta(minutes=5),
        cutoff - pd.Timedelta(minutes=6),
        book="other_book",
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mini_schedule().to_csv(HERE / "mini_schedule.csv", index=False)
    mini_pbp().to_csv(HERE / "mini_pbp.csv", index=False)
    synthetic_odds().to_csv(HERE / "odds_synthetic.csv", index=False)
    print("fixtures written to", HERE)
