"""Odds import and as-of selection failure paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_origination.market.asof import QuotePolicy, closing_proxy, select_quotes
from nfl_origination.market.import_csv import ODDS_CSV_COLUMNS, import_odds
from tests.conftest import ts

BASE = {
    "quote_id": "q1",
    "provider_event_id": "ev1",
    "game_id": "2020_01_T02_T01",
    "bookmaker": "book",
    "market": "spread",
    "selection": "home",
    "line": "-3.0",
    "decimal_odds": "1.91",
    "snapshot_at_utc": "2020-09-12T16:30:00Z",
    "bookmaker_updated_at_utc": "2020-09-12T16:25:00Z",
    "kickoff_at_snapshot_utc": "2020-09-13T17:00:00Z",
    "settlement_rule": "two_way_tie_void",
}


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows, columns=ODDS_CSV_COLUMNS)
    path = tmp_path / "odds.csv"
    df.to_csv(path, index=False)
    return path


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {"game_id": ["2020_01_T02_T01"], "kickoff_utc": [ts("2020-09-13T17:00:00Z")]}
    )


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"line": "-3.25"}, "invalid_line_grid"),
        ({"decimal_odds": "0.95"}, "invalid_decimal_odds"),
        ({"snapshot_at_utc": "2020-09-12 16:30:00"}, "invalid_snapshot_timestamp"),
        ({"bookmaker_updated_at_utc": "2020-09-12T16:35:00Z"}, "update_after_snapshot"),
        (
            {"market": "moneyline", "line": "", "settlement_rule": "three_way"},
            "unsupported_moneyline_rule",
        ),
        ({"market": "moneyline"}, "moneyline_has_line"),
        ({"selection": "over"}, "invalid_selection"),
        ({"game_id": "2020_01_XXX_YYY"}, "ambiguous_event_join"),
        ({"kickoff_at_snapshot_utc": "2020-09-20T17:00:00Z"}, "ambiguous_event_join"),
        ({"bookmaker": ""}, "missing_book_or_event_id"),
    ],
)
def test_rejections(tmp_path, override, reason):
    path = _write(tmp_path, [BASE | override])
    odds, rep = import_odds(path, games=_games())
    assert odds.empty
    assert rep.rejected[0]["reason"] == reason


def test_crosswalk_allows_rescheduled_event(tmp_path):
    path = _write(tmp_path, [BASE | {"kickoff_at_snapshot_utc": "2020-09-20T17:00:00Z"}])
    xw = pd.DataFrame({"provider_event_id": ["ev1"], "game_id": ["2020_01_T02_T01"]})
    odds, rep = import_odds(path, games=_games(), crosswalk=xw)
    assert len(odds) == 1 and not rep.rejected


def test_duplicate_conflict_rejects_both(tmp_path):
    path = _write(tmp_path, [BASE, BASE | {"quote_id": "q2", "decimal_odds": "1.95"}])
    odds, rep = import_odds(path, games=_games())
    assert odds.empty
    assert {r["reason"] for r in rep.rejected} == {"duplicate_conflict"}


def test_missing_update_timestamp_kept_but_ineligible(tmp_path):
    rows = [
        BASE | {"bookmaker_updated_at_utc": ""},
        BASE
        | {"quote_id": "q2", "selection": "away", "line": "3.0", "bookmaker_updated_at_utc": ""},
    ]
    odds, _rep = import_odds(path := _write(tmp_path, rows), games=_games())
    assert len(odds) == 2 and path.exists()
    eligible = select_quotes(odds, ts("2020-09-12T17:00:00Z"), QuotePolicy("book"))
    assert eligible.empty
    assert {e["reason"] for e in eligible.exclusions} == {"missing_update_timestamp"}


def _pair(
    snap: str,
    upd: str,
    line: float = -3.0,
    book: str = "book",
    market: str = "spread",
    suffix: str = "",
) -> list[dict]:
    if market == "spread":
        sels = [("home", str(line)), ("away", str(-line))]
    elif market == "total":
        sels = [("over", str(line)), ("under", str(line))]
    else:
        sels = [("home", ""), ("away", "")]
    return [
        BASE
        | {
            "quote_id": f"{market}{suffix}-{sel}-{snap}",
            "market": market,
            "selection": sel,
            "line": ln,
            "snapshot_at_utc": snap,
            "bookmaker_updated_at_utc": upd,
            "bookmaker": book,
        }
        for sel, ln in sels
    ]


def test_selection_rules(tmp_path):
    decision = ts("2020-09-12T17:00:00Z")
    rows = []
    rows += _pair("2020-09-12T16:50:00Z", "2020-09-12T16:45:00Z")  # eligible, newest complete
    rows += _pair("2020-09-12T16:20:00Z", "2020-09-12T16:15:00Z", line=-2.5)  # superseded
    rows += _pair("2020-09-12T17:05:00Z", "2020-09-12T17:00:00Z", line=-3.5)  # future
    rows += _pair("2020-09-12T15:00:00Z", "2020-09-12T14:55:00Z", line=-1.5)  # stale (>60 min)
    rows += _pair("2020-09-12T16:55:00Z", "2020-09-12T16:50:00Z", book="other")  # wrong book
    rows += _pair("2020-09-12T16:58:00Z", "2020-09-12T16:57:00Z", market="total", line=44.5)[
        :1
    ]  # incomplete pair
    mismatch = _pair("2020-09-12T16:59:00Z", "2020-09-12T16:58:00Z", suffix="mm")
    mismatch[1]["line"] = "2.5"  # lines not opposite
    rows += mismatch
    odds, rep = import_odds(_write(tmp_path, rows), games=_games())
    assert not rep.rejected
    eligible = select_quotes(odds, decision, QuotePolicy("book", 60))
    assert len(eligible.quotes) == 2
    assert set(eligible.quotes["line"]) == {-3.0, 3.0}
    reasons = {e["reason"] for e in eligible.exclusions}
    assert {
        "snapshot_after_decision_time",
        "stale_snapshot",
        "wrong_bookmaker",
        "incomplete_pair_at_snapshot",
        "superseded_by_newer_snapshot",
    } <= reasons


def test_closing_proxy_window(tmp_path):
    kickoff = ts("2020-09-13T17:00:00Z")
    rows = _pair("2020-09-13T16:45:00Z", "2020-09-13T16:44:00Z", line=-3.5)
    rows += _pair(
        "2020-09-13T16:20:00Z", "2020-09-13T16:19:00Z", line=-3.0
    )  # outside 30 min window
    rows += _pair(
        "2020-09-13T17:00:00Z", "2020-09-13T16:59:00Z", line=-4.0
    )  # not strictly before kickoff
    odds, _ = import_odds(_write(tmp_path, rows), games=_games())
    close = closing_proxy(odds, kickoff, QuotePolicy("book"), game_id="2020_01_T02_T01")
    assert set(close["line"]) == {-3.5, 3.5}


def test_synthetic_fixture_imports_with_expected_rejects(fixtures_dir):
    odds, rep = import_odds(fixtures_dir / "odds_synthetic.csv", synthetic=True)
    assert rep.rows_accepted > 8000
    reasons = {r["reason"] for r in rep.rejected}
    assert {"invalid_line_grid", "unsupported_moneyline_rule", "invalid_decimal_odds"} <= reasons
    assert odds["synthetic"].all()
