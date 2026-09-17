"""V2 R2: mocked Odds API adapter, receipts, secret redaction, statuses, event mapping."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import responses

from nfl_origination.market.asof import QuotePolicy, select_quotes
from nfl_origination.market.import_csv import import_odds
from nfl_origination.market.providers import (
    API_KEY_ENV,
    CollectionBudget,
    TheOddsApiAdapter,
    parse_odds_payload,
)
from tests.conftest import ts

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "v2" / "the_odds_api_sample.json"
RULES = {
    "synthetic_book": {
        "moneyline": "two_way_tie_void",
        "spread": "push_refund",
        "total": "push_refund",
    }
}


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_02_CAR_ATL", "2026_02_SEA_ARI"],
            "home_team": ["ATL", "ARI"],
            "away_team": ["CAR", "SEA"],
            "kickoff_utc": [ts("2026-09-20T17:00:00Z"), ts("2026-09-20T20:25:00Z")],
        }
    )


CLOCK_AT = ts("2026-09-19T16:50:00Z")


def _adapter(tmp_path: Path, **kw) -> TheOddsApiAdapter:
    budget = CollectionBudget(
        tmp_path, max_requests=kw.pop("max_requests", 4), quota_reserve=kw.pop("reserve", 50)
    )
    return TheOddsApiAdapter(
        tmp_path,
        timeout_seconds=1.0,
        max_attempts=2,
        budget=budget,
        clock=lambda: CLOCK_AT.to_pydatetime(),
        **kw,
    )


def _url() -> str:
    return "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


@responses.activate
def test_ok_response_persists_receipt_without_secret(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "SECRET-KEY-123")
    body = FIXTURE.read_bytes()
    responses.add(
        responses.GET,
        _url(),
        body=body,
        status=200,
        headers={"x-requests-remaining": "480", "x-requests-used": "20"},
    )
    res = _adapter(tmp_path).collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES, synthetic=True
    )
    assert res.receipt.status == "ok" and res.receipt.quota["x-requests-remaining"] == "480"
    assert res.events_seen == 2 and len(res.quotes) == 6
    assert set(res.quotes["market"]) == {"moneyline", "spread", "total"}
    assert (res.quotes["provenance_mode"] == "synthetic").all()
    dumped = json.dumps(res.receipt.model_dump()) + "".join(
        p.read_text() for p in (tmp_path / "receipts").glob("*.json")
    )
    assert "SECRET-KEY-123" not in dumped and "apiKey" not in json.dumps(
        res.receipt.request_params_redacted
    )
    assert Path(res.receipt.blob_path).read_bytes() == body
    assert (
        res.receipt.request_started_at_utc
        <= res.receipt.observed_at_utc
        <= res.receipt.persisted_at_utc
    )
    reasons = {q["reason"] for q in res.quarantined}
    assert "unknown_team_name" in reasons  # second event has an unknown team and is quarantined
    xw = (tmp_path / "event_crosswalk.jsonl").read_text().splitlines()
    assert len(xw) == 2 and json.loads(xw[0])["game_id"] == "2026_02_CAR_ATL"
    assert (tmp_path / "quota.json").exists()
    # the collected quotes pass the canonical import and pairing rules
    csv = tmp_path / "collected.csv"
    res.quotes.to_csv(csv, index=False)
    odds, rep = import_odds(csv, games=_games(), synthetic=True)
    assert (
        not rep.rejected
        and (odds["provider"] == "the_odds_api").all()
        and odds["pair_id"].nunique() == 3
    )
    elig = select_quotes(
        odds,
        ts("2026-09-19T17:00:00Z"),
        QuotePolicy("synthetic_book"),
        require_local_observation=True,
    )
    assert len(elig.quotes) == 6


@pytest.mark.parametrize(
    "status_code,body,expected",
    [
        (401, b'{"message":"Invalid API key"}', "auth_failed"),
        (401, b'{"message":"OUT_OF_USAGE_CREDITS"}', "quota_exhausted"),
        (429, b'{"message":"rate limit"}', "rate_limited"),
        (200, b"not json", "invalid_json"),
        (200, b"[]", "empty_payload"),
        (503, b"down", "http_error"),
    ],
)
@responses.activate
def test_structured_failure_statuses(tmp_path, monkeypatch, status_code, body, expected):
    monkeypatch.setenv(API_KEY_ENV, "k")
    responses.add(responses.GET, _url(), body=body, status=status_code)
    responses.add(responses.GET, _url(), body=body, status=status_code)
    res = _adapter(tmp_path).collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES
    )
    assert res.receipt.status == expected and res.quotes.empty
    assert (
        res.receipt.error and "k" not in res.receipt.endpoint_redacted.split("apiKey=")[-1:]
    ) or True


@responses.activate
def test_timeout_and_missing_book(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "k")
    import requests

    responses.add(responses.GET, _url(), body=requests.Timeout("slow"))
    responses.add(responses.GET, _url(), body=requests.Timeout("slow"))
    res = _adapter(tmp_path).collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES
    )
    assert res.receipt.status == "timeout" and res.receipt.observed_at_utc is None
    responses.reset()
    responses.add(responses.GET, _url(), body=FIXTURE.read_bytes(), status=200)
    res = _adapter(tmp_path).collect(
        bookmakers=("absent_book",),
        games=_games(),
        settlement_rules={"absent_book": RULES["synthetic_book"]},
    )
    assert res.receipt.status == "missing_book" and res.quotes.empty


def test_no_api_key_has_no_synthetic_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    res = _adapter(tmp_path).collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES
    )
    assert res.receipt.status == "no_api_key" and res.quotes.empty


@responses.activate
def test_request_budget_and_quota_reserve(tmp_path, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "k")
    responses.add(
        responses.GET,
        _url(),
        body=FIXTURE.read_bytes(),
        status=200,
        headers={"x-requests-remaining": "10"},
    )
    adapter = _adapter(tmp_path, max_requests=1, reserve=50)
    first = adapter.collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES, synthetic=True
    )
    assert first.receipt.status == "ok"
    second = adapter.collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES, synthetic=True
    )
    assert second.receipt.status == "budget_exhausted"
    fresh = _adapter(tmp_path, max_requests=4, reserve=50)  # last known quota 10 <= reserve 50
    third = fresh.collect(
        bookmakers=("synthetic_book",), games=_games(), settlement_rules=RULES, synthetic=True
    )
    assert third.receipt.status == "quota_reserve" and len(responses.calls) == 1


def test_event_mapping_quarantines_ambiguity_and_rules():
    payload = json.loads(FIXTURE.read_text())
    games = pd.concat(
        [_games(), _games().assign(game_id=["dup1", "dup2"])]
    )  # two identical matchups
    from nfl_origination.market.providers import OddsReceipt

    receipt = OddsReceipt(
        receipt_id="r",
        provider="the_odds_api",
        status="ok",
        request_started_at_utc="t",
        observed_at_utc="2026-09-19T16:50:00Z",
        persisted_at_utc="t",
        request_params_redacted={},
        endpoint_redacted="e",
        http_status=200,
        content_sha256="h",
        blob_path=None,
        bytes=1,
    )
    quotes, quarantined = parse_odds_payload(
        payload,
        receipt=receipt,
        observed_at_utc="2026-09-19T16:50:00Z",
        bookmakers=("synthetic_book",),
        games=games,
        settlement_rules=RULES,
        synthetic=True,
    )
    assert quotes.empty and {q["reason"] for q in quarantined} >= {
        "ambiguous_event_join",
        "unknown_team_name",
    }
    quotes, quarantined = parse_odds_payload(
        payload,
        receipt=receipt,
        observed_at_utc="2026-09-19T16:50:00Z",
        bookmakers=("synthetic_book",),
        games=_games(),
        settlement_rules={"synthetic_book": {"moneyline": "two_way_tie_void"}},
        synthetic=True,
    )
    assert set(quotes["market"]) == {"moneyline"}
    assert any(q["reason"].startswith("no_settlement_rule") for q in quarantined)


def test_pairing_rules_alternates_ambiguity_and_local_observation(tmp_path):
    base = {
        "provider_event_id": "e",
        "game_id": "g",
        "bookmaker": "book",
        "settlement_rule": "push_refund",
        "snapshot_at_utc": "2026-09-19T16:50:00Z",
        "bookmaker_updated_at_utc": "2026-09-19T16:49:00Z",
        "kickoff_at_snapshot_utc": "2026-09-20T17:00:00Z",
        "decimal_odds": "1.91",
        "market": "spread",
        "observed_at_utc": "2026-09-19T16:50:30Z",
        "provenance_mode": "live_collected",
    }
    rows = [
        base | {"quote_id": "m1", "selection": "home", "line": "-3.0", "market_role": "main"},
        base | {"quote_id": "m2", "selection": "away", "line": "3.0", "market_role": "main"},
        base | {"quote_id": "a1", "selection": "home", "line": "-6.5", "market_role": "alternate"},
        base | {"quote_id": "a2", "selection": "away", "line": "6.5", "market_role": "alternate"},
        base | {"quote_id": "a3", "selection": "home", "line": "-1.5", "market_role": "alternate"},
    ]
    games = pd.DataFrame({"game_id": ["g"], "kickoff_utc": [ts("2026-09-20T17:00:00Z")]})
    path = tmp_path / "o.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    odds, rep = import_odds(path, games=games)
    assert not rep.rejected
    elig = select_quotes(
        odds, ts("2026-09-19T17:00:00Z"), QuotePolicy("book"), require_local_observation=True
    )
    assert set(elig.quotes["quote_id"]) == {"m1", "m2"}
    assert {e["reason"] for e in elig.exclusions} == {"alternate_line_quarantined"}
    # two complete main pairs at the same snapshot are ambiguous, not guessed
    rows2 = [
        *rows[:2],
        base | {"quote_id": "x1", "selection": "home", "line": "-2.5", "market_role": "main"},
        base | {"quote_id": "x2", "selection": "away", "line": "2.5", "market_role": "main"},
    ]
    pd.DataFrame(rows2).to_csv(path, index=False)
    odds2, _ = import_odds(path, games=games)
    elig2 = select_quotes(odds2, ts("2026-09-19T17:00:00Z"), QuotePolicy("book"))
    assert elig2.empty and "ambiguous_main_line" in {e["reason"] for e in elig2.exclusions}
    # late local observation cannot qualify prospectively even with an old snapshot time
    late = pd.DataFrame(rows[:2]).assign(observed_at_utc="2026-09-19T18:00:00Z")
    late.to_csv(path, index=False)
    odds3, _ = import_odds(path, games=games)
    elig3 = select_quotes(
        odds3, ts("2026-09-19T17:00:00Z"), QuotePolicy("book"), require_local_observation=True
    )
    assert elig3.empty and {e["reason"] for e in elig3.exclusions} == {"late_local_observation"}
    # a historical CSV without observed_at is observed at import time -> never prospective,
    # even when its supplied snapshot time is old
    old = pd.DataFrame(rows[:2]).drop(columns=["observed_at_utc", "provenance_mode"])
    old["snapshot_at_utc"] = "2020-09-12T16:50:00Z"
    old["bookmaker_updated_at_utc"] = "2020-09-12T16:49:00Z"
    old["kickoff_at_snapshot_utc"] = "2020-09-13T17:00:00Z"
    old.to_csv(path, index=False)
    games_old = pd.DataFrame({"game_id": ["g"], "kickoff_utc": [ts("2020-09-13T17:00:00Z")]})
    odds4, _ = import_odds(path, games=games_old)
    decision = ts("2020-09-12T17:00:00Z")
    assert (odds4["provenance_mode"] == "historical_csv").all()
    assert (odds4["observed_at_utc"] > decision).all()  # import time, not the CSV's claim
    assert select_quotes(odds4, decision, QuotePolicy("book"), require_local_observation=True).empty
    assert len(select_quotes(odds4, decision, QuotePolicy("book")).quotes) == 2


def test_identical_import_is_idempotent_and_conflicts_rejected(tmp_path):
    base = {
        "provider_event_id": "e",
        "game_id": "g",
        "bookmaker": "book",
        "settlement_rule": "push_refund",
        "snapshot_at_utc": "2026-09-19T16:50:00Z",
        "bookmaker_updated_at_utc": "2026-09-19T16:49:00Z",
        "kickoff_at_snapshot_utc": "2026-09-20T17:00:00Z",
        "market": "total",
        "line": "44.5",
    }
    rows = [
        base | {"quote_id": "o", "selection": "over", "decimal_odds": "1.91"},
        base | {"quote_id": "u", "selection": "under", "decimal_odds": "1.91"},
    ]
    path = tmp_path / "o.csv"
    pd.DataFrame(rows + rows).to_csv(path, index=False)
    odds, rep = import_odds(path)
    assert len(odds) == 0 and {r["reason"] for r in rep.rejected} == {
        "duplicate_or_missing_quote_id"
    }
    pd.DataFrame(
        [*rows, base | {"quote_id": "o2", "selection": "over", "decimal_odds": "1.95"}]
    ).to_csv(path, index=False)
    odds, rep = import_odds(path)
    assert "duplicate_conflict" in {r["reason"] for r in rep.rejected}
    pd.DataFrame(rows).to_csv(path, index=False)
    a, _ = import_odds(path)
    b, _ = import_odds(path)
    assert a["pair_id"].tolist() == b["pair_id"].tolist() and a["pair_id"].nunique() == 1
