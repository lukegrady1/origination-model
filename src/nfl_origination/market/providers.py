"""Optional live odds provider adapter (V2 R2): The Odds API v4, mocked in tests.

One bounded collection per call, never a polling loop. The API key comes only from the
``ODDS_API_KEY`` environment variable and is never written to receipts, manifests, errors or
logs. Every response is persisted as an immutable receipt (raw bytes + timing + non-secret
request parameters + quota headers) before it is parsed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.data.snapshots import redact_url, writer_lock
from nfl_origination.errors import InvalidInputError
from nfl_origination.provenance import iso_utc, read_json, utc_now, write_json

ADAPTER_VERSION = "the-odds-api-v4/1"
API_KEY_ENV = "ODDS_API_KEY"
DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
QUOTA_HEADERS = ("x-requests-remaining", "x-requests-used", "x-requests-last")
CollectionStatus = Literal[
    "ok",
    "timeout",
    "http_error",
    "invalid_json",
    "empty_payload",
    "missing_book",
    "auth_failed",
    "quota_exhausted",
    "rate_limited",
    "quota_reserve",
    "budget_exhausted",
    "no_api_key",
]
PROVIDER_MARKETS = {"h2h": "moneyline", "spreads": "spread", "totals": "total"}

# Provider team names -> canonical franchise IDs (current and recent historical names).
TEAM_NAME_TO_ID: dict[str, str] = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Football Team": "WAS",
    "Washington Redskins": "WAS",
}


class OddsReceipt(BaseModel):
    """Immutable record of one provider response (or failure)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str
    provider: str
    status: CollectionStatus
    request_started_at_utc: str
    observed_at_utc: str | None
    persisted_at_utc: str
    request_params_redacted: dict[str, Any]
    endpoint_redacted: str
    http_status: int | None
    content_sha256: str | None
    blob_path: str | None
    bytes: int
    quota: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    synthetic: bool = False
    adapter_version: str = ADAPTER_VERSION


@dataclass
class CollectionResult:
    receipt: OddsReceipt
    quotes: pd.DataFrame
    quarantined: list[dict[str, Any]] = field(default_factory=list)
    events_seen: int = 0


class OddsProvider(Protocol):
    def collect(
        self,
        *,
        bookmakers: tuple[str, ...],
        markets: tuple[str, ...],
        games: pd.DataFrame,
        settlement_rules: dict[str, dict[str, str]],
        synthetic: bool,
    ) -> CollectionResult: ...


class CollectionBudget:
    """Bounded requests per invocation plus a quota reserve from the last known headers."""

    def __init__(self, odds_dir: Path, max_requests: int, quota_reserve: int) -> None:
        self.odds_dir = Path(odds_dir)
        self.max_requests = max_requests
        self.quota_reserve = quota_reserve
        self.used = 0

    def last_quota(self) -> dict[str, str]:
        path = self.odds_dir / "quota.json"
        return dict(read_json(path)) if path.exists() else {}

    def check(self) -> CollectionStatus | None:
        if self.used >= self.max_requests:
            return "budget_exhausted"
        remaining = self.last_quota().get("x-requests-remaining")
        if remaining is not None:
            try:
                if int(float(remaining)) <= self.quota_reserve:
                    return "quota_reserve"
            except ValueError:
                pass
        return None

    def record(self, quota: dict[str, str]) -> None:
        self.used += 1
        if quota:
            write_json(self.odds_dir / "quota.json", quota)


def _receipt_id(started: str, params: dict[str, Any]) -> str:
    raw = json.dumps({"started": started, "params": params}, sort_keys=True, default=str)
    return "odds-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class TheOddsApiAdapter:
    """The Odds API v4 ``/sports/{sport}/odds`` endpoint with decimal prices."""

    provider = "the_odds_api"

    def __init__(
        self,
        odds_dir: Path,
        *,
        sport_key: str = "americanfootball_nfl",
        regions: str = "us",
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        budget: CollectionBudget | None = None,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.clock = clock or utc_now
        self.odds_dir = Path(odds_dir)
        self.sport_key = sport_key
        self.regions = regions
        self.timeout = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.budget = budget or CollectionBudget(self.odds_dir, 4, 50)
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ persistence
    def _persist(
        self,
        *,
        status: CollectionStatus,
        started: str,
        observed: str | None,
        params: dict[str, Any],
        endpoint: str,
        http_status: int | None,
        body: bytes | None,
        quota: dict[str, str],
        error: str | None,
        synthetic: bool,
    ) -> OddsReceipt:
        digest = hashlib.sha256(body).hexdigest() if body else None
        blob_path = None
        if body:
            blob_dir = self.odds_dir / "blobs"
            blob_dir.mkdir(parents=True, exist_ok=True)
            blob_path = blob_dir / f"{digest}.json"
            if not blob_path.exists():
                tmp = blob_dir / f".{digest}.{os.getpid()}.tmp"
                tmp.write_bytes(body)
                os.replace(tmp, blob_path)
        receipt = OddsReceipt(
            receipt_id=_receipt_id(started, params),
            provider=self.provider,
            status=status,
            request_started_at_utc=started,
            observed_at_utc=observed,
            persisted_at_utc=iso_utc(self.clock()) or "",
            request_params_redacted=params,
            endpoint_redacted=redact_url(endpoint),
            http_status=http_status,
            content_sha256=digest,
            blob_path=None if blob_path is None else str(blob_path),
            bytes=len(body) if body else 0,
            quota=quota,
            error=None if error is None else redact_url(error),
            synthetic=synthetic,
        )
        rdir = self.odds_dir / "receipts"
        rdir.mkdir(parents=True, exist_ok=True)
        with writer_lock(self.odds_dir):
            path = rdir / f"{receipt.receipt_id}.json"
            if not path.exists():
                tmp = rdir / f".{receipt.receipt_id}.{os.getpid()}.tmp"
                tmp.write_text(receipt.model_dump_json(indent=1))
                os.replace(tmp, path)
        return receipt

    # ------------------------------------------------------------------ collection
    def collect(
        self,
        *,
        bookmakers: tuple[str, ...],
        markets: tuple[str, ...] = ("h2h", "spreads", "totals"),
        games: pd.DataFrame,
        settlement_rules: dict[str, dict[str, str]],
        synthetic: bool = False,
    ) -> CollectionResult:
        if not bookmakers:
            raise InvalidInputError("collect requires an explicit bookmaker list")
        params: dict[str, Any] = {
            "sport": self.sport_key,
            "regions": self.regions,
            "markets": ",".join(markets),
            "bookmakers": ",".join(bookmakers),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        endpoint = f"{self.base_url}/sports/{self.sport_key}/odds"
        started = iso_utc(self.clock()) or ""
        empty = pd.DataFrame()
        blocked = self.budget.check()
        if blocked is not None:
            receipt = self._persist(
                status=blocked,
                started=started,
                observed=None,
                params=params,
                endpoint=endpoint,
                http_status=None,
                body=None,
                quota=self.budget.last_quota(),
                error=f"collection skipped: {blocked}",
                synthetic=synthetic,
            )
            return CollectionResult(receipt, empty)
        api_key = os.environ.get(API_KEY_ENV)
        if not api_key:
            receipt = self._persist(
                status="no_api_key",
                started=started,
                observed=None,
                params=params,
                endpoint=endpoint,
                http_status=None,
                body=None,
                quota={},
                error=f"{API_KEY_ENV} is not set; no synthetic fallback in real mode",
                synthetic=synthetic,
            )
            return CollectionResult(receipt, empty)
        query = {**{k: v for k, v in params.items() if k != "sport"}, "apiKey": api_key}
        status: CollectionStatus = "http_error"
        http_status: int | None = None
        body: bytes | None = None
        quota: dict[str, str] = {}
        error: str | None = None
        observed: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.session.get(endpoint, params=query, timeout=self.timeout)
            except requests.Timeout:
                status, error = "timeout", f"timeout after {self.timeout}s (attempt {attempt})"
                if attempt < self.max_attempts:
                    time.sleep(0.05 * attempt)
                continue
            except requests.RequestException as exc:
                status, error = "http_error", f"request failed: {exc.__class__.__name__}"
                if attempt < self.max_attempts:
                    time.sleep(0.05 * attempt)
                continue
            observed = iso_utc(self.clock())
            http_status = resp.status_code
            quota = {h: resp.headers[h] for h in QUOTA_HEADERS if h in resp.headers}
            body = resp.content
            if resp.status_code == 401:
                text = resp.text.upper()
                status = "quota_exhausted" if "USAGE" in text or "QUOTA" in text else "auth_failed"
                error = f"HTTP 401: {resp.text[:200]}"
                break
            if resp.status_code == 429:
                status, error = "rate_limited", f"HTTP 429: {resp.text[:200]}"
                break
            if resp.status_code != 200:
                status, error = "http_error", f"HTTP {resp.status_code}: {resp.text[:200]}"
                if 500 <= resp.status_code < 600 and attempt < self.max_attempts:
                    time.sleep(0.05 * attempt)
                    continue
                break
            try:
                payload = json.loads(body.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                status, error = "invalid_json", "response body is not valid JSON"
                break
            if not isinstance(payload, list) or not payload:
                status, error = "empty_payload", "provider returned no events"
                break
            status, error = "ok", None
            break
        self.budget.record(quota)
        receipt = self._persist(
            status=status,
            started=started,
            observed=observed,
            params=params,
            endpoint=endpoint,
            http_status=http_status,
            body=body,
            quota=quota,
            error=error,
            synthetic=synthetic,
        )
        if status != "ok" or body is None or observed is None:
            return CollectionResult(receipt, empty)
        payload = json.loads(body.decode("utf-8"))
        quotes, quarantined = parse_odds_payload(
            payload,
            receipt=receipt,
            observed_at_utc=observed,
            bookmakers=bookmakers,
            games=games,
            settlement_rules=settlement_rules,
            synthetic=synthetic,
            odds_dir=self.odds_dir,
        )
        if quotes.empty and any(q["reason"] == "missing_book" for q in quarantined):
            receipt = receipt.model_copy(
                update={
                    "status": "missing_book",
                    "error": "configured bookmakers absent from every event",
                }
            )
        return CollectionResult(receipt, quotes, quarantined, events_seen=len(payload))


def _match_event(event: dict[str, Any], games: pd.DataFrame) -> tuple[str | None, str]:
    """Map a provider event to a schedule game: teams plus kickoff proximity, never names alone."""
    home = TEAM_NAME_TO_ID.get(str(event.get("home_team", "")))
    away = TEAM_NAME_TO_ID.get(str(event.get("away_team", "")))
    if home is None or away is None:
        return None, "unknown_team_name"
    try:
        commence = pd.Timestamp(event["commence_time"])
    except (KeyError, ValueError):
        return None, "missing_commence_time"
    if commence.tzinfo is None:
        commence = commence.tz_localize("UTC")
    cand = games[(games["home_team"] == home) & (games["away_team"] == away)]
    if cand.empty:
        return None, "no_schedule_match"
    delta = (cand["kickoff_utc"] - commence).abs()
    close = cand[delta <= pd.Timedelta(hours=36)]
    if len(close) == 1:
        return str(close["game_id"].iloc[0]), "matched"
    if len(close) > 1:
        return None, "ambiguous_event_join"
    return None, "kickoff_mismatch"


def parse_odds_payload(
    payload: list[dict[str, Any]],
    *,
    receipt: OddsReceipt,
    observed_at_utc: str,
    bookmakers: tuple[str, ...],
    games: pd.DataFrame,
    settlement_rules: dict[str, dict[str, str]],
    synthetic: bool,
    odds_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Canonical quotes from a v4 odds payload; unmatched/ambiguous items are quarantined."""
    rows: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    crosswalk: list[dict[str, Any]] = []
    saw_book = False
    for event in payload:
        event_id = str(event.get("id", ""))
        game_id, reason = _match_event(event, games)
        crosswalk.append(
            {
                "provider_event_id": event_id,
                "game_id": game_id,
                "status": reason,
                "observed_at_utc": observed_at_utc,
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
                "commence_time": event.get("commence_time"),
            }
        )
        if game_id is None:
            quarantined.append({"provider_event_id": event_id, "reason": reason})
            continue
        kickoff = pd.Timestamp(event["commence_time"])
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize("UTC")
        home_name, away_name = event.get("home_team"), event.get("away_team")
        for book in event.get("bookmakers", []):
            key = str(book.get("key"))
            if key not in bookmakers:
                continue
            saw_book = True
            rules = settlement_rules.get(key, {})
            for market in book.get("markets", []):
                mkey = str(market.get("key"))
                canonical = PROVIDER_MARKETS.get(mkey)
                if canonical is None:
                    quarantined.append(
                        {"provider_event_id": event_id, "reason": f"unsupported_market:{mkey}"}
                    )
                    continue
                rule = rules.get(canonical)
                if rule is None:
                    quarantined.append(
                        {
                            "provider_event_id": event_id,
                            "reason": f"no_settlement_rule:{key}:{canonical}",
                        }
                    )
                    continue
                updated = market.get("last_update") or book.get("last_update")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    if canonical == "total":
                        selection = (
                            "over"
                            if str(name).lower() == "over"
                            else "under"
                            if str(name).lower() == "under"
                            else None
                        )
                    else:
                        selection = (
                            "home" if name == home_name else "away" if name == away_name else None
                        )
                    if selection is None:
                        quarantined.append(
                            {"provider_event_id": event_id, "reason": f"unknown_outcome:{name}"}
                        )
                        continue
                    line = outcome.get("point")
                    if canonical == "moneyline":
                        line = None
                    price = outcome.get("price")
                    stable = "|".join(
                        [
                            receipt.receipt_id,
                            key,
                            canonical,
                            selection,
                            "" if line is None else str(line),
                            str(price),
                            str(updated),
                        ]
                    )
                    rows.append(
                        {
                            "quote_id": "q-" + hashlib.sha256(stable.encode()).hexdigest()[:20],
                            "provider_event_id": event_id,
                            "game_id": game_id,
                            "bookmaker": key,
                            "market": canonical,
                            "selection": selection,
                            "line": "" if line is None else line,
                            "decimal_odds": price,
                            "snapshot_at_utc": observed_at_utc,
                            "bookmaker_updated_at_utc": updated or "",
                            "kickoff_at_snapshot_utc": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "settlement_rule": rule,
                            "provider": "the_odds_api",
                            "receipt_id": receipt.receipt_id,
                            "raw_hash": receipt.content_sha256 or "",
                            "observed_at_utc": observed_at_utc,
                            "provenance_mode": "synthetic" if synthetic else "live_collected",
                            "market_role": "main",
                        }
                    )
    if not saw_book and payload:
        quarantined.append({"provider_event_id": None, "reason": "missing_book"})
    if odds_dir is not None and crosswalk:
        path = Path(odds_dir) / "event_crosswalk.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            for rec in crosswalk:
                fh.write(json.dumps(rec, default=str) + "\n")
    return pd.DataFrame(rows), quarantined
