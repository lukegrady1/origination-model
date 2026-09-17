"""Canonical odds CSV import with strict validation (spec section 12)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.data.storage import write_parquet
from nfl_origination.errors import InvalidInputError
from nfl_origination.pricing.markets import MONEYLINE_RULES, line_to_half_points
from nfl_origination.provenance import iso_utc, utc_now
from nfl_origination.schemas import ODDS_SCHEMA, validate_frame

ODDS_CSV_COLUMNS = [
    "quote_id",
    "provider_event_id",
    "game_id",
    "bookmaker",
    "market",
    "selection",
    "line",
    "decimal_odds",
    "snapshot_at_utc",
    "bookmaker_updated_at_utc",
    "kickoff_at_snapshot_utc",
    "settlement_rule",
]
# V2 optional columns; absent columns get explicit defaults (never silently "live").
ODDS_CSV_OPTIONAL = [
    "provider",
    "receipt_id",
    "raw_hash",
    "observed_at_utc",
    "provenance_mode",
    "market_role",
]
PROVENANCE_MODES = {"historical_csv", "live_collected", "synthetic"}
MARKET_ROLES = {"main", "alternate"}
MARKET_SELECTIONS = {
    "moneyline": {"home", "away"},
    "spread": {"home", "away"},
    "total": {"over", "under"},
}
KICKOFF_JOIN_TOLERANCE = pd.Timedelta(hours=36)


class OddsImportReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    output_path: str | None
    rows_in: int
    rows_accepted: int
    synthetic: bool
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    ingested_at_utc: str

    def summary(self) -> dict[str, Any]:
        reasons: dict[str, int] = {}
        for r in self.rejected:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        return {"rows_in": self.rows_in, "rows_accepted": self.rows_accepted, "rejected": reasons}


def pair_ids(odds: pd.DataFrame) -> pd.Series:
    """Stable pair identity: book, game, market, unordered line pair, snapshot, role.

    Both sides of a complete pair share the id; a spread pair is identified by |line|.
    """
    keys = []
    for r in odds.itertuples(index=False):
        line = "" if pd.isna(r.line) else f"{abs(float(r.line)):g}"
        raw = "|".join(
            [
                str(r.bookmaker),
                str(r.game_id),
                str(r.market),
                line,
                pd.Timestamp(r.snapshot_at_utc).isoformat(),
                str(getattr(r, "market_role", "main")),
            ]
        )
        keys.append(hashlib.sha256(raw.encode()).hexdigest()[:20])
    return pd.Series(keys, index=odds.index, dtype="object")


def _parse_utc(series: pd.Series) -> pd.Series:
    """Parse ISO timestamps; values without an explicit offset are rejected (NaT)."""
    text = series.astype("string")
    has_tz = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    parsed = pd.to_datetime(text.where(has_tz), utc=True, errors="coerce", format="ISO8601")
    return parsed


def import_odds(
    path: Path,
    *,
    games: pd.DataFrame | None = None,
    crosswalk: pd.DataFrame | None = None,
    synthetic: bool = False,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, OddsImportReport]:
    """Validate an odds CSV row by row; accepted quotes go to Parquet, rejects get reasons."""
    path = Path(path)
    if not path.exists():
        raise InvalidInputError(f"odds file not found: {path}")
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ODDS_CSV_COLUMNS if c not in raw.columns]
    if missing:
        raise InvalidInputError(f"odds CSV is missing required columns {missing}")
    ingested = utc_now()
    rejected: list[dict[str, Any]] = []
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    for col in ODDS_CSV_OPTIONAL:
        if col not in raw.columns:
            raw[col] = ""
    df = raw[ODDS_CSV_COLUMNS + ODDS_CSV_OPTIONAL].copy()
    # A CSV import is a *local* observation at import time unless the row carries its own
    # collector observation time (live adapter output). Supplied snapshot times are preserved
    # separately; a late import can never qualify as a prospective decision.
    df["provider"] = df["provider"].replace("", "csv")
    df["provenance_mode"] = df["provenance_mode"].replace(
        "", "synthetic" if synthetic else "historical_csv"
    )
    df["market_role"] = df["market_role"].replace("", "main")
    df["raw_hash"] = df["raw_hash"].replace("", file_hash)
    parsed_observed = _parse_utc(df["observed_at_utc"]).astype("datetime64[us, UTC]")
    import_time = pd.Timestamp(ingested).as_unit("us")
    df["observed_at_parsed"] = parsed_observed.where(df["observed_at_utc"] != "", import_time)
    for c in ("snapshot_at_utc", "bookmaker_updated_at_utc", "kickoff_at_snapshot_utc"):
        df[c] = _parse_utc(df[c])
    df["line_num"] = pd.to_numeric(df["line"].replace("", np.nan), errors="coerce")
    df["decimal_num"] = pd.to_numeric(df["decimal_odds"], errors="coerce")

    def reject(idx: int, reason: str, detail: str = "") -> None:
        rejected.append({"quote_id": df.at[idx, "quote_id"], "reason": reason, "detail": detail})

    keep = np.ones(len(df), dtype=bool)
    dup_ids = df["quote_id"].duplicated(keep=False)
    for idx in df.index:
        row = df.loc[idx]
        if row["quote_id"] == "" or dup_ids[idx]:
            reject(idx, "duplicate_or_missing_quote_id")
            keep[idx] = False
            continue
        if row["market"] not in MARKET_SELECTIONS:
            reject(idx, "unknown_market", row["market"])
            keep[idx] = False
            continue
        if row["selection"] not in MARKET_SELECTIONS[row["market"]]:
            reject(idx, "invalid_selection", f"{row['market']}:{row['selection']}")
            keep[idx] = False
            continue
        if row["market"] == "moneyline":
            if row["line"] not in ("", "null", "NULL", "None"):
                reject(idx, "moneyline_has_line")
                keep[idx] = False
                continue
        else:
            if pd.isna(row["line_num"]):
                reject(idx, "missing_line")
                keep[idx] = False
                continue
            try:
                line_to_half_points(float(row["line_num"]))
            except InvalidInputError as exc:
                reject(idx, "invalid_line_grid", str(exc))
                keep[idx] = False
                continue
        d = row["decimal_num"]
        if pd.isna(d) or not np.isfinite(d) or d <= 1.0:
            reject(idx, "invalid_decimal_odds", row["decimal_odds"])
            keep[idx] = False
            continue
        if pd.isna(row["snapshot_at_utc"]):
            reject(idx, "invalid_snapshot_timestamp", "must be ISO-8601 with explicit UTC offset")
            keep[idx] = False
            continue
        if pd.isna(row["kickoff_at_snapshot_utc"]):
            reject(idx, "invalid_kickoff_timestamp")
            keep[idx] = False
            continue
        if pd.notna(row["bookmaker_updated_at_utc"]) and (
            row["bookmaker_updated_at_utc"] > row["snapshot_at_utc"]
        ):
            reject(idx, "update_after_snapshot")
            keep[idx] = False
            continue
        if row["bookmaker"] == "" or row["provider_event_id"] == "":
            reject(idx, "missing_book_or_event_id")
            keep[idx] = False
            continue
        if row["market"] == "moneyline" and row["settlement_rule"] not in MONEYLINE_RULES:
            reject(idx, "unsupported_moneyline_rule", row["settlement_rule"])
            keep[idx] = False
            continue
        if row["provenance_mode"] not in PROVENANCE_MODES:
            reject(idx, "unknown_provenance_mode", row["provenance_mode"])
            keep[idx] = False
            continue
        if row["market_role"] not in MARKET_ROLES:
            reject(idx, "unknown_market_role", row["market_role"])
            keep[idx] = False
            continue
        if row["observed_at_utc"] != "" and pd.isna(row["observed_at_parsed"]):
            reject(idx, "invalid_observed_timestamp")
            keep[idx] = False
            continue
        if games is not None:
            gid = row["game_id"]
            match = games[games["game_id"] == gid] if gid else games.iloc[0:0]
            if match.empty:
                reject(idx, "ambiguous_event_join", f"game_id {gid!r} not found")
                keep[idx] = False
                continue
            kick = pd.Timestamp(match["kickoff_utc"].iloc[0])
            if abs(kick - row["kickoff_at_snapshot_utc"]) > KICKOFF_JOIN_TOLERANCE:
                xw = (
                    crosswalk
                    if crosswalk is not None
                    else pd.DataFrame(columns=["provider_event_id", "game_id"])
                )
                ok = (
                    (xw["provider_event_id"] == row["provider_event_id"]) & (xw["game_id"] == gid)
                ).any()
                if not ok:
                    reject(
                        idx,
                        "ambiguous_event_join",
                        "kickoff differs from schedule by more than 36h and no crosswalk entry",
                    )
                    keep[idx] = False
                    continue
    accepted = df[keep].copy()
    # duplicate conflict: same book/market/selection/line/snapshot with different prices
    key = ["bookmaker", "game_id", "market", "selection", "line_num", "snapshot_at_utc"]
    grp = accepted.groupby(key, dropna=False)["decimal_num"].transform("nunique")
    conflict = grp > 1
    for idx in accepted[conflict].index:
        reject(int(idx), "duplicate_conflict", "same quote key with different prices")
    accepted = accepted[~conflict]
    exact_dup = accepted.duplicated(subset=[*key, "decimal_num"], keep="first")
    for idx in accepted[exact_dup].index:
        reject(int(idx), "duplicate_exact", "identical quote repeated")
    accepted = accepted[~exact_dup]

    odds = pd.DataFrame(
        {
            "quote_id": accepted["quote_id"].astype(str),
            "provider_event_id": accepted["provider_event_id"].astype(str),
            "game_id": accepted["game_id"].astype(str),
            "bookmaker": accepted["bookmaker"].astype(str),
            "market": accepted["market"].astype(str),
            "selection": accepted["selection"].astype(str),
            "line": accepted["line_num"].astype(float),
            "decimal_odds": accepted["decimal_num"].astype(float),
            "snapshot_at_utc": accepted["snapshot_at_utc"],
            "bookmaker_updated_at_utc": accepted["bookmaker_updated_at_utc"],
            "kickoff_at_snapshot_utc": accepted["kickoff_at_snapshot_utc"],
            "settlement_rule": accepted["settlement_rule"].astype(str),
            "ingested_at_utc": pd.Timestamp(ingested),
            "synthetic": bool(synthetic),
            "provider": accepted["provider"].astype(str),
            "receipt_id": accepted["receipt_id"].astype(str),
            "raw_hash": accepted["raw_hash"].astype(str),
            "observed_at_utc": accepted["observed_at_parsed"],
            "provenance_mode": accepted["provenance_mode"].astype(str),
            "market_role": accepted["market_role"].astype(str),
        }
    ).reset_index(drop=True)
    for c in (
        "snapshot_at_utc",
        "bookmaker_updated_at_utc",
        "kickoff_at_snapshot_utc",
        "observed_at_utc",
    ):
        odds[c] = pd.to_datetime(odds[c], utc=True)
    odds["pair_id"] = pair_ids(odds)
    validate_frame(odds, ODDS_SCHEMA)
    out = None
    if output_path is not None:
        write_parquet(odds, output_path)
        out = str(output_path)
    report = OddsImportReport(
        source_path=str(path),
        output_path=out,
        rows_in=len(raw),
        rows_accepted=len(odds),
        synthetic=synthetic,
        rejected=rejected,
        ingested_at_utc=iso_utc(ingested) or "",
    )
    return odds, report
