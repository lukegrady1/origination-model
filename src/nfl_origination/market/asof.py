"""Timestamp-safe quote selection at a decision time (spec section 12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nfl_origination.pricing.markets import MONEYLINE_RULES


@dataclass(frozen=True)
class QuotePolicy:
    bookmaker: str
    max_age_minutes: float = 60.0
    require_update_timestamp: bool = True
    moneyline_settlement_rule: str = "two_way_tie_void"


@dataclass
class EligibleQuotes:
    quotes: pd.DataFrame  # complete, paired quotes with a `snapshot_at_utc` per market
    exclusions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.quotes.empty


def _pair_complete(group: pd.DataFrame) -> bool:
    market = group["market"].iloc[0]
    if len(group) != 2 or group["selection"].nunique() != 2:
        return False
    if market == "spread":
        lines = group.sort_values("selection")["line"].to_numpy(float)
        return bool(abs(lines[0] + lines[1]) < 1e-9)
    if market == "total":
        return bool(group["line"].nunique() == 1)
    return True


def _main_pair(group: pd.DataFrame) -> tuple[pd.DataFrame | None, list[tuple[pd.DataFrame, str]]]:
    """The one complete main-market pair at a snapshot; alternates are quarantined.

    Alternate lines never make a valid main pair look incomplete, and two candidate main pairs
    at the same snapshot are ambiguous rather than guessed.
    """
    excluded: list[tuple[pd.DataFrame, str]] = []
    role = (
        group["market_role"]
        if "market_role" in group.columns
        else pd.Series("main", index=group.index)
    )
    alternates = group[role != "main"]
    if len(alternates):
        excluded.append((alternates, "alternate_line_quarantined"))
    main = group[role == "main"]
    if main.empty:
        return None, excluded
    market = main["market"].iloc[0]
    if market == "spread":
        key = main["line"].abs().round(6)
    elif market == "total":
        key = main["line"].round(6)
    else:
        key = pd.Series(0.0, index=main.index)
    complete = [part for _k, part in main.groupby(key) if _pair_complete(part)]
    leftovers = main[~main.index.isin(pd.concat(complete).index)] if complete else main
    if len(leftovers):
        excluded.append((leftovers, "incomplete_pair_at_snapshot"))
    if len(complete) > 1:
        excluded.append((pd.concat(complete), "ambiguous_main_line"))
        return None, excluded
    if not complete:
        return None, excluded
    return complete[0], excluded


def select_quotes(
    odds: pd.DataFrame,
    decision_time: pd.Timestamp,
    policy: QuotePolicy,
    *,
    game_id: str | None = None,
    require_local_observation: bool = False,
) -> EligibleQuotes:
    """Newest complete snapshot per (game, market) from the configured book at decision time.

    Eligibility: snapshot_at <= decision_time, bookmaker_updated_at <= snapshot_at, both no older
    than ``max_age_minutes`` before the decision time, matching book and supported rules, and
    both sides of the market present at the same snapshot (opposite spread lines / equal totals).
    """
    exclusions: list[dict[str, Any]] = []
    df = odds if game_id is None else odds[odds["game_id"] == game_id]
    if df.empty:
        return EligibleQuotes(df.iloc[0:0], exclusions)
    oldest = decision_time - pd.Timedelta(minutes=policy.max_age_minutes)

    def excl(rows: pd.DataFrame, reason: str) -> None:
        exclusions.extend({"quote_id": q, "reason": reason} for q in rows["quote_id"])

    wrong_book = df["bookmaker"] != policy.bookmaker
    excl(df[wrong_book], "wrong_bookmaker")
    df = df[~wrong_book]
    if require_local_observation:
        # prospective use: the quote must have been observed locally by the decision time
        if "observed_at_utc" not in df.columns:
            excl(df, "late_local_observation")
            df = df.iloc[0:0]
        else:
            late = df["observed_at_utc"].isna() | (df["observed_at_utc"] > decision_time)
            excl(df[late], "late_local_observation")
            df = df[~late]
    future = df["snapshot_at_utc"] > decision_time
    excl(df[future], "snapshot_after_decision_time")
    df = df[~future]
    stale = df["snapshot_at_utc"] < oldest
    excl(df[stale], "stale_snapshot")
    df = df[~stale]
    if policy.require_update_timestamp:
        missing = df["bookmaker_updated_at_utc"].isna()
        excl(df[missing], "missing_update_timestamp")
        df = df[~missing]
        late = df["bookmaker_updated_at_utc"] > df["snapshot_at_utc"]
        excl(df[late], "update_after_snapshot")
        df = df[~late]
        old_update = df["bookmaker_updated_at_utc"] < oldest
        excl(df[old_update], "stale_update_timestamp")
        df = df[~old_update]
    bad_rule = (df["market"] == "moneyline") & (
        (df["settlement_rule"] != policy.moneyline_settlement_rule)
        | ~df["settlement_rule"].isin(MONEYLINE_RULES)
    )
    excl(df[bad_rule], "unsupported_settlement_rule")
    df = df[~bad_rule]
    if df.empty:
        return EligibleQuotes(df, exclusions)
    chosen = []
    for (_gid, _market), group in df.groupby(["game_id", "market"], sort=True):
        snapshots = sorted(group["snapshot_at_utc"].unique(), reverse=True)
        picked = None
        for snap in snapshots:
            cand = group[group["snapshot_at_utc"] == snap]
            pair, excluded = _main_pair(cand)
            for rows_, reason in excluded:
                excl(rows_, reason)
            if pair is not None:
                picked = pair
                break
        if picked is not None:
            chosen.append(picked)
            older = group[group["snapshot_at_utc"] < picked["snapshot_at_utc"].iloc[0]]
            excl(older, "superseded_by_newer_snapshot")
    quotes = pd.concat(chosen, ignore_index=True) if chosen else df.iloc[0:0]
    return EligibleQuotes(quotes, exclusions)


def closing_proxy(
    odds: pd.DataFrame,
    kickoff: pd.Timestamp,
    policy: QuotePolicy,
    *,
    game_id: str,
    window_minutes: float = 30.0,
    require_local_observation: bool = False,
) -> pd.DataFrame:
    """Latest complete quote from the same book in the final window strictly before kickoff.

    This is an observed closing proxy, not a guaranteed final close.
    """
    df = odds[(odds["game_id"] == game_id) & (odds["bookmaker"] == policy.bookmaker)]
    df = df[
        (df["snapshot_at_utc"] < kickoff)
        & (df["snapshot_at_utc"] >= kickoff - pd.Timedelta(minutes=window_minutes))
    ]
    # validity and freshness apply to the close too: nonmissing update <= snapshot, both
    # within the policy age, local observation strictly before kickoff when recorded
    if policy.require_update_timestamp and len(df):
        oldest = kickoff - pd.Timedelta(minutes=policy.max_age_minutes)
        ok = (
            df["bookmaker_updated_at_utc"].notna()
            & (df["bookmaker_updated_at_utc"] <= df["snapshot_at_utc"])
            & (df["bookmaker_updated_at_utc"] >= oldest)
        )
        df = df[ok]
    if require_local_observation and len(df):
        if "observed_at_utc" not in df.columns:
            return df.iloc[0:0]
        df = df[df["observed_at_utc"].notna() & (df["observed_at_utc"] < kickoff)]
    bad_rule = (df["market"] == "moneyline") & ~df["settlement_rule"].isin(MONEYLINE_RULES)
    df = df[~bad_rule]
    if df.empty:
        return df
    chosen = []
    for _market, group in df.groupby("market"):
        for snap in sorted(group["snapshot_at_utc"].unique(), reverse=True):
            cand = group[group["snapshot_at_utc"] == snap]
            pair, _excluded = _main_pair(cand)
            if pair is not None:
                chosen.append(pair)
                break
    return pd.concat(chosen, ignore_index=True) if chosen else df.iloc[0:0]
