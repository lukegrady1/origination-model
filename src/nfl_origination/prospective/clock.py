"""Injectable clock. Real commands use the system clock; tests inject a fixed/stepping clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol

import pandas as pd


class Clock(Protocol):
    def now(self) -> pd.Timestamp: ...


class SystemClock:
    def now(self) -> pd.Timestamp:
        return pd.Timestamp(datetime.now(UTC))


class FixedClock:
    """Test clock: returns a fixed time, optionally advancing by a step on each call."""

    def __init__(self, start: pd.Timestamp | str, step: timedelta | None = None) -> None:
        self._now = pd.Timestamp(start)
        if self._now.tzinfo is None:
            self._now = self._now.tz_localize("UTC")
        self._step = step

    def now(self) -> pd.Timestamp:
        current = self._now
        if self._step is not None:
            self._now = self._now + self._step
        return current

    def set(self, when: pd.Timestamp | str) -> None:
        self._now = pd.Timestamp(when)
        if self._now.tzinfo is None:
            self._now = self._now.tz_localize("UTC")

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta


def iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").isoformat().replace("+00:00", "Z")
