"""As-of availability policy (spec section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_origination.config import DataMode
from nfl_origination.errors import ModelValidationError

HOUR = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class AsOfPolicy:
    mode: DataMode = "historical_reconstruction"
    completed_game_lag_hours: float = 48.0
    cutoff_hours_before_kickoff: float = 24.0

    @property
    def forecast_policy_label(self) -> str:
        return f"kickoff_minus_{self.cutoff_hours_before_kickoff:g}h"

    def cutoff_for(self, kickoff_utc: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
        return kickoff_utc - self.cutoff_hours_before_kickoff * HOUR

    def eligible_from(self, kickoff_utc: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
        """Earliest cutoff at which a completed source game may be used."""
        return kickoff_utc + self.completed_game_lag_hours * HOUR

    def eligible_mask(
        self,
        source_kickoff_utc: pd.Series,
        cutoff_utc: pd.Timestamp,
        first_observed_utc: pd.Series | None = None,
    ) -> np.ndarray:
        """Source games usable at ``cutoff_utc`` under this policy."""
        ok = (source_kickoff_utc + self.completed_game_lag_hours * HOUR) <= cutoff_utc
        mask = ok.to_numpy(dtype=bool)
        if self.mode == "recorded_asof":
            if first_observed_utc is None:
                return np.zeros(len(mask), dtype=bool)
            observed_at = _observation_series(first_observed_utc)
            observed = observed_at <= cutoff_utc  # NaT compares False: unknown is ineligible
            mask = mask & observed.fillna(False).to_numpy(dtype=bool)
        return mask


def _observation_series(series: pd.Series) -> pd.Series:
    """Observation times must be tz-aware UTC or entirely unknown; anything else is an error."""
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return series.dt.tz_convert("UTC")
    if series.isna().all():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    if pd.api.types.is_datetime64_any_dtype(series):
        raise ModelValidationError("first_observed_utc is a naive datetime; UTC required")
    try:
        return pd.to_datetime(series, utc=True, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ModelValidationError("first_observed_utc has unparseable timestamps") from exc
