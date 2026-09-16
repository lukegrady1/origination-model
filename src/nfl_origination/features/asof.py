"""As-of availability policy (spec section 6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_origination.config import DataMode

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
            observed = first_observed_utc <= cutoff_utc
            mask = mask & observed.fillna(False).to_numpy(dtype=bool)
        return mask
