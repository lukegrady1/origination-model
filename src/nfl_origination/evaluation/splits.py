"""Chronological fold definitions (spec sections 9 and 11)."""

from __future__ import annotations

import pandas as pd

from nfl_origination.errors import ModelValidationError


def training_seasons(score_season: int, train_start: int) -> list[int]:
    return list(range(train_start, score_season))


def residual_seasons(score_season: int, residual_first: int, window: int) -> list[int]:
    """Most recent ``window`` completed residual seasons before ``score_season``."""
    first = max(residual_first, score_season - window)
    return list(range(first, score_season))


def assert_grouped_split(train: pd.DataFrame, evaluate: pd.DataFrame) -> None:
    """Both perspective rows of a game must stay together; no game may be in both sets."""
    overlap = set(train["game_id"]) & set(evaluate["game_id"])
    if overlap:
        raise ModelValidationError(f"{len(overlap)} games appear in both train and evaluation")
    for name, frame in (("train", train), ("evaluate", evaluate)):
        counts = frame.groupby("game_id")["perspective"].nunique()
        if len(frame) and (counts != 2).any():
            raise ModelValidationError(f"{name}: some games do not have both perspective rows")


def assert_labels_available(train: pd.DataFrame, fit_time: pd.Timestamp) -> None:
    """Training labels must have been available at the fit time."""
    if "label_available_utc" in train.columns and (train["label_available_utc"] > fit_time).any():
        late = int((train["label_available_utc"] > fit_time).sum())
        raise ModelValidationError(f"{late} training labels were not available at fit time")
