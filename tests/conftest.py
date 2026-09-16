from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nfl_origination.config import DistributionConfig, FeaturesConfig  # noqa: E402
from nfl_origination.features.asof import AsOfPolicy  # noqa: E402
from nfl_origination.models.distribution import (  # noqa: E402
    ScoreDistribution,
    score_distribution_from_normal,
)
from nfl_origination.synthetic import generate_synthetic_dataset  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    """CI sets NFL_ORIGINATION_BLOCK_NETWORK=1: any real HTTP attempt fails the test.

    `responses`-mocked tests replace the adapter inside the test body, so they still work.
    """
    if os.environ.get("NFL_ORIGINATION_BLOCK_NETWORK") == "1":
        import requests.adapters

        def _blocked(self, request, *args, **kwargs):
            raise AssertionError(f"network access blocked in tests: {request.url}")

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _blocked)
    yield


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def synthetic_small():
    """Small deterministic synthetic dataset: 2010–2018 (enough for 500+ residual games)."""
    return generate_synthetic_dataset(seed=3, seasons=(2010, 2018))


@pytest.fixture
def policy() -> AsOfPolicy:
    return AsOfPolicy()


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig()


@pytest.fixture
def dist_cfg() -> DistributionConfig:
    return DistributionConfig()


@pytest.fixture
def normal_dist(dist_cfg) -> ScoreDistribution:
    mu = np.array([24.3, 20.1])
    sigma = np.array([[110.0, 15.0], [15.0, 95.0]])
    return score_distribution_from_normal(mu, sigma, dist_cfg)


def point_mass(home: int, away: int, max_score: int = 100) -> ScoreDistribution:
    pmf = np.zeros((max_score + 1, max_score + 1))
    pmf[home, away] = 1.0
    return ScoreDistribution.from_pmf(pmf)


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")
