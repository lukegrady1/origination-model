from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nfl_origination.config import load_config, parse_overrides, parse_season_range
from nfl_origination.errors import InvalidInputError, MissingDataError, ModelValidationError
from nfl_origination.provenance import RunRegistry, hash_frame

ROOT = Path(__file__).resolve().parents[2]


def test_configs_load_and_unknown_key_fails(tmp_path):
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    assert cfg.run.kind == "demo" and cfg.data.source == "synthetic"
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["model"]["bogus"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(InvalidInputError):
        load_config(path)


def test_holdout_requires_selected_alpha(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "holdout.yaml").read_text())
    raw["model"]["selected_alpha"] = None
    path = tmp_path / "h.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(InvalidInputError):
        load_config(path)


def test_overrides_and_season_ranges():
    assert parse_season_range("2010:2012") == [2010, 2011, 2012]
    assert parse_season_range("2024") == [2024]
    with pytest.raises(InvalidInputError):
        parse_season_range("2025:2010")
    with pytest.raises(InvalidInputError):
        parse_season_range("abc")
    cfg = load_config(
        ROOT / "configs" / "demo.yaml",
        parse_overrides(["seed=7", "model.alpha_candidates=[1, 10]"]),
    )
    assert cfg.seed == 7 and cfg.model.alpha_candidates == (1.0, 10.0)
    with pytest.raises(InvalidInputError):
        parse_overrides(["novalue"])


def test_exit_codes():
    assert InvalidInputError.exit_code == 2
    assert MissingDataError.exit_code == 3
    assert ModelValidationError.exit_code == 4


def test_registry_resolution(tmp_path):
    reg = RunRegistry(tmp_path)
    with pytest.raises(InvalidInputError):
        reg.resolve("latest-holdout")
    with pytest.raises(InvalidInputError):
        reg.resolve("nope")


def test_hash_frame_is_content_based():
    import pandas as pd

    a = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
    assert hash_frame(a) == hash_frame(a.copy())
    assert hash_frame(a) != hash_frame(a.assign(x=[1, 3]))
