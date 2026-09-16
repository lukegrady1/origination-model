"""Dashboard smoke test with Streamlit's AppTest: every page renders, empty panels don't crash."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from streamlit.testing.v1 import AppTest

from nfl_origination.config import load_config
from nfl_origination.experiment import run_backtest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "nfl_origination" / "dashboard" / "app.py"


def _run_small(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    raw["run"] = {
        "kind": "development",
        "label": "dash",
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
    }
    raw["evaluation"]["bootstrap_replicates"] = 50
    raw["market"]["enabled"] = False
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    run_backtest(cfg, offline=True)
    return tmp_path / "artifacts"


def test_dashboard_pages_render(tmp_path, monkeypatch):
    artifacts = _run_small(tmp_path)
    monkeypatch.setenv("NFL_ORIGINATION_ARTIFACTS", str(artifacts))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value.startswith("NFL Origination Model")
    seen_unavailable = False
    for page in ("Slate", "Game detail", "Evaluation", "Run audit"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        texts = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
        if "unavailable: no eligible timestamped odds" in texts:
            seen_unavailable = True
    assert seen_unavailable  # optional market panels show the explicit unavailable label


def test_dashboard_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("NFL_ORIGINATION_ARTIFACTS", str(tmp_path / "nothing"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("No runs found" in i.value for i in at.info)
    assert os.environ["NFL_ORIGINATION_ARTIFACTS"].endswith("nothing")
