"""Dashboard smoke for V2: research run, prospective epoch (pending/settled/missing market),
legacy V1 run and evidence labels render without exceptions and without any network or fit."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from nfl_origination.config import load_config
from nfl_origination.experiment import run_backtest
from nfl_origination.prospective.clock import FixedClock
from nfl_origination.prospective.demo import run_demo_v2

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "src" / "nfl_origination" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    base = tmp_path_factory.mktemp("dash")
    raw = yaml.safe_load((ROOT / "configs" / "v2_demo.yaml").read_text())
    raw["run"]["artifacts_dir"] = str(base / "v2")
    raw["run"]["reports_dir"] = str(base / "reports")
    raw["v2"]["storage"] = {
        "artifacts_dir": str(base / "v2"),
        "reports_dir": str(base / "reports"),
        "receipts_cache_dir": str(base / "raw"),
        "odds_dir": str(base / "odds"),
    }
    raw["v2"]["evaluation"]["bootstrap_replicates"] = 50
    cfg_path = base / "v2_demo.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    run_demo_v2(cfg, clock=FixedClock("2021-08-01T00:00:00Z"))
    # a legacy V1-style run in a separate root
    legacy = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
    legacy["run"] = {
        "kind": "development",
        "label": "legacy",
        "artifacts_dir": str(base / "legacy"),
        "reports_dir": str(base / "reports"),
    }
    legacy["evaluation"]["bootstrap_replicates"] = 50
    legacy["market"]["enabled"] = False
    lp = base / "legacy.yaml"
    lp.write_text(yaml.safe_dump(legacy))
    run_backtest(load_config(lp), offline=True)
    return base


def test_all_pages_render_with_evidence_labels(artifacts, monkeypatch):
    monkeypatch.setenv(
        "NFL_ORIGINATION_ARTIFACTS", f"{artifacts / 'legacy'}:{artifacts / 'v2' / 'research'}"
    )
    monkeypatch.setenv("NFL_ORIGINATION_PROSPECTIVE", str(artifacts / "v2"))
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "SYNTHETIC" in warnings or "RETROSPECTIVE" in warnings
    assert "verified" not in warnings.lower()
    for page in ("Slate", "Game detail", "Evaluation", "Prospective", "Run audit"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
        if page == "Prospective":
            text = (
                " ".join(m.value for m in at.markdown)
                + " ".join(c.value for c in at.caption)
                + " ".join(w.value for w in at.warning)
            )
            assert "PROSPECTIVE" in text or "SYNTHETIC" in text
            assert "no odds" in text or "market_unavailable" in text or "missing" in text.lower()
    # switch to the research run and render its pages
    runs = at.sidebar.selectbox[0].options
    research = [r for r in runs if "v2research" in r]
    assert research
    at.sidebar.radio[0].set_value("Evaluation").run()
    at.sidebar.selectbox[0].set_value(research[0]).run()
    assert not at.exception
    assert any("RETROSPECTIVE" in w.value or "SYNTHETIC" in w.value for w in at.warning)


def test_empty_prospective_state(artifacts, monkeypatch):
    monkeypatch.setenv("NFL_ORIGINATION_ARTIFACTS", str(artifacts / "legacy"))
    monkeypatch.setenv("NFL_ORIGINATION_PROSPECTIVE", str(artifacts / "nothing"))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Prospective").run()
    assert not at.exception
    assert any("No prospective epochs" in i.value for i in at.info)
