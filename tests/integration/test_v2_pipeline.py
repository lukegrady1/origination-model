"""V2 offline lifecycle: receipts -> epoch -> forecast/decision commit -> close -> settle ->
report -> verify, deterministic and idempotent under a fixed clock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from nfl_origination.config import load_config
from nfl_origination.prospective.clock import FixedClock
from nfl_origination.prospective.demo import run_demo_v2
from nfl_origination.prospective.ledger import Ledger
from nfl_origination.prospective.protocol import load_epoch, verify_epoch
from nfl_origination.prospective.runner import tick, verify
from tests.conftest import ts

ROOT = Path(__file__).resolve().parents[2]


def _demo_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "v2_demo.yaml").read_text())
    raw["run"]["artifacts_dir"] = str(tmp_path / "artifacts")
    raw["run"]["reports_dir"] = str(tmp_path / "reports")
    raw["v2"]["storage"] = {
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
        "receipts_cache_dir": str(tmp_path / "raw"),
        "odds_dir": str(tmp_path / "odds"),
    }
    raw["data"]["seasons"] = [2010, 2021]
    raw["v2"]["challenger"]["development_seasons"] = [2020]
    raw["v2"]["challenger"]["retrospective_check_seasons"] = [2021]
    raw["evaluation"]["development_seasons"] = [2020]
    raw["evaluation"]["confirmation_season"] = 2021
    raw["evaluation"]["holdout_season"] = 2022
    path = tmp_path / "v2_demo.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="module")
def demo_runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2")
    cfg = load_config(_demo_config(base))
    first = run_demo_v2(cfg, clock=FixedClock("2021-08-01T00:00:00Z"))
    second_base = tmp_path_factory.mktemp("v2b")
    cfg2 = load_config(_demo_config(second_base))
    second = run_demo_v2(cfg2, clock=FixedClock("2021-08-01T00:00:00Z"))
    return cfg, first, cfg2, second


def test_lifecycle_commits_forecasts_misses_one_and_lacks_one_book(demo_runs):
    cfg, log, _cfg2, _second = demo_runs
    assert log["forecasts_committed"] >= 2
    ledger = Ledger(cfg.v2.storage.artifacts_dir, log["protocol_id"])
    events = ledger.events()
    missed = [e for e in events if e.get("event") == "missed_forecast_window"]
    assert len(missed) == 1
    steps = {s["step"]: s for s in log["steps"]}
    assert missed[0]["game_id"] == steps["ticks"]["missed_game"]
    decisions = ledger.list_decisions()
    statuses = {d.game_id: d.status for d in decisions}
    assert statuses[steps["ticks"]["missing_book_game"]] == "market_unavailable"
    assert any(s == "bet" for s in statuses.values())
    for rec, man in ledger.list_forecasts():
        assert man.eligible_for_scoring and man.eligibility_reason == "on_policy"
        assert 23.9 <= man.actual_horizon_hours <= 24.1
        assert rec.synthetic and rec.training_evidence_mode == "retrospective_reconstruction"
        assert "actual_home" not in rec.model_dump()  # outcomes never enter forecasts
    assert steps["settle"]["new_result_versions"]
    assert steps["verify"]["status"] == "ok" and steps["verify"]["epoch_status"] == "ok"
    report = json.loads(Path(log["report"]).with_suffix(".json").read_text())
    assert report["synthetic"] and report["coverage"]["missed_forecast_window"] == 1
    assert report["model_decision"] == "evidence_insufficient"


def test_two_runs_are_deterministic_and_records_idempotent(demo_runs):
    cfg, first, cfg2, second = demo_runs
    l1 = Ledger(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    l2 = Ledger(cfg2.v2.storage.artifacts_dir, second["protocol_id"])
    f1 = {(r.game_id, r.role): r for r, _ in l1.list_forecasts()}
    f2 = {(r.game_id, r.role): r for r, _ in l2.list_forecasts()}
    assert set(f1) == set(f2) and {k[1] for k in f1} == {"champion", "challenger"}
    for gid in f1:
        assert abs(f1[gid].p_home_win - f2[gid].p_home_win) <= 1e-8
        assert f1[gid].fair_home_handicap == f2[gid].fair_home_handicap
        assert f1[gid].pmf_sha256 == f2[gid].pmf_sha256
    # re-running the tick against the committed ledger creates nothing new
    epoch = load_epoch(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    clock = FixedClock(f1[next(iter(f1))].information_cutoff_utc)
    again = tick(cfg, clock=clock, protocol_id=epoch.protocol_id)
    assert not again.forecasts_created


def test_tampering_and_epoch_drift_are_detected(demo_runs):
    cfg, first, _c, _s = demo_runs
    ledger = Ledger(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    rec, _man = ledger.list_forecasts()[0]
    paths = ledger.forecast_paths(rec.forecast_id)
    original = paths["payload"].read_text()
    tampered = json.loads(original)
    tampered["p_home_win"] = 0.99
    paths["payload"].write_text(json.dumps(tampered))
    report = verify(cfg, protocol_id=first["protocol_id"])
    assert report["status"] == "integrity_failure"
    assert any(
        p["forecast_id"] == rec.forecast_id and p["problem"] == "payload_hash_mismatch"
        for p in report["problems"]
    )
    paths["payload"].write_text(original)
    pmf = paths["pmf"].read_bytes()
    paths["pmf"].unlink()
    report = verify(cfg, protocol_id=first["protocol_id"])
    assert any(p["problem"] == "missing_pmf_blob" for p in report["problems"])
    paths["pmf"].write_bytes(pmf)
    assert verify(cfg, protocol_id=first["protocol_id"])["status"] == "ok"
    epoch = load_epoch(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    changed = cfg.model_copy(update={"seed": cfg.seed + 1})
    assert any("configuration" in p for p in verify_epoch(epoch, changed))


def test_uncommitted_partial_record_is_ineligible(demo_runs):
    cfg, first, _c, _s = demo_runs
    ledger = Ledger(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    rec, _man = ledger.list_forecasts()[0]
    partial = ledger.root / "forecasts" / "fc-partial00000000000000.json"
    partial.write_text(
        rec.model_copy(update={"forecast_id": "fc-partial00000000000000"}).model_dump_json()
    )
    assert all(r.forecast_id != "fc-partial00000000000000" for r, _ in ledger.list_forecasts())
    assert any(p["problem"] == "uncommitted_partial_record" for p in ledger.verify()["problems"])
    partial.unlink()


def test_outcome_versions_do_not_modify_forecasts(demo_runs):
    cfg, first, _c, _s = demo_runs
    ledger = Ledger(cfg.v2.storage.artifacts_dir, first["protocol_id"])
    rec, _man = ledger.list_forecasts()[0]
    before = ledger.forecast_paths(rec.forecast_id)["payload"].read_bytes()
    out = ledger.latest_outcome(rec.game_id)
    assert out is not None and out.status == "final"
    corrected, created = ledger.append_outcome(
        rec.game_id,
        status="final",
        home_score=(out.home_score or 0) + 3,
        away_score=out.away_score,
        kickoff_utc=out.kickoff_utc,
        source_receipt_id="manual",
        source_hash=None,
        observed_at_utc="2030-01-01T00:00:00Z",
        clock=FixedClock("2030-01-01T00:00:00Z"),
        note="test correction",
    )
    assert created and corrected.version == out.version + 1
    assert ledger.forecast_paths(rec.forecast_id)["payload"].read_bytes() == before
    old = ledger.latest_outcome(rec.game_id, as_of=ts("2029-01-01T00:00:00Z"))
    assert old is not None and old.version == out.version
