"""Ledger contracts: concurrency, conflicting duplicates, commit ordering, decisions, events."""

from __future__ import annotations

import threading
from datetime import timedelta

import numpy as np
import pytest

from nfl_origination.errors import InvalidInputError
from nfl_origination.prospective.clock import FixedClock
from nfl_origination.prospective.ledger import (
    DecisionRecord,
    ForecastRecord,
    Ledger,
    forecast_id,
    pmf_sha256,
)
from tests.conftest import point_mass, ts

KICK = ts("2020-09-13T17:00:00Z")
TARGET = KICK - timedelta(hours=24)


def _record(fid: str, pmf: np.ndarray, *, cutoff=TARGET, status="valid") -> ForecastRecord:
    return ForecastRecord(
        forecast_id=fid,
        protocol_id="epoch-x",
        game_id="g1",
        season=2020,
        week=1,
        home_team="H",
        away_team="A",
        role="champion",
        model_id="M1",
        model_family="ridge_score",
        bundle_hash="b" * 64,
        horizon_policy_id="h",
        kickoff_at_forecast_utc="2020-09-13T17:00:00Z",
        target_cutoff_utc="2020-09-12T17:00:00Z",
        information_cutoff_utc=cutoff.isoformat().replace("+00:00", "Z"),
        created_at_utc="2020-09-12T17:00:00Z",
        source_receipts={},
        source_hashes={},
        max_source_observed_utc=None,
        feature_rows_hash="f",
        feature_rows=[],
        mu_home_score=24.0,
        mu_away_score=21.0,
        dist_mean_home_score=24.0,
        dist_mean_away_score=21.0,
        mean_margin=3.0,
        mean_total=45.0,
        p_home_win=1.0,
        p_tie=0.0,
        p_away_win=0.0,
        fair_home_handicap=-3.5,
        fair_total=44.5,
        fair_decimal_home=1.0,
        fair_decimal_away=None,
        intervals={},
        key_probabilities={},
        max_score=100,
        margin_pmf=[0.0] * 201,
        total_pmf=[0.0] * 201,
        pmf_sha256=pmf_sha256(pmf),
        tail_diagnostics={},
        code_digest="c",
        config_hash="k",
        lock_digest=None,
        training_evidence_mode="retrospective_reconstruction",
        forecast_evidence_mode="observed_prospective",
        synthetic=True,
        status=status,
    )


def test_uniqueness_key_is_deterministic():
    assert forecast_id("p", "g", "h", "b") == forecast_id("p", "g", "h", "b")
    assert forecast_id("p", "g", "h", "b") != forecast_id("p", "g", "h", "c")


def test_concurrent_writers_commit_exactly_once(tmp_path):
    ledger = Ledger(tmp_path, "epoch-x")
    pmf = point_mass(24, 21).pmf
    fid = forecast_id("epoch-x", "g1", "h", "b" * 64)
    rec = _record(fid, pmf)
    created: list[bool] = []

    def worker() -> None:
        clock = FixedClock(TARGET + timedelta(minutes=1))
        _r, _m, was_created = ledger.commit_forecast(
            rec, pmf, clock, kickoff=KICK, window_minutes=5, target_hours=24
        )
        created.append(was_created)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(created) == 1 and len(ledger.list_forecasts()) == 1


def test_conflicting_duplicate_rejected_and_bad_timing_ineligible(tmp_path):
    ledger = Ledger(tmp_path, "epoch-x")
    pmf = point_mass(24, 21).pmf
    fid = forecast_id("epoch-x", "g1", "h", "b" * 64)
    clock = FixedClock(TARGET + timedelta(minutes=2))
    _r, man, created = ledger.commit_forecast(
        _record(fid, pmf), pmf, clock, kickoff=KICK, window_minutes=5, target_hours=24
    )
    assert created and man.eligible_for_scoring and abs(man.actual_horizon_hours - 24.0) < 1e-9
    with pytest.raises(InvalidInputError, match="different content"):
        ledger.commit_forecast(
            _record(fid, pmf).model_copy(update={"mu_home_score": 30.0}),
            pmf,
            clock,
            kickoff=KICK,
            window_minutes=5,
            target_hours=24,
        )
    # late commitment (outside the window) is committed but ineligible; cutoff after kickoff too
    late_fid = forecast_id("epoch-x", "g2", "h", "b" * 64)
    late_clock = FixedClock(TARGET + timedelta(minutes=30))
    _r, man2, _ = ledger.commit_forecast(
        _record(late_fid, pmf), pmf, late_clock, kickoff=KICK, window_minutes=5, target_hours=24
    )
    assert (
        not man2.eligible_for_scoring
        and man2.eligibility_reason == "outside_window_or_after_kickoff"
    )
    bad = forecast_id("epoch-x", "g3", "h", "b" * 64)
    _r, man3, _ = ledger.commit_forecast(
        _record(bad, pmf, status="invalid_timing"),
        pmf,
        clock,
        kickoff=KICK,
        window_minutes=5,
        target_hours=24,
    )
    assert not man3.eligible_for_scoring and man3.eligibility_reason == "invalid_timing"


def test_pmf_hash_must_match_record(tmp_path):
    from nfl_origination.errors import ModelValidationError

    ledger = Ledger(tmp_path, "epoch-x")
    pmf = point_mass(24, 21).pmf
    other = point_mass(20, 20).pmf
    fid = forecast_id("epoch-x", "g1", "h", "b" * 64)
    with pytest.raises(ModelValidationError):
        ledger.commit_forecast(
            _record(fid, pmf),
            other,
            FixedClock(TARGET),
            kickoff=KICK,
            window_minutes=5,
            target_hours=24,
        )
    assert ledger.committed_forecast(fid) is None  # nothing partially eligible


def test_decisions_events_and_outcomes(tmp_path):
    ledger = Ledger(tmp_path, "epoch-x")
    clock = FixedClock(TARGET)
    d = DecisionRecord(
        forecast_id="fc-a",
        protocol_id="epoch-x",
        game_id="g1",
        role="champion",
        model_id="M1",
        decision_time_utc="t",
        policy_id="p",
        status="no_bet",
        reason="r",
        market_probabilities=[],
        selection=None,
        exclusions=[],
        committed_at_utc="t",
        synthetic=True,
    )
    _d, created = ledger.commit_decision(d)
    _d2, created2 = ledger.commit_decision(d.model_copy(update={"status": "bet"}))
    assert created and not created2 and ledger.committed_decision("fc-a").status == "no_bet"
    ledger.append_event({"event": "missed_forecast_window", "game_id": "g9"}, clock)
    assert ledger.events()[0]["event"] == "missed_forecast_window"
    _o1, c1 = ledger.append_outcome(
        "g1",
        status="pending",
        home_score=None,
        away_score=None,
        kickoff_utc=None,
        source_receipt_id=None,
        source_hash=None,
        observed_at_utc="t",
        clock=clock,
    )
    _o2, c2 = ledger.append_outcome(
        "g1",
        status="pending",
        home_score=None,
        away_score=None,
        kickoff_utc=None,
        source_receipt_id=None,
        source_hash=None,
        observed_at_utc="t",
        clock=clock,
    )
    o3, c3 = ledger.append_outcome(
        "g1",
        status="final",
        home_score=24,
        away_score=21,
        kickoff_utc=None,
        source_receipt_id=None,
        source_hash=None,
        observed_at_utc="t",
        clock=clock,
    )
    assert (c1, c2, c3) == (True, False, True) and o3.version == 2
    assert ledger.latest_outcome("g1").status == "final"
    assert ledger.verify()["status"] == "ok"
