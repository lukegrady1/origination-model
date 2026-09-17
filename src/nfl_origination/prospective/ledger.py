"""Immutable, locally auditable forecast/decision/outcome ledger (V2 R3).

Layout under ``<artifacts>/prospective/<protocol_id>/``::

    forecasts/<forecast_id>.json            payload (no actual scores, ever)
    forecasts/<forecast_id>.pmf.npz         lossless joint PMF referenced by hash
    forecasts/<forecast_id>.manifest.json   payload/pmf hashes, commitment time
    forecasts/<forecast_id>.committed       atomic commit marker (manifest hash)
    decisions/<forecast_id>.json + .committed
    outcomes/<game_id>.v<N>.json            result versions, appended only
    events.jsonl                            append-only events

A record without its commit marker is not eligible. Re-running a tick returns the existing
committed record; conflicting duplicates are rejected. Hashes prove local content
consistency only; this is not external notarization.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.data.snapshots import writer_lock
from nfl_origination.errors import InvalidInputError, ModelValidationError
from nfl_origination.prospective.clock import Clock, iso
from nfl_origination.provenance import read_json

FORECAST_STATUSES = ("valid", "invalid_timing", "failed")


def forecast_id(protocol_id: str, game_id: str, horizon_policy_id: str, bundle_hash: str) -> str:
    raw = "|".join([protocol_id, game_id, horizon_policy_id, bundle_hash])
    return "fc-" + hashlib.sha256(raw.encode()).hexdigest()[:20]


class ForecastRecord(BaseModel):
    """The immutable forecast payload. Outcomes live elsewhere and are never written here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    forecast_id: str
    protocol_id: str
    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    role: str
    model_id: str
    model_family: str
    bundle_hash: str
    horizon_policy_id: str
    # times
    kickoff_at_forecast_utc: str
    target_cutoff_utc: str
    information_cutoff_utc: str
    created_at_utc: str
    # inputs
    source_receipts: dict[str, str]
    source_hashes: dict[str, str]
    max_source_observed_utc: str | None
    feature_rows_hash: str
    feature_rows: list[dict[str, Any]]
    # outputs
    mu_home_score: float
    mu_away_score: float
    dist_mean_home_score: float
    dist_mean_away_score: float
    mean_margin: float
    mean_total: float
    p_home_win: float
    p_tie: float
    p_away_win: float
    fair_home_handicap: float
    fair_total: float
    fair_decimal_home: float | None
    fair_decimal_away: float | None
    intervals: dict[str, list[int]]
    key_probabilities: dict[str, float]
    max_score: int
    margin_pmf: list[float]
    total_pmf: list[float]
    pmf_sha256: str
    tail_diagnostics: dict[str, Any]
    # provenance
    code_digest: str
    config_hash: str
    lock_digest: str | None
    training_evidence_mode: str
    forecast_evidence_mode: str
    synthetic: bool
    quality_flags: list[str] = Field(default_factory=list)
    status: str = "valid"
    status_reason: str | None = None


class ForecastManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    forecast_id: str
    payload_sha256: str
    pmf_sha256: str
    committed_at_utc: str
    actual_horizon_hours: float
    eligible_for_scoring: bool
    eligibility_reason: str
    manifest_sha256: str = ""


class DecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    forecast_id: str
    protocol_id: str
    game_id: str
    role: str
    model_id: str
    decision_time_utc: str
    policy_id: str
    status: str  # bet | no_bet | market_unavailable | model_unavailable
    reason: str
    market_probabilities: list[dict[str, Any]]
    selection: dict[str, Any] | None
    exclusions: list[dict[str, Any]]
    committed_at_utc: str
    synthetic: bool


class OutcomeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    game_id: str
    version: int
    status: str  # final | canceled | pending | rescheduled_review_required
    home_score: int | None
    away_score: int | None
    kickoff_utc: str | None
    source_receipt_id: str | None
    source_hash: str | None
    observed_at_utc: str
    recorded_at_utc: str
    note: str | None = None


def _sha_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class Ledger:
    def __init__(self, artifacts_dir: Path, protocol_id: str) -> None:
        self.root = Path(artifacts_dir) / "prospective" / protocol_id
        self.protocol_id = protocol_id
        for sub in ("forecasts", "decisions", "outcomes"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _atomic_write(self, path: Path, text: str) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    def _lock(self):  # type: ignore[no-untyped-def]
        return writer_lock(self.root, ".ledger.lock")

    # ------------------------------------------------------------------ forecasts
    def forecast_paths(self, fid: str) -> dict[str, Path]:
        base = self.root / "forecasts"
        return {
            "payload": base / f"{fid}.json",
            "pmf": base / f"{fid}.pmf.npz",
            "manifest": base / f"{fid}.manifest.json",
            "marker": base / f"{fid}.committed",
        }

    def committed_forecast(self, fid: str) -> tuple[ForecastRecord, ForecastManifest] | None:
        p = self.forecast_paths(fid)
        if not p["marker"].exists():
            return None
        manifest = ForecastManifest.model_validate(read_json(p["manifest"]))
        record = ForecastRecord.model_validate(read_json(p["payload"]))
        return record, manifest

    def commit_forecast(
        self,
        record: ForecastRecord,
        joint_pmf: np.ndarray,
        clock: Clock,
        *,
        kickoff: pd.Timestamp,
        window_minutes: float,
        target_hours: float,
    ) -> tuple[ForecastRecord, ForecastManifest, bool]:
        """Durably publish a forecast; returns (record, manifest, created).

        Idempotent per forecast_id: an existing committed record is returned unchanged and a
        conflicting payload for the same id is rejected. Commitment time is taken *after* the
        payload and PMF are durable; eligibility needs both cutoff and commitment inside the
        window and strictly before kickoff.
        """
        with self._lock():
            existing = self.committed_forecast(record.forecast_id)
            if existing is not None:
                if _sha_json(existing[0].model_dump()) != _sha_json(record.model_dump()):
                    raise InvalidInputError(
                        f"forecast {record.forecast_id} already committed with different content"
                    )
                return existing[0], existing[1], False
            p = self.forecast_paths(record.forecast_id)
            # payload + pmf first (durable), marker last
            pmf_bytes = _npz_bytes(joint_pmf)
            pmf_hash = hashlib.sha256(pmf_bytes).hexdigest()
            if pmf_hash != record.pmf_sha256:
                raise ModelValidationError("joint PMF hash does not match the record")
            tmp = p["pmf"].with_name(f".{p['pmf'].name}.{os.getpid()}.tmp")
            tmp.write_bytes(pmf_bytes)
            os.replace(tmp, p["pmf"])
            self._atomic_write(p["payload"], record.model_dump_json(indent=1))
            committed_at = clock.now()
            cutoff = pd.Timestamp(record.information_cutoff_utc)
            target = pd.Timestamp(record.target_cutoff_utc)
            window = pd.Timedelta(minutes=window_minutes)
            in_window = (
                abs(cutoff - target) <= window
                and abs(committed_at - target) <= window
                and committed_at < kickoff
                and cutoff < kickoff
            )
            if record.status != "valid":
                eligible, reason = False, record.status_reason or record.status
            elif in_window:
                eligible, reason = True, "on_policy"
            else:
                eligible, reason = False, "outside_window_or_after_kickoff"
            manifest = ForecastManifest(
                forecast_id=record.forecast_id,
                payload_sha256=_sha_json(record.model_dump()),
                pmf_sha256=pmf_hash,
                committed_at_utc=iso(committed_at),
                actual_horizon_hours=float((kickoff - cutoff).total_seconds() / 3600.0),
                eligible_for_scoring=eligible,
                eligibility_reason=reason,
            )
            manifest = manifest.model_copy(
                update={
                    "manifest_sha256": _sha_json(manifest.model_dump(exclude={"manifest_sha256"}))
                }
            )
            self._atomic_write(p["manifest"], manifest.model_dump_json(indent=1))
            self._atomic_write(p["marker"], manifest.manifest_sha256 + "\n")
            return record, manifest, True

    def list_forecasts(
        self, *, committed_only: bool = True
    ) -> list[tuple[ForecastRecord, ForecastManifest]]:
        out = []
        for payload in sorted((self.root / "forecasts").glob("fc-*.json")):
            if payload.name.endswith(".manifest.json"):
                continue
            fid = payload.name[: -len(".json")]
            got = self.committed_forecast(fid)
            if got is None and committed_only:
                continue
            if got is not None:
                out.append(got)
        return out

    def load_joint_pmf(self, fid: str) -> np.ndarray:
        with np.load(self.forecast_paths(fid)["pmf"]) as data:
            return np.asarray(data["pmf"], dtype=float)

    # ------------------------------------------------------------------ decisions
    def committed_decision(self, fid: str) -> DecisionRecord | None:
        path = self.root / "decisions" / f"{fid}.json"
        marker = self.root / "decisions" / f"{fid}.committed"
        if not (path.exists() and marker.exists()):
            return None
        return DecisionRecord.model_validate(read_json(path))

    def commit_decision(self, decision: DecisionRecord) -> tuple[DecisionRecord, bool]:
        with self._lock():
            existing = self.committed_decision(decision.forecast_id)
            if existing is not None:
                return existing, False
            path = self.root / "decisions" / f"{decision.forecast_id}.json"
            self._atomic_write(path, decision.model_dump_json(indent=1))
            self._atomic_write(
                self.root / "decisions" / f"{decision.forecast_id}.committed",
                _sha_json(decision.model_dump()) + "\n",
            )
            return decision, True

    def list_decisions(self) -> list[DecisionRecord]:
        out = []
        for path in sorted((self.root / "decisions").glob("fc-*.json")):
            d = self.committed_decision(path.name[: -len(".json")])
            if d is not None:
                out.append(d)
        return out

    # ------------------------------------------------------------------ events
    def append_event(self, event: dict[str, Any], clock: Clock) -> None:
        with self._lock():
            rec = {"recorded_at_utc": iso(clock.now()), **event}
            with (self.root / "events.jsonl").open("a") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def events(self) -> list[dict[str, Any]]:
        path = self.root / "events.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    # ------------------------------------------------------------------ outcomes
    def outcome_versions(self, game_id: str) -> list[OutcomeRecord]:
        paths = sorted((self.root / "outcomes").glob(f"{game_id}.v*.json"))
        return [OutcomeRecord.model_validate(read_json(p)) for p in paths]

    def latest_outcome(
        self, game_id: str, *, as_of: pd.Timestamp | None = None
    ) -> OutcomeRecord | None:
        versions = self.outcome_versions(game_id)
        if as_of is not None:
            versions = [v for v in versions if pd.Timestamp(v.recorded_at_utc) <= as_of]
        return versions[-1] if versions else None

    def append_outcome(
        self,
        game_id: str,
        *,
        status: str,
        home_score: int | None,
        away_score: int | None,
        kickoff_utc: str | None,
        source_receipt_id: str | None,
        source_hash: str | None,
        observed_at_utc: str,
        clock: Clock,
        note: str | None = None,
    ) -> tuple[OutcomeRecord, bool]:
        """Append a new result version only when it differs from the latest one."""
        with self._lock():
            versions = self.outcome_versions(game_id)
            if versions:
                last = versions[-1]
                same = (
                    last.status == status
                    and last.home_score == home_score
                    and last.away_score == away_score
                    and last.kickoff_utc == kickoff_utc
                )
                if same:
                    return last, False
            record = OutcomeRecord(
                game_id=game_id,
                version=len(versions) + 1,
                status=status,
                home_score=home_score,
                away_score=away_score,
                kickoff_utc=kickoff_utc,
                source_receipt_id=source_receipt_id,
                source_hash=source_hash,
                observed_at_utc=observed_at_utc,
                recorded_at_utc=iso(clock.now()),
                note=note,
            )
            path = self.root / "outcomes" / f"{game_id}.v{record.version}.json"
            self._atomic_write(path, record.model_dump_json(indent=1))
            return record, True

    # ------------------------------------------------------------------ integrity
    def verify(self) -> dict[str, Any]:
        """Recompute every hash and marker; returns a report with per-record problems."""
        problems: list[dict[str, str]] = []
        n_ok = 0
        for payload in sorted((self.root / "forecasts").glob("fc-*.json")):
            if payload.name.endswith(".manifest.json"):
                continue
            fid = payload.name[: -len(".json")]
            p = self.forecast_paths(fid)
            if not p["marker"].exists():
                problems.append({"forecast_id": fid, "problem": "uncommitted_partial_record"})
                continue
            try:
                record = ForecastRecord.model_validate(read_json(p["payload"]))
                manifest = ForecastManifest.model_validate(read_json(p["manifest"]))
            except Exception as exc:
                problems.append({"forecast_id": fid, "problem": f"unreadable: {exc}"})
                continue
            if _sha_json(record.model_dump()) != manifest.payload_sha256:
                problems.append({"forecast_id": fid, "problem": "payload_hash_mismatch"})
            if not p["pmf"].exists():
                problems.append({"forecast_id": fid, "problem": "missing_pmf_blob"})
            else:
                pmf_hash = hashlib.sha256(p["pmf"].read_bytes()).hexdigest()
                if pmf_hash != manifest.pmf_sha256 or pmf_hash != record.pmf_sha256:
                    problems.append({"forecast_id": fid, "problem": "pmf_hash_mismatch"})
            expected_marker = _sha_json(manifest.model_dump(exclude={"manifest_sha256"}))
            if (
                manifest.manifest_sha256 != expected_marker
                or p["marker"].read_text().strip() != expected_marker
            ):
                problems.append({"forecast_id": fid, "problem": "manifest_or_marker_mismatch"})
            if not any(x["forecast_id"] == fid for x in problems):
                n_ok += 1
        for path in sorted((self.root / "decisions").glob("fc-*.json")):
            marker = path.with_suffix(".committed")
            if not marker.exists():
                problems.append({"forecast_id": path.stem, "problem": "uncommitted_decision"})
                continue
            d = DecisionRecord.model_validate(read_json(path))
            if marker.read_text().strip() != _sha_json(d.model_dump()):
                problems.append({"forecast_id": path.stem, "problem": "decision_marker_mismatch"})
        return {
            "protocol_id": self.protocol_id,
            "forecasts_ok": n_ok,
            "problems": problems,
            "status": "ok" if not problems else "integrity_failure",
            "note": "local content-consistency check only; not external notarization",
        }

    def excluded_forecast_ids(self) -> set[str]:
        return {p["forecast_id"] for p in self.verify()["problems"]}


def _npz_bytes(pmf: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    np.savez(buf, pmf=np.asarray(pmf, dtype=np.float64))
    return buf.getvalue()


def pmf_sha256(pmf: np.ndarray) -> str:
    return hashlib.sha256(_npz_bytes(pmf)).hexdigest()


def summarize_ledger(ledger: Ledger, *, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    forecasts = ledger.list_forecasts()
    events = ledger.events()
    counts: dict[str, int] = {"committed": len(forecasts), "eligible": 0, "ineligible": 0}
    for _rec, man in forecasts:
        counts["eligible" if man.eligible_for_scoring else "ineligible"] += 1
    counts["missed_forecast_window"] = sum(
        1 for e in events if e.get("event") == "missed_forecast_window"
    )
    counts["collection_failures"] = sum(1 for e in events if e.get("event") == "collection_failure")
    settled = 0
    pending = 0
    for rec, _man in forecasts:
        out = ledger.latest_outcome(rec.game_id, as_of=as_of)
        if out is not None and out.status == "final":
            settled += 1
        else:
            pending += 1
    counts["settled"] = settled
    counts["pending"] = pending
    return counts
