"""V2 R1: immutable receipts, as-of version selection, legacy migration, writer lock."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import pytest
import responses

from nfl_origination.data.download import SCHEDULE_URL, ingest, pbp_url
from nfl_origination.data.snapshots import (
    Receipt,
    list_receipts,
    manifest_as_of,
    migrate_legacy,
    record_observation,
    redact_url,
    select_version,
    snapshot_inventory,
    write_receipt,
    writer_lock,
)
from nfl_origination.errors import MissingDataError
from tests.conftest import ts


def _blob(tmp_path: Path, name: str, content: str) -> tuple[Path, str]:
    import hashlib

    data = content.encode()
    digest = hashlib.sha256(data).hexdigest()
    path = tmp_path / "pbp" / "2020" / f"{digest}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, digest


def _observe(cache, path, digest, at: str, season=2020, dataset="pbp"):
    return record_observation(
        cache,
        dataset=dataset,
        season=season,
        blob_path=path,
        sha256=digest,
        request_started_at_utc=at,
        observed_at_utc=at,
        source_url=f"https://example.test/{dataset}?apiKey=SECRET",
    )


def test_receipts_are_immutable_and_redacted(tmp_path):
    path, digest = _blob(tmp_path, "a", "game_id,season\ng1,2020\n")
    r = _observe(tmp_path, path, digest, "2020-09-01T00:00:00Z")
    assert "SECRET" not in r.source_url_redacted and "REDACTED" in r.source_url_redacted
    assert r.provenance_quality == "observed"
    stored = list_receipts(tmp_path)
    assert [x.receipt_id for x in stored] == [r.receipt_id]
    # writing the same receipt again does not overwrite
    again = write_receipt(tmp_path, r.model_copy(update={"note": "tampered"}))
    assert Receipt.model_validate(json.loads(again.read_text())).note is None
    assert redact_url("https://x/y?token=abc&z=1") == "https://x/y?token=REDACTED&z=1"


def test_asof_selection_follows_observation_sequence(tmp_path):
    pa, da = _blob(tmp_path, "a", "A\n")
    pb, db = _blob(tmp_path, "b", "B\n")
    _observe(tmp_path, pa, da, "2020-09-01T00:00:00Z")
    _observe(tmp_path, pb, db, "2020-09-05T00:00:00Z")
    _observe(tmp_path, pa, da, "2020-09-09T00:00:00Z")  # A -> B -> A
    assert select_version(tmp_path, "pbp", 2020, ts("2020-09-03T00:00:00Z")).content_sha256 == da
    assert select_version(tmp_path, "pbp", 2020, ts("2020-09-06T00:00:00Z")).content_sha256 == db
    chosen = select_version(tmp_path, "pbp", 2020, ts("2020-09-10T00:00:00Z"))
    assert chosen.content_sha256 == da and chosen.observed_at_utc == "2020-09-09T00:00:00Z"
    assert select_version(tmp_path, "pbp", 2020, ts("2020-08-01T00:00:00Z")) is None
    # identical repeat fetch keeps the earlier receipt and adds a later one; never backdates
    assert len([r for r in list_receipts(tmp_path) if r.content_sha256 == da]) == 2
    with pytest.raises(MissingDataError, match="unknown availability is ineligible"):
        manifest_as_of(tmp_path, [2020], ts("2020-08-01T00:00:00Z"))


def test_tiebreak_is_deterministic_by_receipt_id(tmp_path):
    pa, da = _blob(tmp_path, "a", "A\n")
    pb, db = _blob(tmp_path, "b", "B\n")
    _observe(tmp_path, pa, da, "2020-09-01T00:00:00Z")
    _observe(tmp_path, pb, db, "2020-09-01T00:00:00Z")
    first = select_version(tmp_path, "pbp", 2020, ts("2020-09-02T00:00:00Z"))
    ids = sorted(r.receipt_id for r in list_receipts(tmp_path))
    assert first.receipt_id == ids[-1]


def test_unknown_availability_never_selected(tmp_path):
    path, digest = _blob(tmp_path, "a", "A\n")
    write_receipt(
        tmp_path,
        Receipt(
            receipt_id="unknown-pbp-2020-unknown-" + digest[:12],
            dataset="pbp",
            season=2020,
            content_sha256=digest,
            blob_path=str(path),
            bytes=2,
            request_started_at_utc=None,
            observed_at_utc=None,
            persisted_at_utc="2026-01-01T00:00:00Z",
            source_url_redacted="?",
            adapter_version="x",
            provenance_quality="availability_unknown",
        ),
    )
    assert select_version(tmp_path, "pbp", 2020, ts("2030-01-01T00:00:00Z")) is None


def test_missing_blob_is_not_selectable(tmp_path):
    path, digest = _blob(tmp_path, "a", "A\n")
    _observe(tmp_path, path, digest, "2020-09-01T00:00:00Z")
    path.unlink()
    assert select_version(tmp_path, "pbp", 2020, ts("2020-09-02T00:00:00Z")) is None


@responses.activate
def test_ingest_writes_observed_receipts_and_failed_downloads_do_not(tmp_path):
    responses.add(responses.GET, SCHEDULE_URL, body="game_id,season\ng1,2020\n", status=200)
    responses.add(responses.GET, pbp_url(2020), body="boom", status=500)
    with pytest.raises(MissingDataError):
        ingest([2020], tmp_path, offline=False, retries=1)
    receipts = list_receipts(tmp_path)
    assert [r.dataset for r in receipts] == ["schedules"]
    assert (
        receipts[0].request_started_at_utc
        <= receipts[0].observed_at_utc
        <= receipts[0].persisted_at_utc
    )
    responses.reset()
    responses.add(responses.GET, SCHEDULE_URL, body="game_id,season\ng1,2020\n", status=200)
    responses.add(responses.GET, pbp_url(2020), body="boom", status=500)
    with pytest.raises(MissingDataError):
        ingest([2020], tmp_path, offline=False, retries=1)
    assert len(list_receipts(tmp_path, "schedules")) == 2  # a second observation, same blob
    assert len(list((tmp_path / "schedules" / "all").glob("*.csv"))) == 1  # stored once


def test_legacy_migration_is_additive_and_idempotent(tmp_path):
    path, digest = _blob(tmp_path, "a", "game_id,season\ng1,2020\n")
    entry = {
        "dataset": "pbp",
        "season": 2020,
        "url": "https://x/pbp?token=S",
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "downloaded_at_utc": "2026-09-16T17:00:00Z",
        "first_observed_at_utc": "2026-09-16T17:00:00Z",
        "row_count": 1,
        "columns": ["game_id", "season"],
    }
    (path.parent / "history.jsonl").write_text(json.dumps(entry) + "\n")
    (path.parent / "entry.json").write_text(json.dumps(entry))
    orphan, _ = _blob(tmp_path, "o", "orphan\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    rep = migrate_legacy(tmp_path)
    after = {p: before.get(p) == p.read_bytes() for p in before}
    assert all(after.values())  # originals byte-for-byte unchanged
    qualities = {r.receipt_id: r for r in list_receipts(tmp_path)}
    legacy = [r for r in qualities.values() if r.provenance_quality == "legacy_metadata"]
    assert len(legacy) == 1 and legacy[0].observed_at_utc == "2026-09-16T17:00:00Z"
    unknown = [r for r in qualities.values() if r.provenance_quality == "availability_unknown"]
    assert len(unknown) == 1 and unknown[0].blob_path == str(orphan)
    baseline = [r for r in qualities.values() if r.provenance_quality == "observed"]
    assert baseline and all(r.observed_at_utc >= rep["at_utc"][:19] for r in baseline)
    n = len(qualities)
    rep2 = migrate_legacy(tmp_path)
    assert len(list_receipts(tmp_path)) == n and rep2["created"] == []
    earliest = min(r.observed_at_utc for r in list_receipts(tmp_path) if r.observed_at_utc)
    assert earliest == "2026-09-16T17:00:00Z"  # no earlier availability was created
    inv = snapshot_inventory(tmp_path)
    assert inv["orphan_blobs"] == [] and inv["receipts_by_quality"]["legacy_metadata"] == 1


def test_writer_lock_serializes_concurrent_writers(tmp_path):
    path, digest = _blob(tmp_path, "a", "A\n")
    order: list[str] = []

    def worker(name: str) -> None:
        with writer_lock(tmp_path):
            order.append(f"{name}-in")
            _observe(tmp_path, path, digest, f"2020-09-0{1 if name == 'a' else 2}T00:00:00Z")
            order.append(f"{name}-out")

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])
    assert len(list_receipts(tmp_path)) == 2


def test_manifest_as_of_carries_receipt_identity(tmp_path):
    sched_dir = tmp_path / "schedules" / "all"
    sched_dir.mkdir(parents=True)
    sp = sched_dir / "abc.csv"
    sp.write_text("game_id\n")
    import hashlib

    sd = hashlib.sha256(sp.read_bytes()).hexdigest()
    sp2 = sched_dir / f"{sd}.csv"
    sp.rename(sp2)
    record_observation(
        tmp_path,
        dataset="schedules",
        season=None,
        blob_path=sp2,
        sha256=sd,
        request_started_at_utc="2020-09-01T00:00:00Z",
        observed_at_utc="2020-09-01T00:00:00Z",
        source_url="u",
    )
    pa, da = _blob(tmp_path, "a", "A\n")
    _observe(tmp_path, pa, da, "2020-09-02T00:00:00Z")
    m = manifest_as_of(tmp_path, [2020], ts("2020-09-03T00:00:00Z"))
    assert {e.dataset for e in m.entries} == {"schedules", "pbp"}
    assert all(e.receipt_id and e.provenance_quality == "observed" for e in m.entries)
    assert m.entry("pbp", 2020).first_observed_at_utc == "2020-09-02T00:00:00Z"
    assert pd.Timestamp(m.entry("pbp", 2020).first_observed_at_utc) <= ts("2020-09-03T00:00:00Z")
