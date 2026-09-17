"""Immutable source observation receipts, as-of version selection, legacy migration (V2 R1).

A blob is stored once per content hash; every successful observation of a source produces its
own immutable receipt. The as-of selector picks the most recently *observed* valid receipt at or
before a cutoff (A -> B -> A histories follow the receipt sequence), with a deterministic
receipt-ID tiebreaker. Legacy V1 metadata (``entry.json``/``history.jsonl``) is imported
additively with ``provenance_quality=legacy_metadata``; orphan blobs without timing evidence
stay ``availability_unknown`` and can never support a strict as-of claim.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.data.download import (
    ALL_SEASONS,
    PBP_DATASET,
    SCHEDULE_DATASET,
    SourceEntry,
    SourceManifest,
    _describe,
)
from nfl_origination.errors import InvalidInputError, MissingDataError
from nfl_origination.provenance import iso_utc, read_json, sha256_file, utc_now

ProvenanceQuality = Literal["observed", "legacy_metadata", "availability_unknown", "synthetic"]
USABLE_QUALITIES = ("observed", "legacy_metadata", "synthetic")
ADAPTER_VERSION = "nflverse-receipts/2"
RECEIPTS_DIR = "receipts"
SECRET_PATTERN = re.compile(r"(api[_-]?key|token|secret)=([^&\s]+)", re.IGNORECASE)


def redact_url(url: str) -> str:
    return SECRET_PATTERN.sub(lambda m: f"{m.group(1)}=REDACTED", url)


class Receipt(BaseModel):
    """One successful observation of one source version. Never modified after persistence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: str
    dataset: str
    season: int | None
    content_sha256: str
    blob_path: str
    bytes: int
    request_started_at_utc: str | None
    observed_at_utc: str | None
    persisted_at_utc: str
    provider_timestamp_utc: str | None = None
    source_url_redacted: str
    adapter_version: str
    synthetic: bool = False
    provenance_quality: ProvenanceQuality
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    note: str | None = None

    @property
    def usable_for_asof(self) -> bool:
        return self.provenance_quality in USABLE_QUALITIES and self.observed_at_utc is not None


def receipts_root(cache_dir: Path) -> Path:
    return Path(cache_dir) / RECEIPTS_DIR


def _receipt_dir(cache_dir: Path, dataset: str, season: int | None) -> Path:
    return receipts_root(cache_dir) / dataset / (str(season) if season is not None else ALL_SEASONS)


_PROCESS_LOCK = threading.RLock()
_LOCK_DEPTH: dict[str, int] = {}
_LOCK_HANDLES: dict[str, Any] = {}


@contextmanager
def writer_lock(cache_dir: Path, name: str = ".writer.lock") -> Iterator[None]:
    """Exclusive local lock so two collectors cannot publish conflicting receipts.

    Re-entrant within a process (a process-level RLock guards the file lock); exclusive across
    processes via ``flock`` on a lock file inside the receipts root.
    """
    root = receipts_root(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / name
    key = str(lock_path)
    with _PROCESS_LOCK:
        depth = _LOCK_DEPTH.get(key, 0)
        if depth == 0:
            fh = lock_path.open("a+")
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            _LOCK_HANDLES[key] = fh
        _LOCK_DEPTH[key] = depth + 1
        try:
            yield
        finally:
            _LOCK_DEPTH[key] -= 1
            if _LOCK_DEPTH[key] == 0:
                fh = _LOCK_HANDLES.pop(key)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()


def _compact(ts: str | None) -> str:
    """Full-precision compact timestamp; sub-second digits keep same-second receipts distinct."""
    return "unknown" if ts is None else re.sub(r"[^0-9T]", "", ts)


def make_receipt_id(
    dataset: str, season: int | None, sha256: str, observed_at_utc: str | None, quality: str
) -> str:
    prefix = {
        "observed": "obs",
        "legacy_metadata": "legacy",
        "availability_unknown": "unknown",
        "synthetic": "syn",
    }[quality]
    season_txt = season if season is not None else ALL_SEASONS
    return f"{prefix}-{dataset}-{season_txt}-{_compact(observed_at_utc)}-{sha256[:12]}"


def write_receipt(cache_dir: Path, receipt: Receipt) -> Path:
    """Atomically persist a receipt. Existing receipts are never overwritten."""
    rdir = _receipt_dir(cache_dir, receipt.dataset, receipt.season)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"{receipt.receipt_id}.json"
    if path.exists():
        return path
    tmp = rdir / f".{receipt.receipt_id}.{os.getpid()}.tmp"
    tmp.write_text(receipt.model_dump_json(indent=1))
    os.replace(tmp, path)
    return path


def list_receipts(
    cache_dir: Path, dataset: str | None = None, season: int | str | None = "any"
) -> list[Receipt]:
    root = receipts_root(cache_dir)
    if not root.exists():
        return []
    out: list[Receipt] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        r = Receipt.model_validate(read_json(path))
        if dataset is not None and r.dataset != dataset:
            continue
        if season != "any" and r.season != season:
            continue
        out.append(r)
    return out


def record_observation(
    cache_dir: Path,
    *,
    dataset: str,
    season: int | None,
    blob_path: Path,
    sha256: str,
    request_started_at_utc: str | None,
    observed_at_utc: str,
    source_url: str,
    synthetic: bool = False,
    provider_timestamp_utc: str | None = None,
    row_count: int | None = None,
    columns: list[str] | None = None,
) -> Receipt:
    """Create an ``observed`` receipt for a complete, verified download."""
    receipt = Receipt(
        receipt_id=make_receipt_id(
            dataset, season, sha256, observed_at_utc, "synthetic" if synthetic else "observed"
        ),
        dataset=dataset,
        season=season,
        content_sha256=sha256,
        blob_path=str(blob_path),
        bytes=Path(blob_path).stat().st_size,
        request_started_at_utc=request_started_at_utc,
        observed_at_utc=observed_at_utc,
        persisted_at_utc=iso_utc(utc_now()) or "",
        provider_timestamp_utc=provider_timestamp_utc,
        source_url_redacted=redact_url(source_url),
        adapter_version=ADAPTER_VERSION,
        synthetic=synthetic,
        provenance_quality="synthetic" if synthetic else "observed",
        row_count=row_count,
        columns=columns or [],
    )
    with writer_lock(cache_dir):
        write_receipt(cache_dir, receipt)
    return receipt


def select_version(
    cache_dir: Path,
    dataset: str,
    season: int | None,
    cutoff: pd.Timestamp,
    *,
    allow_synthetic: bool = False,
) -> Receipt | None:
    """Most recently observed usable receipt with ``observed_at <= cutoff``; ties by receipt_id."""
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise InvalidInputError("as-of cutoff must be timezone-aware UTC")
    best: Receipt | None = None
    for r in list_receipts(cache_dir, dataset, season):
        if not r.usable_for_asof or (r.synthetic and not allow_synthetic):
            continue
        observed = pd.Timestamp(r.observed_at_utc)
        if observed > cutoff or not Path(r.blob_path).exists():
            continue
        if best is None:
            best = r
            continue
        best_obs = pd.Timestamp(best.observed_at_utc)
        if observed > best_obs or (observed == best_obs and r.receipt_id > best.receipt_id):
            best = r
    return best


def manifest_as_of(
    cache_dir: Path,
    seasons: list[int],
    cutoff: pd.Timestamp,
    *,
    allow_synthetic: bool = False,
    targets: list[tuple[str, int | None]] | None = None,
) -> SourceManifest:
    """Source manifest pinned to the versions observed at or before ``cutoff``.

    ``first_observed_at_utc`` on each entry is the selected receipt's observation time, so
    downstream availability checks see exactly when this version became available locally.
    """
    entries: list[SourceEntry] = []
    missing: list[str] = []
    if targets is None:
        targets = [(SCHEDULE_DATASET, None), *[(PBP_DATASET, s) for s in seasons]]
    for dataset, season in targets:
        r = select_version(cache_dir, dataset, season, cutoff, allow_synthetic=allow_synthetic)
        if r is None:
            missing.append(f"{dataset}/{season if season is not None else ALL_SEASONS}")
            continue
        assert r.observed_at_utc is not None
        entries.append(
            SourceEntry(
                dataset=dataset,
                season=season,
                url=r.source_url_redacted,
                path=r.blob_path,
                sha256=r.content_sha256,
                bytes=r.bytes,
                downloaded_at_utc=r.observed_at_utc,
                first_observed_at_utc=r.observed_at_utc,
                row_count=r.row_count or 0,
                columns=r.columns,
                receipt_id=r.receipt_id,
                provenance_quality=r.provenance_quality,
            )
        )
    if missing:
        raise MissingDataError(
            f"no source version observed at or before {iso_utc(cutoff.to_pydatetime())} "
            f"for {missing}; "
            "unknown availability is ineligible"
        )
    return SourceManifest(
        created_at_utc=iso_utc(utc_now()) or "",
        offline=True,
        cache_dir=str(cache_dir),
        entries=entries,
    )


# ---------------------------------------------------------------------------------------------
# Legacy inventory and migration
# ---------------------------------------------------------------------------------------------


def _legacy_entries(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry_dir in sorted(Path(cache_dir).glob("*/*")):
        if not entry_dir.is_dir() or entry_dir.parent.name == RECEIPTS_DIR:
            continue
        history = entry_dir / "history.jsonl"
        if history.exists():
            for line in history.read_text().splitlines():
                if line.strip():
                    rows.append({"source": str(history), **json.loads(line)})
        entry = entry_dir / "entry.json"
        if entry.exists():
            rows.append({"source": str(entry), **read_json(entry)})
    return rows


def snapshot_inventory(cache_dir: Path) -> dict[str, Any]:
    """Read-only inventory: legacy metadata, receipts, orphan blobs. Never writes."""
    cache_dir = Path(cache_dir)
    legacy = _legacy_entries(cache_dir)
    receipts = list_receipts(cache_dir)
    known_hashes = {r["sha256"] for r in legacy} | {r.content_sha256 for r in receipts}
    blobs = [
        p
        for p in cache_dir.glob("*/*/*")
        if p.suffix in (".parquet", ".csv") and p.parent.parent.name != RECEIPTS_DIR
    ]
    orphans = [str(p) for p in blobs if p.stem not in known_hashes]
    by_quality: dict[str, int] = {}
    for r in receipts:
        by_quality[r.provenance_quality] = by_quality.get(r.provenance_quality, 0) + 1
    per_source: dict[str, dict[str, Any]] = {}
    for r in receipts:
        key = f"{r.dataset}/{r.season if r.season is not None else ALL_SEASONS}"
        slot = per_source.setdefault(
            key,
            {"receipts": 0, "versions": set(), "earliest_observed": None, "latest_observed": None},
        )
        slot["receipts"] += 1
        slot["versions"].add(r.content_sha256[:12])
        if r.observed_at_utc:
            slot["earliest_observed"] = min(
                filter(None, [slot["earliest_observed"], r.observed_at_utc])
            )
            slot["latest_observed"] = max(
                filter(None, [slot["latest_observed"], r.observed_at_utc])
            )
    for slot in per_source.values():
        slot["versions"] = sorted(slot["versions"])
    return {
        "cache_dir": str(cache_dir),
        "legacy_metadata_records": len(legacy),
        "receipts": len(receipts),
        "receipts_by_quality": by_quality,
        "blobs": len(blobs),
        "orphan_blobs": orphans,
        "per_source": per_source,
        "note": "legacy timestamps are the V1 collector's own metadata, not independent evidence; "
        "orphan blobs have unknown availability",
    }


def migrate_legacy(cache_dir: Path, *, baseline: bool = True) -> dict[str, Any]:
    """Additive, idempotent migration of V1 metadata into receipts. Original files untouched.

    - Every ``history.jsonl``/``entry.json`` record becomes a ``legacy_metadata`` receipt whose
      observation time is the legacy ``downloaded_at_utc`` (a V1 collector claim).
    - Blobs with no timing evidence become ``availability_unknown`` receipts (no observed time).
    - With ``baseline=True``, each blob that exists and hashes correctly now also gets an
      ``observed`` receipt at the current time: availability from now on, never earlier.
    """
    cache_dir = Path(cache_dir)
    created: list[str] = []
    skipped = 0
    now = iso_utc(utc_now()) or ""
    with writer_lock(cache_dir):
        existing = {r.receipt_id for r in list_receipts(cache_dir)}
        seen_hashes: set[str] = set()
        for rec in _legacy_entries(cache_dir):
            blob = Path(rec["path"])
            rid = make_receipt_id(
                rec["dataset"],
                rec["season"],
                rec["sha256"],
                rec["downloaded_at_utc"],
                "legacy_metadata",
            )
            seen_hashes.add(rec["sha256"])
            if rid in existing:
                skipped += 1
                continue
            receipt = Receipt(
                receipt_id=rid,
                dataset=rec["dataset"],
                season=rec["season"],
                content_sha256=rec["sha256"],
                blob_path=str(blob),
                bytes=int(rec["bytes"]),
                request_started_at_utc=None,
                observed_at_utc=rec["downloaded_at_utc"],
                persisted_at_utc=now,
                source_url_redacted=redact_url(rec["url"]),
                adapter_version="legacy-v1-entry",
                provenance_quality="legacy_metadata",
                row_count=rec.get("row_count"),
                columns=list(rec.get("columns", [])),
                note=f"imported from {Path(rec['source']).name}; "
                "downloaded_at_utc used as observation claim",
            )
            write_receipt(cache_dir, receipt)
            existing.add(rid)
            created.append(rid)
        for blob in sorted(cache_dir.glob("*/*/*")):
            if blob.suffix not in (".parquet", ".csv") or blob.parent.parent.name == RECEIPTS_DIR:
                continue
            dataset = blob.parent.parent.name
            season_txt = blob.parent.name
            season = None if season_txt == ALL_SEASONS else int(season_txt)
            digest = blob.stem
            if digest not in seen_hashes:
                rid = make_receipt_id(dataset, season, digest, None, "availability_unknown")
                if rid not in existing:
                    write_receipt(
                        cache_dir,
                        Receipt(
                            receipt_id=rid,
                            dataset=dataset,
                            season=season,
                            content_sha256=digest,
                            blob_path=str(blob),
                            bytes=blob.stat().st_size,
                            request_started_at_utc=None,
                            observed_at_utc=None,
                            persisted_at_utc=now,
                            source_url_redacted="unknown",
                            adapter_version="legacy-orphan",
                            provenance_quality="availability_unknown",
                            note="no timing evidence; usable only for labeled "
                            "retrospective reconstruction",
                        ),
                    )
                    existing.add(rid)
                    created.append(rid)
            if baseline:
                has_observed = any(
                    r.provenance_quality == "observed" and r.content_sha256 == digest
                    for r in list_receipts(cache_dir, dataset, season)
                )
                if not has_observed and sha256_file(blob) == digest:
                    rows, cols = _describe(blob)
                    receipt = Receipt(
                        receipt_id=make_receipt_id(dataset, season, digest, now, "observed"),
                        dataset=dataset,
                        season=season,
                        content_sha256=digest,
                        blob_path=str(blob),
                        bytes=blob.stat().st_size,
                        request_started_at_utc=now,
                        observed_at_utc=now,
                        persisted_at_utc=now,
                        source_url_redacted="local-baseline-inventory",
                        adapter_version="baseline-inventory/2",
                        provenance_quality="observed",
                        row_count=rows,
                        columns=cols,
                        note="baseline inventory: hashed locally at V2 activation; "
                        "availability from now only",
                    )
                    write_receipt(cache_dir, receipt)
                    existing.add(receipt.receipt_id)
                    created.append(receipt.receipt_id)
    return {
        "created": created,
        "skipped_existing": skipped,
        "at_utc": now,
        "inventory": snapshot_inventory(cache_dir),
    }


def receipt_hash_map(manifest: SourceManifest) -> dict[str, str]:
    return {
        f"{e.dataset}/{e.season if e.season is not None else ALL_SEASONS}": (e.receipt_id or "")
        for e in manifest.entries
    }


def verify_receipt_blob(receipt: Receipt) -> bool:
    path = Path(receipt.blob_path)
    return path.exists() and sha256_file(path) == receipt.content_sha256


def digest_manifest(manifest: SourceManifest) -> str:
    payload = sorted(
        (e.dataset, e.season, e.sha256, e.receipt_id, e.first_observed_at_utc)
        for e in manifest.entries
    )
    return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()
