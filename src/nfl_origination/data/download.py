"""nflverse asset adapter: cached, checksummed, atomic downloads (spec section 5).

Cache layout: ``<cache_dir>/<dataset>/<season>/<sha256>.<ext>`` with an ``entry.json`` per
dataset/season pointing at the current snapshot and recording provenance. Old snapshots are
never deleted, so a failed refresh cannot corrupt the last complete cache and recorded-asof
mode can reason about when a snapshot was first observed.
"""

from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.errors import InvalidInputError, MissingDataError
from nfl_origination.provenance import iso_utc, read_json, sha256_file, utc_now, write_json

NFLVERSE_RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULE_URL = f"{NFLVERSE_RELEASES}/schedules/games.csv"
ATTRIBUTION = (
    "Data from nflverse (https://github.com/nflverse/nflverse-data), schedules maintained by "
    "Lee Sharpe; play-by-play from nflfastR. Review the upstream license before redistribution."
)
SCHEDULE_DATASET = "schedules"
PBP_DATASET = "pbp"
ALL_SEASONS = "all"


def pbp_url(season: int) -> str:
    return f"{NFLVERSE_RELEASES}/pbp/play_by_play_{season}.parquet"


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    season: int | None
    url: str
    path: str
    sha256: str
    bytes: int
    downloaded_at_utc: str
    first_observed_at_utc: str
    row_count: int
    columns: list[str]
    source_version: str | None = None
    attribution: str = ATTRIBUTION
    receipt_id: str | None = None
    provenance_quality: str | None = None


class SourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at_utc: str
    offline: bool
    cache_dir: str
    entries: list[SourceEntry] = Field(default_factory=list)

    def entry(self, dataset: str, season: int | None) -> SourceEntry:
        for e in self.entries:
            if e.dataset == dataset and e.season == season:
                return e
        raise MissingDataError(f"source manifest has no entry for {dataset}/{season}")

    def file_hashes(self) -> dict[str, str]:
        return {f"{e.dataset}/{e.season or ALL_SEASONS}": e.sha256 for e in self.entries}


def _entry_dir(cache_dir: Path, dataset: str, season: int | None) -> Path:
    return cache_dir / dataset / (str(season) if season is not None else ALL_SEASONS)


def _describe(path: Path) -> tuple[int, list[str]]:
    if path.suffix == ".parquet":
        meta = pq.read_metadata(path)
        return meta.num_rows, list(pq.read_schema(path).names)
    table = pacsv.read_csv(path)
    return table.num_rows, list(table.column_names)


def fetch_to_file(
    url: str,
    dest: Path,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 1.5,
    session: requests.Session | None = None,
) -> None:
    """Download with bounded retries and an atomic rename. Partial files never replace data."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sess = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with sess.get(url, stream=True, timeout=(10.0, timeout)) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
            os.replace(part, dest)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if part.exists():
                part.unlink()
            if attempt < retries:
                time.sleep(backoff ** (attempt - 1) * 0.2)
    raise MissingDataError(f"download failed after {retries} attempts: {url}: {last_error}")


def _load_entry(cache_dir: Path, dataset: str, season: int | None) -> SourceEntry | None:
    path = _entry_dir(cache_dir, dataset, season) / "entry.json"
    if not path.exists():
        return None
    return SourceEntry.model_validate(read_json(path))


def _refresh_entry(
    cache_dir: Path,
    dataset: str,
    season: int | None,
    url: str,
    *,
    timeout: float,
    retries: int,
    session: requests.Session | None,
) -> SourceEntry:
    entry_dir = _entry_dir(cache_dir, dataset, season)
    entry_dir.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if url.endswith(".parquet") else ".csv"
    tmp = entry_dir / f"download-{os.getpid()}-{secrets.token_hex(4)}{ext}"
    request_started = iso_utc(utc_now())
    fetch_to_file(url, tmp, timeout=timeout, retries=retries, session=session)
    observed_at = iso_utc(utc_now())  # complete response received
    digest = sha256_file(tmp)
    final = entry_dir / f"{digest}{ext}"
    now = iso_utc(utc_now())
    assert now is not None
    previous = _load_entry(cache_dir, dataset, season)
    if final.exists():
        tmp.unlink()
    else:
        os.replace(tmp, final)
    try:
        rows, cols = _describe(final)
    except Exception as exc:  # corrupt download: keep previous entry untouched
        final.unlink(missing_ok=True)
        raise MissingDataError(
            f"downloaded file for {dataset}/{season} is unreadable: {exc}"
        ) from exc
    first_observed = now
    if previous is not None and previous.sha256 == digest:
        first_observed = previous.first_observed_at_utc
    entry = SourceEntry(
        dataset=dataset,
        season=season,
        url=url,
        path=str(final),
        sha256=digest,
        bytes=final.stat().st_size,
        downloaded_at_utc=now,
        first_observed_at_utc=first_observed,
        row_count=rows,
        columns=cols,
    )
    history_path = entry_dir / "history.jsonl"
    with history_path.open("a") as fh:
        fh.write(entry.model_dump_json() + "\n")
    write_json(entry_dir / "entry.json", entry.model_dump())
    # V2: immutable per-observation receipt (blob stored once per hash, receipt per observation)
    from nfl_origination.data.snapshots import record_observation

    assert observed_at is not None
    receipt = record_observation(
        cache_dir,
        dataset=dataset,
        season=season,
        blob_path=final,
        sha256=digest,
        request_started_at_utc=request_started,
        observed_at_utc=observed_at,
        source_url=url,
        row_count=rows,
        columns=cols,
    )
    return entry.model_copy(
        update={"receipt_id": receipt.receipt_id, "provenance_quality": "observed"}
    )


def ingest(
    seasons: list[int],
    cache_dir: Path,
    offline: bool,
    *,
    refresh: bool = True,
    timeout: float = 120.0,
    retries: int = 3,
    session: requests.Session | None = None,
) -> SourceManifest:
    """Ensure the schedule file and each season's PBP asset are cached; return the manifest.

    With ``offline=True`` no network access happens; missing cache entries raise
    ``MissingDataError``. With ``refresh=False`` existing entries are reused without a fetch.
    """
    if not seasons:
        raise InvalidInputError("ingest requires at least one season")
    cache_dir = Path(cache_dir)
    entries: list[SourceEntry] = []
    targets: list[tuple[str, int | None, str]] = [(SCHEDULE_DATASET, None, SCHEDULE_URL)]
    targets += [(PBP_DATASET, s, pbp_url(s)) for s in seasons]
    for dataset, season, url in targets:
        existing = _load_entry(cache_dir, dataset, season)
        if offline or not refresh:
            if existing is None:
                if offline:
                    raise MissingDataError(
                        f"--offline: no cached {dataset}/{season or ALL_SEASONS} under {cache_dir}"
                    )
                existing = _refresh_entry(
                    cache_dir,
                    dataset,
                    season,
                    url,
                    timeout=timeout,
                    retries=retries,
                    session=session,
                )
            elif not Path(existing.path).exists():
                raise MissingDataError(
                    f"cache entry for {dataset}/{season} points to a missing file"
                )
            entries.append(existing)
            continue
        try:
            entries.append(
                _refresh_entry(
                    cache_dir,
                    dataset,
                    season,
                    url,
                    timeout=timeout,
                    retries=retries,
                    session=session,
                )
            )
        except MissingDataError:
            if existing is not None and Path(existing.path).exists():
                entries.append(existing)  # failed refresh keeps the last complete snapshot
            else:
                raise
    now = iso_utc(utc_now())
    assert now is not None
    return SourceManifest(
        created_at_utc=now, offline=offline, cache_dir=str(cache_dir), entries=entries
    )


def load_manifest_from_cache(seasons: list[int], cache_dir: Path) -> SourceManifest:
    """Build a manifest from cached entries only (no network)."""
    return ingest(seasons, cache_dir, offline=True)


def read_entry(entry: SourceEntry, columns: list[str] | None = None) -> pd.DataFrame:
    path = Path(entry.path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def manifest_summary(manifest: SourceManifest) -> dict[str, Any]:
    return {
        "entries": len(manifest.entries),
        "seasons": sorted({e.season for e in manifest.entries if e.season is not None}),
        "bytes": sum(e.bytes for e in manifest.entries),
        "offline": manifest.offline,
    }
