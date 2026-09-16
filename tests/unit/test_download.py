"""Mocked download behaviour: retries, atomic writes, corruption safety, offline mode."""

from __future__ import annotations

import pytest
import responses

from nfl_origination.data.download import PBP_DATASET, SCHEDULE_URL, ingest, pbp_url
from nfl_origination.errors import MissingDataError


def _csv(rows: int) -> str:
    return "game_id,season\n" + "\n".join(f"g{i},2020" for i in range(rows)) + "\n"


@responses.activate
def test_ingest_downloads_and_caches(tmp_path):
    responses.add(responses.GET, SCHEDULE_URL, body=_csv(3), status=200)
    responses.add(responses.GET, pbp_url(2020), body=b"not parquet", status=500)
    responses.add(responses.GET, pbp_url(2020), body=b"still bad", status=500)
    responses.add(responses.GET, pbp_url(2020), body=_csv(2), status=200)  # served as csv content
    with pytest.raises(MissingDataError):
        # third attempt returns csv bytes at a .parquet url -> unreadable -> cache stays empty
        ingest([2020], tmp_path, offline=False, retries=3)
    assert not (tmp_path / PBP_DATASET / "2020" / "entry.json").exists()
    assert (tmp_path / "schedules" / "all" / "entry.json").exists()
    assert (
        not list((tmp_path / PBP_DATASET / "2020").glob("*.part"))
        if (tmp_path / PBP_DATASET / "2020").exists()
        else True
    )


@responses.activate
def test_failed_refresh_keeps_previous_snapshot(tmp_path):
    responses.add(responses.GET, SCHEDULE_URL, body=_csv(3), status=200)
    manifest = ingest([], tmp_path, offline=False) if False else None  # noqa: F841
    # first: only schedule
    responses.add(responses.GET, pbp_url(2019), body=_csv(1), status=200)
    with pytest.raises(MissingDataError):
        ingest([2019], tmp_path, offline=False)  # csv at parquet url is unreadable
    entry = tmp_path / "schedules" / "all" / "entry.json"
    assert entry.exists()
    first = entry.read_text()
    responses.reset()
    responses.add(responses.GET, SCHEDULE_URL, body="boom", status=503)
    responses.add(responses.GET, pbp_url(2019), body="boom", status=503)
    with pytest.raises(MissingDataError):
        ingest([2019], tmp_path, offline=False, retries=1)
    assert entry.read_text() == first  # last complete cache untouched


def test_offline_prohibits_network(tmp_path):
    with pytest.raises(MissingDataError):
        ingest([2020], tmp_path, offline=True)
