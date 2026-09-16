"""Run manifests, hashing, and run-directory registry (spec section 4)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from nfl_origination import __version__
from nfl_origination.errors import InvalidInputError

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "duckdb",
    "pydantic",
    "typer",
    "matplotlib",
    "streamlit",
)


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_utc(ts: datetime | None) -> str | None:
    return None if ts is None else ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def hash_frame(df: pd.DataFrame) -> str:
    """Deterministic content hash of a dataframe (column names + row hashes)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    if len(df):
        row_hashes = pd.util.hash_pandas_object(df.reset_index(drop=True), index=False)
        h.update(row_hashes.to_numpy().tobytes())
    return h.hexdigest()


def hash_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")))


def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def dependency_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version(), "nfl_origination": __version__}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def new_run_id(prefix: str) -> str:
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


class RunManifest(BaseModel):
    """Immutable description of one run. Written once at run completion."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_kind: str
    label: str
    created_at_utc: str
    git_commit: str | None
    git_dirty: bool | None
    platform: str
    dependency_versions: dict[str, str]
    seed: int
    config: dict[str, Any]
    config_hash: str
    source_file_hashes: dict[str, str]
    schema_version: int
    feature_version: str
    model_id: str | None
    training_cutoff_utc: str | None
    forecast_policy: str
    data_mode: str
    evaluation_mode: str
    output_hashes: dict[str, str] = Field(default_factory=dict)
    protocol_checksum: str | None = None
    notes: list[str] = Field(default_factory=list)
    status: str = "completed"
    runtime_seconds: float | None = None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


class RunRegistry:
    """Maps `latest-<kind>` aliases to recorded run IDs under artifacts/runs."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.runs_dir = artifacts_dir / "runs"
        self.index_path = self.runs_dir / "index.json"

    def _load(self) -> dict[str, Any]:
        if self.index_path.exists():
            data = read_json(self.index_path)
            assert isinstance(data, dict)
            return data
        return {"latest": {}, "runs": []}

    def create_run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=False)
        return path

    def record(self, manifest: RunManifest) -> None:
        index = self._load()
        index.setdefault("latest", {})[manifest.run_kind] = manifest.run_id
        index.setdefault("latest", {})[f"{manifest.run_kind}:{manifest.label}"] = manifest.run_id
        index.setdefault("runs", []).append(
            {
                "run_id": manifest.run_id,
                "kind": manifest.run_kind,
                "created_at_utc": manifest.created_at_utc,
            }
        )
        write_json(self.index_path, index)

    def resolve(self, ref: str) -> str:
        """Resolve `latest-<kind>` or an exact run ID to a run ID."""
        if ref.startswith("latest-"):
            kind = ref[len("latest-") :]
            latest = self._load().get("latest", {})
            if kind not in latest:
                raise InvalidInputError(
                    f"no recorded run for alias {ref!r}; known: {sorted(latest)}"
                )
            return str(latest[kind])
        if (self.runs_dir / ref).exists():
            return ref
        raise InvalidInputError(f"unknown run reference {ref!r}")

    def run_dir(self, ref: str) -> Path:
        return self.runs_dir / self.resolve(ref)

    def load_manifest(self, ref: str) -> RunManifest:
        path = self.run_dir(ref) / "manifest.json"
        if not path.exists():
            raise InvalidInputError(f"run {ref!r} has no manifest.json")
        return RunManifest.model_validate(read_json(path))


def environment_note() -> str:
    return f"{platform.platform()} python={sys.version.split()[0]}"
