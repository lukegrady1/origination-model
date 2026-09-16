"""Frozen experiment protocol: an automated integrity gate for the holdout (spec section 15)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.config import ExperimentConfig
from nfl_origination.errors import InvalidInputError
from nfl_origination.provenance import (
    git_info,
    hash_json,
    iso_utc,
    read_json,
    sha256_file,
    utc_now,
    write_json,
)

PROTOCOL_FILENAME = "frozen_protocol.json"
DEFAULT_PROTOCOL_LABELS = {"", "holdout", "v1"}
# Source files whose behaviour defines the research implementation. Presentation code
# (CLI wiring, report rendering, dashboard) is excluded so regenerating a report cannot
# silently redefine the experiment while a fitting/pricing change always invalidates it.
CODE_DIGEST_EXCLUDE = ("dashboard/", "cli.py", "evaluation/report.py")


def code_digest(package_dir: Path | None = None) -> str:
    """Deterministic SHA-256 over the research-relevant source tree (paths + contents)."""
    root = package_dir or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(x) or rel == x for x in CODE_DIGEST_EXCLUDE):
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def lock_digest(lock_path: Path | None = None) -> str | None:
    path = lock_path or Path(__file__).resolve().parents[2] / "uv.lock"
    return sha256_file(path) if path.exists() else None


class FrozenProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frozen_at_utc: str
    checksum: str
    label: str = "holdout"
    code_digest: str | None = None
    lock_digest: str | None = None
    selected_alpha: float | None
    ablation_selected_alpha: float | None
    holdout_season: int
    data_mode: str
    forecast_policy: str
    config_hash: str
    config: dict[str, Any]
    source_file_hashes: dict[str, str]
    exclusions_hash: str
    git_commit: str | None
    git_dirty: bool | None
    development_run_id: str | None
    confirmation_run_id: str | None
    holdout_runs: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def protocol_path(artifacts_dir: Path, label: str = "holdout") -> Path:
    """One frozen protocol per experiment label; the original V1 holdout keeps its file name."""
    if label in DEFAULT_PROTOCOL_LABELS:
        return Path(artifacts_dir) / "protocol" / PROTOCOL_FILENAME
    return Path(artifacts_dir) / "protocol" / f"frozen_protocol_{label}.json"


def _payload(
    config: ExperimentConfig,
    source_hashes: dict[str, str],
    exclusions_hash: str,
    *,
    code: str | None = None,
    lock: str | None = None,
) -> dict[str, Any]:
    return {
        "code_digest": code if code is not None else code_digest(),
        "lock_digest": lock if lock is not None else lock_digest(),
        "selected_alpha": config.model.selected_alpha,
        "ablation_selected_alpha": config.model.ablation_selected_alpha,
        "holdout_season": config.evaluation.holdout_season,
        "data_mode": config.data.mode,
        "forecast_policy": f"kickoff_minus_{config.forecast.cutoff_hours_before_kickoff:g}h",
        "config_hash": config.config_hash(),
        "source_file_hashes": source_hashes,
        "exclusions_hash": exclusions_hash,
    }


def freeze_protocol(
    config: ExperimentConfig,
    source_hashes: dict[str, str],
    exclusions_hash: str,
    *,
    development_run_id: str | None,
    confirmation_run_id: str | None,
    notes: list[str] | None = None,
) -> FrozenProtocol:
    if config.model.selected_alpha is None:
        raise InvalidInputError("cannot freeze: model.selected_alpha is unresolved")
    path = protocol_path(config.run.artifacts_dir, config.run.label)
    if path.exists():
        existing = FrozenProtocol.model_validate(read_json(path))
        if existing.holdout_runs:
            raise InvalidInputError(
                "protocol already frozen and the holdout has been run; refreezing after seeing "
                "holdout results is not allowed. Document a revised protocol instead."
            )
    payload = _payload(config, source_hashes, exclusions_hash)
    git = git_info()
    frozen = FrozenProtocol(
        frozen_at_utc=iso_utc(utc_now()) or "",
        checksum=hash_json(payload),
        label=config.run.label,
        config=config.resolved_dict(),
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        development_run_id=development_run_id,
        confirmation_run_id=confirmation_run_id,
        notes=notes or [],
        **payload,
    )
    write_json(path, frozen.model_dump())
    return frozen


def load_protocol(artifacts_dir: Path, label: str = "holdout") -> FrozenProtocol:
    path = protocol_path(artifacts_dir, label)
    if not path.exists():
        raise InvalidInputError(
            f"no frozen protocol at {path}; run `freeze-protocol` before the holdout backtest"
        )
    return FrozenProtocol.model_validate(read_json(path))


def verify_protocol(
    config: ExperimentConfig, source_hashes: dict[str, str], exclusions_hash: str
) -> FrozenProtocol:
    """Refuse a holdout run whose config/data/alpha differ from the frozen protocol."""
    frozen = load_protocol(config.run.artifacts_dir, config.run.label)
    if frozen.code_digest is None or frozen.lock_digest is None:
        raise InvalidInputError(
            "holdout refused: the frozen protocol predates code/lockfile enforcement and cannot "
            "be verified against the current implementation. Its original artifacts stay as "
            "they are; freeze a new, explicitly labeled protocol (run.label) for any rerun."
        )
    payload = _payload(config, source_hashes, exclusions_hash)
    checksum = hash_json(payload)
    if checksum != frozen.checksum:
        diffs: dict[str, Any] = {
            k: {"frozen": getattr(frozen, k, None), "current": v}
            for k, v in payload.items()
            if getattr(frozen, k, None) != v
        }
        raise InvalidInputError(
            "holdout refused: current code/config/data do not match the frozen protocol. "
            f"Differences: {diffs}"
        )
    return frozen


def record_holdout_run(
    artifacts_dir: Path, run_id: str, *, rerun_reason: str | None, label: str = "holdout"
) -> int:
    """Append a holdout execution to the protocol log; returns the 1-based run count."""
    path = protocol_path(artifacts_dir, label)
    frozen = load_protocol(artifacts_dir, label)
    entry = {"run_id": run_id, "at_utc": iso_utc(utc_now()), "rerun_reason": rerun_reason}
    frozen.holdout_runs.append(entry)
    write_json(path, frozen.model_dump())
    return len(frozen.holdout_runs)
