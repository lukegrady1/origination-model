"""Frozen experiment protocol: an automated integrity gate for the holdout (spec section 15)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.config import ExperimentConfig
from nfl_origination.errors import InvalidInputError
from nfl_origination.provenance import git_info, hash_json, iso_utc, read_json, utc_now, write_json

PROTOCOL_FILENAME = "frozen_protocol.json"


class FrozenProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frozen_at_utc: str
    checksum: str
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


def protocol_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "protocol" / PROTOCOL_FILENAME


def _payload(
    config: ExperimentConfig,
    source_hashes: dict[str, str],
    exclusions_hash: str,
) -> dict[str, Any]:
    return {
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
    path = protocol_path(config.run.artifacts_dir)
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


def load_protocol(artifacts_dir: Path) -> FrozenProtocol:
    path = protocol_path(artifacts_dir)
    if not path.exists():
        raise InvalidInputError(
            f"no frozen protocol at {path}; run `freeze-protocol` before the holdout backtest"
        )
    return FrozenProtocol.model_validate(read_json(path))


def verify_protocol(
    config: ExperimentConfig, source_hashes: dict[str, str], exclusions_hash: str
) -> FrozenProtocol:
    """Refuse a holdout run whose config/data/alpha differ from the frozen protocol."""
    frozen = load_protocol(config.run.artifacts_dir)
    payload = _payload(config, source_hashes, exclusions_hash)
    checksum = hash_json(payload)
    if checksum != frozen.checksum:
        diffs = {
            k: {"frozen": getattr(frozen, k), "current": v}
            for k, v in payload.items()
            if getattr(frozen, k) != v
        }
        raise InvalidInputError(
            "holdout refused: current config/data do not match the frozen protocol. "
            f"Differences: {diffs}"
        )
    return frozen


def record_holdout_run(artifacts_dir: Path, run_id: str, *, rerun_reason: str | None) -> int:
    """Append a holdout execution to the protocol log; returns the 1-based run count."""
    path = protocol_path(artifacts_dir)
    frozen = load_protocol(artifacts_dir)
    entry = {"run_id": run_id, "at_utc": iso_utc(utc_now()), "rerun_reason": rerun_reason}
    frozen.holdout_runs.append(entry)
    write_json(path, frozen.model_dump())
    return len(frozen.holdout_runs)
