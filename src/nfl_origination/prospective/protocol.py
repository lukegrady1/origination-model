"""Prospective epochs (V2 R3): immutable, system-timed, code/lock/config/bundle pinned."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nfl_origination.config import ExperimentConfig
from nfl_origination.errors import InvalidInputError, MissingDataError, ModelValidationError
from nfl_origination.models.bundle import ModelBundle
from nfl_origination.prospective.clock import Clock, iso
from nfl_origination.protocol import lock_digest
from nfl_origination.provenance import hash_json, read_json, sha256_file, write_json
from nfl_origination.schemas import FEATURE_VERSION, SCHEMA_VERSION

# Everything that predicts, ingests, times, matches, evaluates, settles or orchestrates.
V2_CODE_EXCLUDE = ("dashboard/",)


def v2_code_digest(package_dir: Path | None = None) -> str:
    root = package_dir or Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(x) for x in V2_CODE_EXCLUDE):
            continue
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


class BundleRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    model_id: str
    family: str
    bundle_path: str
    bundle_hash: str
    created_at_utc: str
    training_evidence_mode: str
    forecast_evidence_mode: str
    training_source_manifest_hash: str | None
    contract_hash: str | None
    calibration: dict[str, Any] = Field(default_factory=dict)


class Epoch(BaseModel):
    """An immutable prospective protocol. Any change to code/model/policy needs a new epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: str
    label: str
    activated_at_utc: str
    synthetic: bool
    evidence_mode: str
    champion: BundleRef
    challenger: BundleRef | None
    code_digest: str
    lock_digest: str | None
    config: dict[str, Any]
    config_hash: str
    feature_version: str
    schema_version: int
    seed: int
    scope: dict[str, Any]
    horizon: dict[str, Any]
    market_policy: dict[str, Any]
    paper_policy: dict[str, Any]
    review_thresholds: dict[str, Any]
    planned_population: str
    payload_hash: str
    notes: list[str] = Field(default_factory=list)


def epochs_dir(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "prospective" / "epochs"


def active_pointer_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / "prospective" / "active_epoch.json"


def bundle_ref(
    role: str,
    path: Path,
    *,
    forecast_evidence_mode: str,
) -> BundleRef:
    bundle = ModelBundle.load(path)
    training_mode = bundle.data_mode
    notes = " ".join(bundle.notes)
    manifest_hash = None
    for n in bundle.notes:
        if n.startswith("training_source_manifest_hash="):
            manifest_hash = n.split("=", 1)[1]
    calibration = {}
    if bundle.family == "key_number_adjusted":
        calibration = dict(bundle.model_params.get("calibration", {}))
    return BundleRef(
        role=role,
        model_id=bundle.model_id,
        family=bundle.family,
        bundle_path=str(path),
        bundle_hash=sha256_file(path),
        created_at_utc=bundle.created_at_utc,
        training_evidence_mode=(
            "retrospective_reconstruction"
            if training_mode == "historical_reconstruction" or "retrospective" in notes
            else "observed_prospective"
        ),
        forecast_evidence_mode=forecast_evidence_mode,
        training_source_manifest_hash=manifest_hash,
        contract_hash=bundle.contract_hash,
        calibration=calibration,
    )


def freeze_epoch(
    config: ExperimentConfig,
    clock: Clock,
    *,
    champion_path: Path,
    challenger_path: Path | None,
    label: str,
    notes: list[str] | None = None,
) -> Epoch:
    """Freeze an epoch with the system-assigned activation time and set the active pointer."""
    v2 = config.require_v2()
    activated = clock.now()
    champion = bundle_ref("champion", champion_path, forecast_evidence_mode=v2.evidence.mode)
    challenger = (
        bundle_ref("challenger", challenger_path, forecast_evidence_mode=v2.evidence.mode)
        if challenger_path is not None
        else None
    )
    for ref in (champion, challenger):
        if ref is not None and ref.created_at_utc > iso(activated):
            raise InvalidInputError(
                f"{ref.role} bundle created at {ref.created_at_utc} after activation "
                f"{iso(activated)}"
            )
    payload: dict[str, Any] = {
        "label": label,
        "synthetic": v2.evidence.synthetic,
        "evidence_mode": v2.evidence.mode,
        "champion": champion.model_dump(),
        "challenger": None if challenger is None else challenger.model_dump(),
        "code_digest": v2_code_digest(),
        "lock_digest": lock_digest(),
        "config": config.resolved_dict(),
        "config_hash": config.config_hash(),
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "seed": config.seed,
        "scope": {"game_type": config.data.game_type, "seasons": list(config.data.seasons)},
        "horizon": v2.horizon.model_dump(),
        "market_policy": {
            **v2.market.model_dump(),
            "closing_proxy": v2.closing_proxy.model_dump(),
            "collection": v2.collection.model_dump(),
        },
        "paper_policy": v2.paper.model_dump(),
        "review_thresholds": {
            "planned_review_games": v2.evaluation.planned_review_games,
            "planned_review_blocks": v2.evaluation.planned_review_blocks,
            "min_blocks_for_intervals": v2.evaluation.min_blocks,
            "note": "review thresholds, not statistical power guarantees",
        },
        "planned_population": "all eligible NFL regular-season games after activation; "
        "predeclared review after >=100 matched settled games and 8 season-week blocks, "
        "else end-of-season descriptive report",
    }
    payload_hash = hash_json(payload)
    protocol_id = "epoch-" + payload_hash[:16]
    epoch = Epoch(
        protocol_id=protocol_id,
        activated_at_utc=iso(activated),
        payload_hash=payload_hash,
        notes=notes or [],
        **payload,
    )
    root = epochs_dir(v2.storage.artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{protocol_id}.json"
    if path.exists():
        existing = Epoch.model_validate(read_json(path))
        if existing.payload_hash != payload_hash:
            raise InvalidInputError(f"epoch {protocol_id} exists with a different payload")
        epoch = existing
    else:
        write_json(path, epoch.model_dump())
    write_json(
        active_pointer_path(v2.storage.artifacts_dir),
        {"protocol_id": epoch.protocol_id, "set_at_utc": iso(clock.now()), "label": label},
    )
    return epoch


def load_epoch(artifacts_dir: Path, protocol_id: str | None = None) -> Epoch:
    if protocol_id is None:
        pointer = active_pointer_path(artifacts_dir)
        if not pointer.exists():
            raise MissingDataError(
                f"no active epoch pointer at {pointer}; run freeze-prospective first"
            )
        protocol_id = str(read_json(pointer)["protocol_id"])
    path = epochs_dir(artifacts_dir) / f"{protocol_id}.json"
    if not path.exists():
        raise MissingDataError(f"epoch {protocol_id} not found under {epochs_dir(artifacts_dir)}")
    epoch = Epoch.model_validate(read_json(path))
    recomputed = hash_json(
        {
            k: v
            for k, v in epoch.model_dump().items()
            if k not in ("protocol_id", "activated_at_utc", "payload_hash", "notes")
        }
    )
    if recomputed != epoch.payload_hash:
        raise ModelValidationError(f"epoch {protocol_id} payload hash mismatch: file was altered")
    return epoch


def verify_epoch(epoch: Epoch, config: ExperimentConfig) -> list[str]:
    """Return the list of mismatches between the frozen epoch and the current environment."""
    problems: list[str] = []
    if v2_code_digest() != epoch.code_digest:
        problems.append("code_digest changed since the epoch was frozen")
    if lock_digest() != epoch.lock_digest:
        problems.append("uv.lock digest changed since the epoch was frozen")
    if config.config_hash() != epoch.config_hash:
        problems.append("resolved configuration differs from the frozen epoch")
    for ref in (epoch.champion, epoch.challenger):
        if ref is None:
            continue
        path = Path(ref.bundle_path)
        if not path.exists():
            problems.append(f"{ref.role} bundle missing: {path}")
        elif sha256_file(path) != ref.bundle_hash:
            problems.append(f"{ref.role} bundle hash changed: {path}")
    return problems


def require_verified_epoch(config: ExperimentConfig, protocol_id: str | None = None) -> Epoch:
    v2 = config.require_v2()
    epoch = load_epoch(v2.storage.artifacts_dir, protocol_id)
    problems = verify_epoch(epoch, config)
    if problems:
        raise InvalidInputError(
            "epoch verification failed; changes to code, model, scoring rules or policy need a "
            f"new epoch: {problems}"
        )
    return epoch
