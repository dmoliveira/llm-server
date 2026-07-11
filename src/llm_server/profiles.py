"""Deterministic, single-service profile lock resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from .catalog import resolve
from .contracts import ModelRef, ServiceSpec
from .runtime import Service

PROFILE_SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class Profile(BaseModel):
    schema_version: int = Field(default=PROFILE_SCHEMA_VERSION)
    service: ServiceSpec


class Lockfile(BaseModel):
    schema_version: int = Field(default=LOCK_SCHEMA_VERSION)
    profile_schema_version: int
    service: ServiceSpec
    resolved_model: ModelRef


class ApplyPlan(BaseModel):
    action: str
    service: str
    detail: str


def load_profile(path: Path) -> Profile:
    """Load a JSON profile; reject unsupported schemas before resolving network state."""
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("Profile must be a JSON object")
        if raw.get("schema_version", PROFILE_SCHEMA_VERSION) != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported profile schema version: {raw['schema_version']}")
        profile = Profile.model_validate(raw)
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"Could not load profile {path}: {error}") from error
    return profile


def resolve_lock(profile: Profile, api: HfApi | None = None) -> Lockfile:
    """Resolve the requested alias/repository to a commit SHA without launching a service."""
    requested = profile.service.model
    repository = resolve(requested.repository).repository
    info = (api or HfApi()).model_info(repository, revision=requested.revision)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not COMMIT_SHA_PATTERN.fullmatch(revision):
        raise ValueError(f"Hub did not provide a valid immutable revision for {repository}")
    return Lockfile(
        profile_schema_version=profile.schema_version,
        service=profile.service.model_copy(
            update={"model": ModelRef(repository=repository, revision=revision)}
        ),
        resolved_model=ModelRef(repository=repository, revision=revision),
    )


def write_lock(lock: Lockfile, path: Path) -> None:
    """Write stable JSON suitable for review and source control."""
    path.write_text(json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def load_lock(path: Path) -> Lockfile:
    """Load a lockfile without contacting the Hub."""
    try:
        lock = Lockfile.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load lockfile {path}: {error}") from error
    if lock.schema_version != LOCK_SCHEMA_VERSION:
        raise ValueError(f"Unsupported lock schema version: {lock.schema_version}")
    if not lock.resolved_model.revision or not COMMIT_SHA_PATTERN.fullmatch(
        lock.resolved_model.revision
    ):
        raise ValueError("Lockfile does not contain a valid immutable revision")
    return lock


def diff_profile(profile: Profile, lock: Lockfile) -> list[str]:
    """Return deterministic intent differences; never mutates a service."""
    differences: list[str] = []
    if profile.schema_version != lock.profile_schema_version:
        differences.append(
            f"profile schema: {profile.schema_version} → {lock.profile_schema_version}"
        )
    if profile.service.name != lock.service.name:
        differences.append(f"service name: {profile.service.name} → {lock.service.name}")
    if profile.service.port != lock.service.port:
        differences.append(f"port: {profile.service.port} → {lock.service.port}")
    if profile.service.max_kv_size != lock.service.max_kv_size:
        differences.append(
            f"max_kv_size: {profile.service.max_kv_size} → {lock.service.max_kv_size}"
        )
    if profile.service.model.repository != lock.resolved_model.repository:
        differences.append(
            f"model: {profile.service.model.repository} → {lock.resolved_model.repository}"
        )
    if profile.service.model.revision != lock.resolved_model.revision:
        differences.append(
            f"revision: {profile.service.model.revision} → {lock.resolved_model.revision}"
        )
    return differences


def plan_apply(lock: Lockfile, services: list[Service]) -> ApplyPlan:
    """Describe a lock-aware apply action without spawning, stopping, or mutating anything."""
    desired = lock.service
    current = next((service for service in services if service.name == desired.name), None)
    if current is None or current.status in {"stopped", "failed"}:
        return ApplyPlan(
            action="start",
            service=desired.name,
            detail=(
                f"would start {lock.resolved_model.repository}@{lock.resolved_model.revision} "
                f"on port {desired.port}"
            ),
        )
    if (
        current.repository != lock.resolved_model.repository
        or current.port != desired.port
        or current.max_kv_size != desired.max_kv_size
        or current.revision != lock.resolved_model.revision
        or not current.offline
        or current.provenance != "locked-and-cached"
        or not current.snapshot_path
    ):
        return ApplyPlan(
            action="conflict",
            service=desired.name,
            detail="running service differs from lock; no mutation will be performed",
        )
    return ApplyPlan(
        action="unchanged", service=desired.name, detail="running service matches lock intent"
    )
