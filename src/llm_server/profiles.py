"""Deterministic, single-service profile lock resolution."""

from __future__ import annotations

import json
import re
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import BaseModel, Field

from .catalog import resolve
from .contracts import ModelRef, ServiceSpec

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
