"""Revision-pinned model acquisition for reproducible local serving."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from .profiles import Lockfile


def acquire_locked_snapshot(lock: Lockfile, cache_dir: Path | None = None) -> Path:
    """Fetch exactly the lock's immutable revision and return its local snapshot directory."""
    revision = lock.resolved_model.revision
    if revision is None:
        raise ValueError("Lockfile has no immutable revision")
    snapshot = snapshot_download(
        repo_id=lock.resolved_model.repository,
        revision=revision,
        cache_dir=cache_dir,
    )
    return Path(snapshot)
