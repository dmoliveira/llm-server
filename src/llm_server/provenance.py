"""Revision-pinned model acquisition for reproducible local serving."""

from __future__ import annotations

from hashlib import sha256
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


def snapshot_digest(snapshot: Path) -> str:
    """Hash snapshot entries deterministically without following cache symlinks."""
    root = snapshot.resolve()
    if not root.is_dir():
        raise ValueError(f"Snapshot path does not exist: {snapshot}")
    digest = sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(path.readlink().as_posix().encode())
            continue
        if not path.is_file():
            continue
        digest.update(path.read_bytes())
    return digest.hexdigest()
