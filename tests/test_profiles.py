from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from llm_server.__main__ import app
from llm_server.profiles import (
    Profile,
    diff_profile,
    load_lock,
    load_profile,
    resolve_lock,
    write_lock,
)
from llm_server.provenance import acquire_locked_snapshot


def profile() -> Profile:
    return Profile.model_validate(
        {
            "schema_version": 1,
            "service": {
                "name": "writer",
                "model": {"repository": "qwen3-8b"},
                "port": 8080,
                "max_kv_size": 4096,
            },
        }
    )


def test_lock_resolves_an_alias_to_an_immutable_revision(tmp_path: Path) -> None:
    class Api:
        def model_info(self, repository: str, revision: str | None = None):
            assert (repository, revision) == ("mlx-community/Qwen3-8B-4bit", None)
            return SimpleNamespace(sha="a" * 40)

    lock = resolve_lock(profile(), Api())
    assert lock.resolved_model.revision == "a" * 40
    output = tmp_path / "lock.json"
    write_lock(lock, output)
    assert '"schema_version": 1' in output.read_text()


def test_profile_loading_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text('{"schema_version": 999, "service": {}}')
    with pytest.raises(ValueError, match="Unsupported profile schema"):
        load_profile(path)


def test_lock_rejects_a_non_commit_revision() -> None:
    class Api:
        def model_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="main")

    with pytest.raises(ValueError, match="valid immutable revision"):
        resolve_lock(profile(), Api())


def test_profile_validate_cli_path(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(profile().model_dump_json())
    result = CliRunner().invoke(app, ["profiles", "validate", str(path)])
    assert result.exit_code == 0
    assert "Valid profile" in result.output


def test_lock_can_be_loaded_and_diffed_offline(tmp_path: Path) -> None:
    class Api:
        def model_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="b" * 40)

    lock = resolve_lock(profile(), Api())
    path = tmp_path / "lock.json"
    write_lock(lock, path)
    assert load_lock(path).resolved_model.revision == "b" * 40
    assert diff_profile(profile(), load_lock(path))


def test_acquire_uses_the_locked_immutable_revision(monkeypatch, tmp_path: Path) -> None:
    lock = resolve_lock(
        profile(),
        type("Api", (), {"model_info": lambda *_args, **_kwargs: SimpleNamespace(sha="d" * 40)})(),
    )
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr("llm_server.provenance.snapshot_download", fake_download)
    assert acquire_locked_snapshot(lock, tmp_path) == tmp_path / "snapshot"
    assert captured["revision"] == "d" * 40
