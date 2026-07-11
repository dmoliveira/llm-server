from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_server.runtime import Service, ServiceManager


def manager(tmp_path: Path) -> ServiceManager:
    return ServiceManager(tmp_path / "state")


@patch("llm_server.runtime.subprocess.Popen")
@patch("llm_server.runtime.shutil.which", return_value="/mock/mlx_lm.server")
@patch.object(ServiceManager, "_identity", return_value="started")
def test_start_uses_localhost_argument_array(
    _: MagicMock, __: MagicMock, popen: MagicMock, tmp_path: Path
) -> None:
    popen.return_value.pid = 4321
    service = manager(tmp_path).start("qwen3-8b", "qwen", 8080, 4096)
    assert service.status == "starting"
    assert popen.call_args.args[0] == [
        "/mock/mlx_lm.server",
        "--model",
        "mlx-community/Qwen3-8B-4bit",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--max-kv-size",
        "4096",
    ]
    assert popen.call_args.kwargs["start_new_session"] is True


@patch.object(ServiceManager, "_alive", return_value=True)
@patch("llm_server.runtime.subprocess.Popen")
@patch.object(ServiceManager, "_identity", return_value="started")
def test_rejects_duplicate_managed_port(
    _: MagicMock, popen: MagicMock, __: MagicMock, tmp_path: Path
) -> None:
    popen.return_value.pid = 4321
    subject = manager(tmp_path)
    subject.start("qwen3-8b", "first", 8080)
    with pytest.raises(ValueError, match="Port 8080"):
        subject.start("gemma3-12b", "second", 8080)


def test_log_tail_is_bounded_to_service_log(tmp_path: Path) -> None:
    subject = manager(tmp_path)
    subject.logs_dir.mkdir(parents=True)
    subject._write(
        {
            "safe": Service(
                name="safe", repository="model", port=8080, created_at=1, log_file="safe.log"
            )
        }
    )
    (subject.logs_dir / "safe.log").write_text("one\ntwo\nthree\n")
    assert subject.logs("safe", 2) == "two\nthree"


@patch.object(ServiceManager, "_alive", return_value=True)
@patch.object(ServiceManager, "_identity", return_value=None)
@patch("llm_server.runtime.os.killpg")
def test_stop_refuses_unverified_pid(
    killpg: MagicMock, _: MagicMock, __: MagicMock, tmp_path: Path
) -> None:
    subject = manager(tmp_path)
    subject._write(
        {
            "safe": Service(
                name="safe",
                repository="model",
                port=8080,
                pid=123,
                created_at=1,
                log_file="safe.log",
                process_identity="old",
            )
        }
    )
    with pytest.raises(RuntimeError, match="unverified"):
        subject.stop("safe")
    killpg.assert_not_called()


def test_restart_preserves_kv_setting(tmp_path: Path) -> None:
    subject = manager(tmp_path)
    service = Service(
        name="safe",
        repository="model",
        port=8080,
        created_at=1,
        log_file="safe.log",
        max_kv_size=4096,
    )
    with (
        patch.object(subject, "get", return_value=service),
        patch.object(subject, "stop"),
        patch.object(subject, "start", return_value=service) as start,
    ):
        subject.restart("safe")
    start.assert_called_once_with("model", "safe", 8080, 4096)
