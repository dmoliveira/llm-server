"""Safe, localhost-only lifecycle management for MLX-LM child processes."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .catalog import resolve
from .contracts import STATE_SCHEMA_VERSION


class Service(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
    repository: str
    port: int = Field(ge=1024, le=65535)
    pid: int | None = None
    status: str = "stopped"
    created_at: float
    log_file: str
    max_kv_size: int | None = None
    process_identity: str | None = None
    error: str | None = None
    revision: str | None = None
    snapshot_path: str | None = None
    offline: bool = False


class StateCorruptError(RuntimeError):
    """Persisted state cannot safely be used by a control-plane operation."""


class ServiceManager:
    """The sole process/state mutation boundary for CLI and HTTP adapters."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path.home() / ".local" / "share" / "llm-server"
        self.logs_dir = self.data_dir / "logs"
        self.state_file = self.data_dir / "services.json"
        self.lock_file = self.data_dir / ".lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, Service]:
        if not self.state_file.exists():
            return {}
        try:
            raw = json.loads(self.state_file.read_text())
            if not isinstance(raw, dict):
                raise ValueError("Service state must be a JSON object")
            if "schema_version" in raw:
                if raw["schema_version"] != STATE_SCHEMA_VERSION:
                    raise RuntimeError(
                        f"Unsupported service state schema version: {raw['schema_version']}"
                    )
                raw = raw.get("services")
                if not isinstance(raw, dict):
                    raise StateCorruptError(
                        "Versioned service state is missing its services mapping"
                    )
            return {name: Service.model_validate(item) for name, item in raw.items()}
        except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as error:
            raise StateCorruptError(f"Service state is corrupt: {self.state_file}") from error

    def _write(self, services: dict[str, Service]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "services": {name: item.model_dump() for name, item in services.items()},
                },
                indent=2,
            )
        )
        os.replace(temporary, self.state_file)

    @staticmethod
    def _alive(pid: int | None) -> bool:
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _identity(pid: int | None) -> str | None:
        """Return a macOS process start token to defend against PID reuse."""
        if pid is None:
            return None
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, check=False, text=True
        )
        return result.stdout.strip() or None

    def _owned(self, service: Service) -> bool:
        return (
            self._alive(service.pid)
            and bool(service.process_identity)
            and (service.process_identity == self._identity(service.pid))
        )

    @staticmethod
    def _matches(current: Service, observed: Service) -> bool:
        """Prevent a late lifecycle operation from overwriting a replacement process."""
        return current.pid == observed.pid and current.process_identity == observed.process_identity

    @staticmethod
    def _port_available(port: int) -> bool:
        """Fail fast when localhost already owns a requested model-service port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    @staticmethod
    def _signal_process_group(service: Service, signal_type: signal.Signals) -> bool:
        """Signal a verified group; an already-exited process is successfully stopped."""
        try:
            os.killpg(service.pid, signal_type)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise RuntimeError(f"Could not signal service process: {error}") from error
        return True

    def list(self) -> list[Service]:
        with self._locked():
            services, changed = self._read(), False
            for service in services.values():
                if service.status in {"ready", "starting", "stopping"} and not self._owned(service):
                    service.status, service.error, service.pid, changed = (
                        "stopped",
                        "Process is not running",
                        None,
                        True,
                    )
                    service.process_identity = None
            if changed:
                self._write(services)
            return list(services.values())

    def start(
        self,
        identifier: str,
        name: str,
        port: int,
        max_kv_size: int | None = None,
        revision: str | None = None,
        snapshot_path: Path | None = None,
        offline: bool = False,
    ) -> Service:
        if not name.replace("-", "_").replace("_", "a").isalnum():
            raise ValueError("Service names may contain letters, numbers, hyphens, and underscores")
        if max_kv_size is not None and max_kv_size < 128:
            raise ValueError("max_kv_size must be at least 128 when set")
        with self._locked():
            services = self._read()
            if name in services and self._owned(services[name]):
                raise ValueError(f"Service {name!r} is already running")
            if any(item.port == port and self._owned(item) for item in services.values()):
                raise ValueError(f"Port {port} is already managed by a running service")
            if not self._port_available(port):
                raise ValueError(f"Port {port} is already in use on 127.0.0.1")
            model = resolve(identifier)
            if offline and snapshot_path is None:
                raise ValueError("Offline launch requires a verified local snapshot path")
            if snapshot_path is not None and not snapshot_path.is_dir():
                raise ValueError(f"Snapshot path does not exist: {snapshot_path}")
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / f"{name}.log"
            executable = shutil.which("mlx_lm.server")
            command = [executable] if executable else [sys.executable, "-m", "mlx_lm.server"]
            command += [
                "--model",
                str(snapshot_path) if snapshot_path else model.repository,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            if max_kv_size is not None:
                command += ["--max-kv-size", str(max_kv_size)]
            with log_path.open("a") as log:
                log.write(f"\n--- llm-server starting {model.repository} on 127.0.0.1:{port} ---\n")
                environment = os.environ.copy()
                if offline:
                    environment["HF_HUB_OFFLINE"] = "1"
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
            service = Service(
                name=name,
                repository=model.repository,
                port=port,
                pid=process.pid,
                status="starting",
                created_at=time.time(),
                log_file=log_path.name,
                max_kv_size=max_kv_size,
                process_identity=self._identity(process.pid),
                revision=revision,
                snapshot_path=str(snapshot_path) if snapshot_path else None,
                offline=offline,
            )
            services[name] = service
            self._write(services)
            return service

    def get(self, name: str) -> Service:
        services = {service.name: service for service in self.list()}
        if name not in services:
            raise ValueError(f"Unknown service {name!r}")
        return services[name]

    def _set(
        self,
        name: str,
        status: str,
        error: str | None = None,
        observed: Service | None = None,
    ) -> Service:
        with self._locked():
            services = self._read()
            if observed and not self._matches(services[name], observed):
                raise RuntimeError("Service state changed concurrently; retry the operation")
            services[name].status, services[name].error = status, error
            if status == "failed":
                services[name].pid = None
                services[name].process_identity = None
            self._write(services)
            return services[name]

    def mark_ready(self, name: str, timeout: float = 30) -> Service:
        service = self.get(name)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._owned(service):
                return self._set(
                    name, "failed", "MLX-LM exited before becoming ready", observed=service
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{service.port}/v1/models", timeout=1
                ):
                    return self._set(name, "ready", observed=service)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        return self.stop(
            name,
            observed=service,
            final_status="failed",
            final_error=f"Timed out waiting {timeout}s for /v1/models",
        )

    def stop(
        self,
        name: str,
        timeout: float = 10,
        observed: Service | None = None,
        final_status: str = "stopped",
        final_error: str | None = None,
    ) -> Service:
        service = observed or self.get(name)
        if self._alive(service.pid) and not self._owned(service):
            raise RuntimeError("Refusing to signal an unverified process; inspect its PID and logs")
        with self._locked():
            services = self._read()
            current = services.get(name)
            if current is None or not self._matches(current, service):
                raise RuntimeError("Service state changed concurrently; retry the operation")
            current.status = "stopping"
            self._write(services)
        operation_error: RuntimeError | None = None
        try:
            if self._owned(service):
                if not self._owned(service):
                    raise RuntimeError(
                        "Refusing to signal an unverified process; inspect its PID and logs"
                    )
                self._signal_process_group(service, signal.SIGTERM)
                deadline = time.monotonic() + timeout
                while self._alive(service.pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if self._owned(service):
                    self._signal_process_group(service, signal.SIGKILL)
        except RuntimeError as error:
            operation_error = error
        with self._locked():
            services = self._read()
            current = services.get(name)
            if current is None or not self._matches(current, service):
                return current if current is not None else service
            services[name].status, services[name].pid, services[name].error = (
                "failed" if operation_error else final_status,
                None,
                str(operation_error) if operation_error else final_error,
            )
            services[name].process_identity = None
            self._write(services)
            stopped = services[name]
        if operation_error:
            raise operation_error
        return stopped

    def restart(self, name: str) -> Service:
        service = self.get(name)
        self.stop(name, observed=service)
        return self.start(service.repository, name, service.port, service.max_kv_size)

    def logs(self, name: str, lines: int = 80) -> str:
        if not 1 <= lines <= 500:
            raise ValueError("lines must be between 1 and 500")
        service = self.get(name)
        logs_root = self.logs_dir.resolve()
        path = (logs_root / service.log_file).resolve()
        if logs_root not in path.parents:
            raise RuntimeError("Service log path is outside the managed logs directory")
        if not path.exists():
            return "No log entries yet."
        with path.open(errors="replace") as log:
            return "".join(deque(log, maxlen=lines)).rstrip()
