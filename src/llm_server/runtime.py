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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, Field

from .catalog import resolve


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
            return {
                name: Service.model_validate(item)
                for name, item in json.loads(self.state_file.read_text()).items()
            }
        except (json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"Service state is corrupt: {self.state_file}") from error

    def _write(self, services: dict[str, Service]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({name: item.model_dump() for name, item in services.items()}, indent=2)
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
        self, identifier: str, name: str, port: int, max_kv_size: int | None = None
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
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.logs_dir / f"{name}.log"
            executable = shutil.which("mlx_lm.server")
            command = [executable] if executable else [sys.executable, "-m", "mlx_lm.server"]
            command += ["--model", model.repository, "--host", "127.0.0.1", "--port", str(port)]
            if max_kv_size is not None:
                command += ["--max-kv-size", str(max_kv_size)]
            with log_path.open("a") as log:
                log.write(f"\n--- llm-server starting {model.repository} on 127.0.0.1:{port} ---\n")
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
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
        if self._owned(service):
            if not self._owned(service):
                raise RuntimeError(
                    "Refusing to signal an unverified process; inspect its PID and logs"
                )
            os.killpg(service.pid, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while self._alive(service.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if self._owned(service):
                os.killpg(service.pid, signal.SIGKILL)
        with self._locked():
            services = self._read()
            current = services.get(name)
            if current is None or not self._matches(current, service):
                return current if current is not None else service
            services[name].status, services[name].pid, services[name].error = (
                final_status,
                None,
                final_error,
            )
            services[name].process_identity = None
            self._write(services)
            return services[name]

    def restart(self, name: str) -> Service:
        service = self.get(name)
        self.stop(name, observed=service)
        return self.start(service.repository, name, service.port, service.max_kv_size)

    def logs(self, name: str, lines: int = 80) -> str:
        if not 1 <= lines <= 500:
            raise ValueError("lines must be between 1 and 500")
        service = self.get(name)
        path = self.logs_dir / service.log_file
        return (
            "No log entries yet."
            if not path.exists()
            else "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
        )
