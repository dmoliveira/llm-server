"""Small, failure-tolerant Apple-Silicon host facts for future capacity planning."""

from __future__ import annotations

import platform
import subprocess

from pydantic import BaseModel


class HostFacts(BaseModel):
    system: str
    machine: str
    processor: str
    memory_bytes: int | None = None


def host_facts() -> HostFacts:
    memory_bytes: int | None = None
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        try:
            memory_bytes = int(result.stdout.strip())
        except ValueError:
            pass
    return HostFacts(
        system=platform.system(),
        machine=platform.machine(),
        processor=platform.processor(),
        memory_bytes=memory_bytes,
    )
