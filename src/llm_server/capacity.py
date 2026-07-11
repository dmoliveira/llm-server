"""Conservative, explainable unified-memory capacity estimates."""

from __future__ import annotations

from pydantic import BaseModel


class CapacityPlan(BaseModel):
    status: str
    estimated_bytes: int | None
    available_bytes: int | None
    headroom_bytes: int | None
    reason: str


def plan_capacity(
    total_memory_bytes: int | None,
    model_bytes: int | None,
    max_kv_size: int | None,
    headroom_fraction: float = 0.25,
) -> CapacityPlan:
    """Estimate model + conservative KV demand; unknown inputs never produce a false fit."""
    if total_memory_bytes is None or model_bytes is None:
        return CapacityPlan(
            status="unknown",
            estimated_bytes=None,
            available_bytes=total_memory_bytes,
            headroom_bytes=None,
            reason="Model size or host memory is unavailable",
        )
    kv_bytes = (max_kv_size or 0) * 2048
    estimated = model_bytes + kv_bytes
    headroom = int(total_memory_bytes * headroom_fraction)
    available = total_memory_bytes - headroom
    status = "fits" if estimated <= available else "warning"
    return CapacityPlan(
        status=status,
        estimated_bytes=estimated,
        available_bytes=available,
        headroom_bytes=headroom,
        reason="Estimate includes configured KV capacity and reserved unified-memory headroom",
    )
