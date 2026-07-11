"""Pure admission decisions for multi-service unified-memory safety."""

from __future__ import annotations

from .capacity import CapacityPlan, plan_capacity


def admit(
    total_memory_bytes: int | None, service_estimates: list[tuple[int | None, int | None]]
) -> CapacityPlan:
    """Aggregate declared model/KV estimates; unknown inputs remain non-authoritative."""
    if any(model is None for model, _ in service_estimates):
        return plan_capacity(total_memory_bytes, None, None)
    model_bytes = sum(model or 0 for model, _ in service_estimates)
    kv_size = sum(kv or 0 for _, kv in service_estimates)
    return plan_capacity(total_memory_bytes, model_bytes, kv_size)
