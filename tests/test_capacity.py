from llm_server.capacity import plan_capacity


def test_capacity_plan_is_conservative() -> None:
    plan = plan_capacity(1000, 700, None)
    assert plan.status == "fits"
    assert plan.headroom_bytes == 250


def test_capacity_plan_warns_when_estimate_exceeds_headroom() -> None:
    assert plan_capacity(1000, 900, None).status == "warning"


def test_capacity_plan_never_claims_fit_for_unknown_inputs() -> None:
    assert plan_capacity(None, 10, None).status == "unknown"
