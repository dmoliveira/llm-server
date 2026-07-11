from llm_server.admission import admit


def test_admission_aggregates_multiple_services() -> None:
    assert admit(1000, [(400, None), (400, None)]).status == "warning"


def test_admission_stays_unknown_when_model_size_is_unknown() -> None:
    assert admit(1000, [(None, None)]).status == "unknown"
