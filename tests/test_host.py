from llm_server.host import host_facts


def test_host_facts_are_serializable() -> None:
    facts = host_facts()
    assert facts.system
    assert facts.machine
