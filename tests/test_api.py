from fastapi.testclient import TestClient

from llm_server.api import app


def test_health_and_catalog_are_available() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/api/v1/models/catalog").json()["models"]
    assert any(model["alias"] == "qwen3-8b" for model in models)


def test_invalid_service_name_is_rejected() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/services", json={"model": "qwen3-8b", "name": "../bad", "port": 8080}
    )
    assert response.status_code == 422


def test_log_tail_query_is_bounded() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/services/example/logs?lines=0").status_code == 422
    assert client.get("/api/v1/services/example/logs?lines=501").status_code == 422


def test_unknown_service_is_not_found() -> None:
    client = TestClient(app)
    assert client.post("/api/v1/services/missing/stop").status_code == 404
