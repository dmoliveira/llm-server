from fastapi.testclient import TestClient

import llm_server.api as api
from llm_server.api import app, get_manager
from llm_server.runtime import Service, ServiceManager


def test_health_and_catalog_are_available() -> None:
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    models = client.get("/api/v1/models/catalog").json()["models"]
    assert any(model["alias"] == "qwen3-8b" for model in models)
    assert models[0]["capability_confidence"] == "declared"
    assert client.get("/api/v1/host").json()["machine"]
    assert client.get("/api/v1/capacity?model_bytes=1").status_code == 200


def test_invalid_service_name_is_rejected() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/services", json={"model": "qwen3-8b", "name": "../bad", "port": 8080}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Request validation failed"}


def test_log_tail_query_is_bounded() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/services/example/logs?lines=0").status_code == 422
    assert client.get("/api/v1/services/example/logs?lines=501").status_code == 422


def test_model_search_limit_is_bounded() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/models/search?query=qwen&limit=0").status_code == 422
    assert client.get("/api/v1/models/search?query=qwen&limit=-1").status_code == 422
    assert client.get("/api/v1/models/search?query=qwen&limit=51").status_code == 422


def test_model_search_query_is_non_blank() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/models/search?query=").json() == {
        "detail": "Request validation failed"
    }
    assert client.get("/api/v1/models/search?query=%20%20").status_code == 422


def test_unknown_service_is_not_found() -> None:
    client = TestClient(app)
    assert client.post("/api/v1/services/missing/stop").status_code == 404


def test_dashboard_escapes_service_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        api.manager,
        "list",
        lambda: [
            Service(
                name="safe",
                repository='<script>alert("x")</script>',
                port=8080,
                created_at=1,
                log_file="safe.log",
            )
        ],
    )
    response = TestClient(app).get("/")
    assert "<script>" not in response.text
    assert "&lt;script&gt;" in response.text


def test_api_manager_dependency_can_be_overridden(tmp_path) -> None:
    isolated_manager = ServiceManager(tmp_path / "state")
    app.dependency_overrides[get_manager] = lambda: isolated_manager
    try:
        response = TestClient(app).get("/api/v1/status")
        assert response.status_code == 200
        assert response.json() == {"services": []}
    finally:
        app.dependency_overrides.clear()


def test_openapi_exposes_stable_v1_contract_models() -> None:
    schema = TestClient(app).get("/openapi.json").json()
    assert "/api/v1/status" in schema["paths"]
    assert "ServicesResponse" in schema["components"]["schemas"]
    assert {"name", "repository", "port", "status"} <= set(
        schema["components"]["schemas"]["ServiceResponse"]["properties"]
    )
    assert schema["paths"]["/api/v1/services"]["post"]["responses"]["202"]["content"]
    assert schema["paths"]["/api/v1/services/{name}/logs"]["get"]["responses"]["200"]["content"]


def test_corrupt_state_has_a_stable_server_error_contract(tmp_path) -> None:
    isolated_manager = ServiceManager(tmp_path / "state")
    isolated_manager.data_dir.mkdir(parents=True)
    isolated_manager.state_file.write_text("[]")
    app.dependency_overrides[get_manager] = lambda: isolated_manager
    try:
        response = TestClient(app).get("/api/v1/status")
        assert response.status_code == 500
        assert response.json() == {"detail": "Service state is corrupt"}
    finally:
        app.dependency_overrides.clear()


def test_corrupt_versioned_state_has_the_same_server_error_contract(tmp_path) -> None:
    isolated_manager = ServiceManager(tmp_path / "state")
    isolated_manager.data_dir.mkdir(parents=True)
    isolated_manager.state_file.write_text('{"schema_version": 1, "services": []}')
    app.dependency_overrides[get_manager] = lambda: isolated_manager
    try:
        response = TestClient(app).get("/api/v1/status")
        assert response.status_code == 500
        assert response.json() == {"detail": "Service state is corrupt"}
    finally:
        app.dependency_overrides.clear()


def test_future_state_schema_has_the_same_server_error_contract(tmp_path) -> None:
    isolated_manager = ServiceManager(tmp_path / "state")
    isolated_manager.data_dir.mkdir(parents=True)
    isolated_manager.state_file.write_text('{"schema_version": 999, "services": {}}')
    app.dependency_overrides[get_manager] = lambda: isolated_manager
    try:
        response = TestClient(app).get("/api/v1/status")
        assert response.status_code == 500
        assert response.json() == {"detail": "Service state is corrupt"}
    finally:
        app.dependency_overrides.clear()
