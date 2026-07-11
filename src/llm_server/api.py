"""Local HTTP control plane and a tiny operations dashboard."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .catalog import cached_models, models, search
from .runtime import ServiceManager

app = FastAPI(title="LLM Server", version=__version__)
manager = ServiceManager()


class StartRequest(BaseModel):
    model: str
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
    port: int = Field(ge=1024, le=65535)
    max_kv_size: int | None = Field(default=None, ge=128)


def safe(action):
    try:
        return action()
    except (ValueError, RuntimeError) as error:
        detail = str(error)
        status_code = 404 if detail.startswith("Unknown service") else 400
        if "changed concurrently" in detail or "unverified process" in detail:
            status_code = 409
        raise HTTPException(status_code=status_code, detail=detail) from error


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    rows = "".join(
        f"<tr><td>{s.name}</td><td>{s.repository}</td><td><b>{s.status}</b></td><td>{s.port}</td></tr>"
        for s in manager.list()
    ) or (
        "<tr><td colspan=4>No managed services. Start one from the CLI or "
        "<a href='/docs'>API docs</a>.</td></tr>"
    )
    return (
        "<!doctype html><title>LLM Server</title><style>"
        "body{background:#0d1117;color:#e6edf3;font:16px system-ui;max-width:960px;"
        "margin:5rem auto;padding:0 2rem}.tag{color:#7ee787}table{width:100%;"
        "border-collapse:collapse;background:#161b22}td,th{padding:14px;"
        "border-bottom:1px solid #30363d;text-align:left}a{color:#58a6ff}</style>"
        "<h1>⚡ LLM Server <span class=tag>LOCAL</span></h1><p>Mac-first MLX model "
        "control plane · <a href='/docs'>API docs</a> · <a href='/api/v1/status'>"
        f"JSON status</a></p><table><tr><th>Service</th><th>Model</th><th>Status</th>"
        f"<th>Port</th></tr>{rows}</table>"
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/models/catalog")
def catalog() -> dict[str, object]:
    return {"models": models()}


@app.get("/api/v1/models/downloaded")
def downloaded() -> dict[str, object]:
    return {"models": cached_models()}


@app.get("/api/v1/models/search")
def model_search(query: str, limit: int = 10) -> dict[str, object]:
    return safe(lambda: {"models": search(query, limit)})


@app.get("/api/v1/status")
def status() -> dict[str, object]:
    return {"services": [s.model_dump() for s in manager.list()]}


@app.post("/api/v1/services", status_code=202)
def start(request: StartRequest) -> dict[str, object]:
    return safe(
        lambda: manager.start(
            request.model, request.name, request.port, request.max_kv_size
        ).model_dump()
    )


@app.post("/api/v1/services/{name}/ready")
def ready(name: str) -> dict[str, object]:
    return safe(lambda: manager.mark_ready(name).model_dump())


@app.post("/api/v1/services/{name}/stop")
def stop(name: str) -> dict[str, object]:
    return safe(lambda: manager.stop(name).model_dump())


@app.post("/api/v1/services/{name}/restart", status_code=202)
def restart(name: str) -> dict[str, object]:
    return safe(lambda: manager.restart(name).model_dump())


@app.get("/api/v1/services/{name}/logs")
def logs(name: str, lines: int = Query(default=80, ge=1, le=500)) -> dict[str, str]:
    return safe(lambda: {"logs": manager.logs(name, lines)})
