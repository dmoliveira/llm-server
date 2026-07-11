"""Local HTTP control plane and a tiny operations dashboard."""

from __future__ import annotations

from html import escape

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .capacity import CapacityPlan, plan_capacity
from .catalog import cached_models, models, search
from .contracts import (
    CatalogResponse,
    DownloadedModelsResponse,
    ErrorResponse,
    HealthResponse,
    LogsResponse,
    SearchModelsResponse,
    ServiceResponse,
    ServicesResponse,
)
from .host import HostFacts, host_facts
from .runtime import ServiceManager, StateCorruptError

app = FastAPI(title="LLM Server", version=__version__)
manager = ServiceManager()
ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


def get_manager() -> ServiceManager:
    """Dependency seam used by API contract tests and future app factories."""
    return manager


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    """Keep v1 client errors stable without exposing framework-specific validation details."""
    return JSONResponse(
        status_code=422, content=ErrorResponse(detail="Request validation failed").model_dump()
    )


@app.exception_handler(StateCorruptError)
async def corrupt_state_error(_: Request, __: StateCorruptError) -> JSONResponse:
    """Do not leak local state paths while returning a stable server-side failure."""
    return JSONResponse(
        status_code=500, content=ErrorResponse(detail="Service state is corrupt").model_dump()
    )


class StartRequest(BaseModel):
    model: str
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
    port: int = Field(ge=1024, le=65535)
    max_kv_size: int | None = Field(default=None, ge=128)


def safe(action):
    try:
        return action()
    except StateCorruptError:
        raise
    except (ValueError, RuntimeError) as error:
        detail = str(error)
        status_code = 404 if detail.startswith("Unknown service") else 400
        if "changed concurrently" in detail or "unverified process" in detail:
            status_code = 409
        raise HTTPException(status_code=status_code, detail=detail) from error


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard(service_manager: ServiceManager = Depends(get_manager)) -> str:
    rows = "".join(
        f"<tr><td>{escape(s.name)}</td><td>{escape(s.repository)}</td>"
        f"<td><b>{escape(s.status)}</b></td><td>{s.port}</td></tr>"
        for s in service_manager.list()
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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/host", response_model=HostFacts)
def host() -> HostFacts:
    """Return local machine facts used by capacity planning; no remote telemetry is emitted."""
    return host_facts()


@app.get("/api/v1/capacity", response_model=CapacityPlan)
def capacity(
    model_bytes: int | None = Query(default=None, ge=0),
    max_kv_size: int | None = Query(default=None, ge=0),
) -> CapacityPlan:
    """Return a conservative local unified-memory recommendation without starting a service."""
    return plan_capacity(host_facts().memory_bytes, model_bytes, max_kv_size)


@app.get("/api/v1/models/catalog", response_model=CatalogResponse)
def catalog() -> CatalogResponse:
    return {"models": models()}


@app.get("/api/v1/models/downloaded", response_model=DownloadedModelsResponse)
def downloaded() -> DownloadedModelsResponse:
    return {"models": cached_models()}


@app.get("/api/v1/models/search", response_model=SearchModelsResponse, responses=ERROR_RESPONSES)
def model_search(
    query: str = Query(min_length=1), limit: int = Query(default=10, ge=1, le=50)
) -> SearchModelsResponse:
    if not query.strip():
        raise HTTPException(status_code=422, detail="Request validation failed")
    return safe(lambda: {"models": search(query, limit)})


@app.get(
    "/api/v1/status", response_model=ServicesResponse, responses={500: {"model": ErrorResponse}}
)
def status(service_manager: ServiceManager = Depends(get_manager)) -> ServicesResponse:
    return {"services": [s.model_dump() for s in service_manager.list()]}


@app.post(
    "/api/v1/services", status_code=202, response_model=ServiceResponse, responses=ERROR_RESPONSES
)
def start(
    request: StartRequest, service_manager: ServiceManager = Depends(get_manager)
) -> ServiceResponse:
    return safe(
        lambda: service_manager.start(
            request.model, request.name, request.port, request.max_kv_size
        ).model_dump()
    )


@app.post(
    "/api/v1/services/{name}/ready", response_model=ServiceResponse, responses=ERROR_RESPONSES
)
def ready(name: str, service_manager: ServiceManager = Depends(get_manager)) -> ServiceResponse:
    return safe(lambda: service_manager.mark_ready(name).model_dump())


@app.post("/api/v1/services/{name}/stop", response_model=ServiceResponse, responses=ERROR_RESPONSES)
def stop(name: str, service_manager: ServiceManager = Depends(get_manager)) -> ServiceResponse:
    return safe(lambda: service_manager.stop(name).model_dump())


@app.post(
    "/api/v1/services/{name}/restart",
    status_code=202,
    response_model=ServiceResponse,
    responses=ERROR_RESPONSES,
)
def restart(name: str, service_manager: ServiceManager = Depends(get_manager)) -> ServiceResponse:
    return safe(lambda: service_manager.restart(name).model_dump())


@app.get("/api/v1/services/{name}/logs", response_model=LogsResponse, responses=ERROR_RESPONSES)
def logs(
    name: str,
    lines: int = Query(default=80, ge=1, le=500),
    service_manager: ServiceManager = Depends(get_manager),
) -> LogsResponse:
    return safe(lambda: {"logs": service_manager.logs(name, lines)})
