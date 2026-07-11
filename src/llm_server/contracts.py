"""Versioned public and persisted contracts for the local control plane."""

from __future__ import annotations

from pydantic import BaseModel, Field

STATE_SCHEMA_VERSION = 1


class ModelRef(BaseModel):
    """A requested or resolved model identity; revision is optional until lock support ships."""

    repository: str
    revision: str | None = None


class ServiceSpec(BaseModel):
    """Typed, allowlisted launch intent shared by future profiles and locks."""

    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}$")
    model: ModelRef
    port: int = Field(ge=1024, le=65535)
    max_kv_size: int | None = Field(default=None, ge=128)


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ServicesResponse(BaseModel):
    services: list[ServiceResponse]


class ServiceResponse(BaseModel):
    name: str
    repository: str
    port: int
    pid: int | None = None
    status: str
    created_at: float
    log_file: str
    max_kv_size: int | None = None
    process_identity: str | None = None
    error: str | None = None
    revision: str | None = None
    snapshot_path: str | None = None
    offline: bool = False
    provenance: str = "unlocked"


class LogsResponse(BaseModel):
    logs: str


class CatalogModelResponse(BaseModel):
    alias: str
    repository: str
    provider: str
    family: str
    size: str
    quantization: str
    context: str
    note: str
    modalities: str = "text"
    tool_calling: str = "unknown"
    structured_output: str = "unknown"
    capability_confidence: str = "declared"


class DownloadedModelResponse(BaseModel):
    repository: str
    size_bytes: int
    revisions: int


class SearchModelResponse(BaseModel):
    repository: str
    downloads: int
    likes: int
    updated: str


class CatalogResponse(BaseModel):
    models: list[CatalogModelResponse]


class DownloadedModelsResponse(BaseModel):
    models: list[DownloadedModelResponse]


class SearchModelsResponse(BaseModel):
    models: list[SearchModelResponse]
