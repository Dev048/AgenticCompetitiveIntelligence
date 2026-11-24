"""Common API schemas: error responses, pagination, and health."""

from typing import Any, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    """Detailed error item."""

    field: str | None = None
    message: str
    code: str | None = None


class StandardErrorResponse(BaseModel):
    """Standardized RFC 7807-style error envelope."""

    error_code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    correlation_id: str | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard pagination wrapper."""

    items: list[T]
    total: int
    page: int = 1
    page_size: int = 50
    has_next: bool = False


class HealthResponse(BaseModel):
    """Service health and connectivity status."""

    status: str = "ok"
    database: str = "connected"
    qdrant: str = "connected"
    environment: str
    version: str = "0.1.0"


class LivenessResponse(BaseModel):
    """Process liveness probe response."""

    status: str = "ok"
    live: bool = True


class ReadinessResponse(BaseModel):
    """Infrastructure readiness probe response."""

    status: str
    database: str
    qdrant: str
    environment: str
    ready: bool
