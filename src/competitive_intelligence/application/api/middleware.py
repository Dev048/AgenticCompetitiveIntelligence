"""HTTP middleware for request correlation and execution timing."""

import time
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from competitive_intelligence.core.logger import (
    bind_log_context,
    clear_log_context,
    get_logger,
)
from competitive_intelligence.core.metrics import track_http_request

logger = get_logger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Injects and propagates X-Correlation-ID header across requests and responses,
    binding it to contextvars for structured logging."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
        request.state.correlation_id = correlation_id
        
        # Check for optional workspace-id header
        workspace_id = request.headers.get("X-Workspace-ID")
        bind_log_context(correlation_id=correlation_id, workspace_id=workspace_id)

        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            clear_log_context()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs HTTP request method, path, status, and duration in milliseconds,
    and increments Prometheus HTTP metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_s = time.perf_counter() - start_time
        duration_ms = duration_s * 1000.0

        correlation_id = getattr(request.state, "correlation_id", None)
        
        # Track metrics safely (fail-safe)
        track_http_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration_s,
        )

        logger.info(
            "HTTP Request Processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            correlation_id=correlation_id,
        )
        return response

