"""Global exception handlers mapping domain/system errors to standard HTTP responses."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from competitive_intelligence.application.api.v1.schemas.common import (
    ErrorDetail,
    StandardErrorResponse,
)
from competitive_intelligence.core.errors import (
    AcquisitionError,
    BudgetExceededError,
    CompetitiveIntelligenceError,
    DuplicateEntityError,
    EntityNotFoundError,
    PipelineError,
    RateLimitExceededError,
    RetrievalError,
    ValidationError as DomainValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)


def _get_correlation_id(request: Request) -> str | None:
    return getattr(request.state, "correlation_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    """Register application-wide exception-to-response handlers."""

    @app.exception_handler(EntityNotFoundError)
    async def entity_not_found_handler(request: Request, exc: EntityNotFoundError) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="NOT_FOUND",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=error_resp.model_dump())

    @app.exception_handler(WorkspaceIsolationError)
    async def workspace_isolation_handler(
        request: Request, exc: WorkspaceIsolationError
    ) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="WORKSPACE_ACCESS_DENIED",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content=error_resp.model_dump())

    @app.exception_handler(DuplicateEntityError)
    async def duplicate_entity_handler(request: Request, exc: DuplicateEntityError) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="CONFLICT",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content=error_resp.model_dump())

    @app.exception_handler(DomainValidationError)
    async def domain_validation_handler(
        request: Request, exc: DomainValidationError
    ) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="VALIDATION_ERROR",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_resp.model_dump()
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = [
            ErrorDetail(
                field=".".join(str(loc) for loc in err.get("loc", [])),
                message=err.get("msg", "Invalid request parameter"),
                code=err.get("type"),
            )
            for err in exc.errors()
        ]
        error_resp = StandardErrorResponse(
            error_code="UNPROCESSABLE_ENTITY",
            message="Request body or query validation failed",
            details=details,
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_resp.model_dump()
        )

    @app.exception_handler(BudgetExceededError)
    async def budget_exceeded_handler(request: Request, exc: BudgetExceededError) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="BUDGET_EXCEEDED",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, content=error_resp.model_dump()
        )

    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_handler(request: Request, exc: RateLimitExceededError) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="RATE_LIMITED",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_resp.model_dump(),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(AcquisitionError)
    @app.exception_handler(RetrievalError)
    @app.exception_handler(PipelineError)
    async def upstream_error_handler(
        request: Request, exc: CompetitiveIntelligenceError
    ) -> JSONResponse:
        error_resp = StandardErrorResponse(
            error_code="UPSTREAM_SERVICE_ERROR",
            message=str(exc.message),
            correlation_id=_get_correlation_id(request),
        )
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content=error_resp.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = _get_correlation_id(request)
        logger.error(
            "Unhandled server exception",
            path=request.url.path,
            error=str(exc),
            correlation_id=correlation_id,
            exc_info=True,
        )
        error_resp = StandardErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal server error occurred",
            correlation_id=correlation_id,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_resp.model_dump()
        )
