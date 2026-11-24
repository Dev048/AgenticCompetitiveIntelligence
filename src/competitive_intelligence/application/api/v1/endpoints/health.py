"""Health, Liveness, and Readiness endpoints."""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from competitive_intelligence.application.api.deps import get_vector_store
from competitive_intelligence.application.api.v1.schemas.common import (
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
)
from competitive_intelligence.core.config import Settings, get_settings
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.knowledge_pipeline.interfaces import BaseVectorStore

router = APIRouter(tags=["Health"])


async def _check_database(manager: DatabaseSessionManager) -> str:
    """Execute lightweight query to check relational database connectivity."""
    try:
        async with manager.session() as session:
            await session.execute(text("SELECT 1"))
        return "connected"
    except Exception:
        return "unhealthy"


def _check_qdrant(store: BaseVectorStore) -> str:
    """Verify vector database connectivity and collection state."""
    try:
        store.ensure_collection()
        return "connected"
    except Exception:
        return "unhealthy"


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings),
    db_manager: DatabaseSessionManager = Depends(get_db_manager),
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> HealthResponse:
    """Comprehensive health check returning database and vector store connectivity."""
    db_status = await _check_database(db_manager)
    qdrant_status = _check_qdrant(vector_store)

    is_ok = db_status == "connected" and qdrant_status == "connected"
    return HealthResponse(
        status="ok" if is_ok else "degraded",
        database=db_status,
        qdrant=qdrant_status,
        environment=settings.environment,
    )


@router.get("/health/live", response_model=LivenessResponse)
async def liveness_probe() -> LivenessResponse:
    """Process liveness probe indicating the HTTP process is running."""
    return LivenessResponse(status="ok", live=True)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_probe(
    response: Response,
    settings: Settings = Depends(get_settings),
    db_manager: DatabaseSessionManager = Depends(get_db_manager),
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> ReadinessResponse:
    """Readiness probe checking PostgreSQL and Qdrant infrastructure connectivity.

    Returns HTTP 200 when ready, HTTP 503 when degraded or disconnected.
    """
    db_status = await _check_database(db_manager)
    qdrant_status = _check_qdrant(vector_store)

    is_ready = db_status == "connected" and qdrant_status == "connected"
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if is_ready else "unready",
        database=db_status,
        qdrant=qdrant_status,
        environment=settings.environment,
        ready=is_ready,
    )
