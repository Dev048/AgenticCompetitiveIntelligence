"""FastAPI application factory and lifecycle configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response

from competitive_intelligence.application.api.error_handlers import (
    register_exception_handlers,
)
from competitive_intelligence.application.api.middleware import (
    CorrelationIdMiddleware,
    RequestLoggingMiddleware,
)
from competitive_intelligence.application.api.v1.endpoints.health import (
    router as health_router,
)
from competitive_intelligence.application.api.v1.router import api_v1_router
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.core.logger import configure_logging, get_logger
from competitive_intelligence.core.metrics import (
    generate_prometheus_metrics,
    get_metrics_content_type,
)
from competitive_intelligence.infrastructure.database.session import get_db_manager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling startup validation, infrastructure, and shutdown hooks."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger.info("Starting Competitive Intelligence API service", environment=settings.environment)

    # 1. Validate production configuration at startup (fails fast if invalid)
    settings.validate_production_configuration(raise_exception=True)

    # 2. Verify or initialize database schema
    db_manager = get_db_manager()
    if settings.environment in ("development", "test"):
        try:
            await db_manager.create_tables()
            logger.info("Database schema initialized successfully")
        except Exception as err:
            logger.warning("Database schema initialization encountered notice", error=str(err))
    else:
        # In production/staging, verify database connection
        try:
            from sqlalchemy import text
            async with db_manager.session() as session:
                await session.execute(text("SELECT 1"))
            logger.info("Production database connectivity verified successfully")
        except Exception as err:
            logger.error("Production database connection failed", error=str(err))
            raise

    # 3. Verify vector store collection lifecycle
    try:
        from competitive_intelligence.application.api.deps import get_vector_store
        vector_store = get_vector_store(settings)
        vector_store.ensure_collection()
        logger.info(
            "Vector store collection verified successfully",
            collection=settings.qdrant_collection_name,
        )
    except Exception as err:
        if settings.environment in ("production", "staging"):
            logger.error("Failed to initialize vector store collection in production", error=str(err))
            raise
        logger.warning("Vector store collection notice", error=str(err))

    yield

    # Shutdown hooks
    try:
        await db_manager.close()
        logger.info("Database connection pool closed successfully")
    except Exception as err:
        logger.warning("Error during database shutdown", error=str(err))

    try:
        from competitive_intelligence.application.api.deps import reset_resource_caches

        reset_resource_caches()
    except Exception as err:
        logger.warning("Error releasing cached resources", error=str(err))

    logger.info("Competitive Intelligence API service stopped")


def create_app() -> FastAPI:
    """Application factory building and configuring the FastAPI instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.project_name,
        description=(
            "Production-oriented evidence-backed Competitive Intelligence Monitoring Platform "
            "providing multi-workspace context, grounded provenance, decision support, and operational workflows."
        ),
        version="0.1.0",
        openapi_url=f"{settings.api_v1_str}/openapi.json",
        docs_url=f"{settings.api_v1_str}/docs",
        redoc_url=f"{settings.api_v1_str}/redoc",
        lifespan=lifespan,
    )

    # Middlewares (Executed in reverse addition order)
    origins = settings.cors_origins if settings.cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # Register global exception handlers
    register_exception_handlers(app)

    # Mount routers: root health check and versioned API router
    app.include_router(health_router, tags=["Health"])
    app.include_router(api_v1_router, prefix=settings.api_v1_str)

    # Mount Prometheus metrics endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint(request: Request) -> Response:
        """Prometheus metrics scrape endpoint."""
        if not settings.metrics_enabled:
            return Response("Metrics disabled", status_code=404)
        if settings.metrics_token:
            auth_header = request.headers.get("Authorization", "")
            query_token = request.query_params.get("token", "")
            expected = f"Bearer {settings.metrics_token}"
            if auth_header != expected and query_token != settings.metrics_token:
                return Response("Unauthorized", status_code=401)
        return Response(
            content=generate_prometheus_metrics(),
            media_type=get_metrics_content_type(),
        )

    # Mount UI static files
    from pathlib import Path
    from starlette.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


# Default ASGI app instance
app = create_app()
