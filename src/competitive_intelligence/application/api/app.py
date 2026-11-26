"""FastAPI application factory and lifecycle configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

from competitive_intelligence.application.api.demo_guard import DemoReadOnlyMiddleware
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
from competitive_intelligence.core.config import Settings, get_settings
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
    if settings.environment in ("development", "test", "demo"):
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

    # 4. Public demo: seed the read-only workspace from recorded LLM fixtures. Idempotent:
    #    the demo image ships pre-seeded data, so this only skips completed runs; it seeds
    #    from scratch only if the baked data is missing.
    if settings.environment == "demo" and settings.demo_seed_on_startup:
        await _seed_demo(settings)

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


async def _seed_demo(settings: Settings) -> None:
    from competitive_intelligence.demo.seed import load_replay_client, seed_demo_workspace

    fixtures = Path(settings.demo_llm_fixtures_path) if settings.demo_llm_fixtures_path else None
    try:
        summary = await seed_demo_workspace(llm_client=load_replay_client(fixtures))
    except Exception as err:
        # Never take the demo down because seeding failed: health and UI still come up and
        # the failure is visible in the logs.
        logger.error("Demo seeding failed", error=str(err), exc_info=True)
        return
    logger.info(
        "Demo workspace ready",
        workspace_id=summary.workspace_id,
        steps_run=summary.steps_run,
        steps_skipped=summary.steps_skipped,
        steps_failed=summary.steps_failed,
        llm_replay=summary.llm.model_dump(),
        duration_seconds=summary.duration_seconds,
    )


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
    app.add_middleware(DemoReadOnlyMiddleware)
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
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.exists():
        app.mount("/", _RevalidatingStaticFiles(directory=str(static_dir), html=True), name="ui")

    return app


class _RevalidatingStaticFiles(StaticFiles):
    """Static UI files that browsers must revalidate (ETag) on every load.

    The UI is unbundled ES modules with no content hashes, so heuristic caching would
    keep serving stale modules after a deploy.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# Default ASGI app instance
app = create_app()
