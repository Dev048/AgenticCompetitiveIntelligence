"""CLI entry point for running the Continuous Monitoring scheduler as a standalone worker process.

Usage:
    python -m competitive_intelligence.monitoring.worker_cli
"""

import asyncio
import os
import signal
import sys
from typing import Any

from competitive_intelligence.application.services.monitoring_service import (
    SourceMonitoringService,
)
from competitive_intelligence.application.services.workflow_service import (
    WorkflowExecutionService,
)
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.core.logger import configure_logging, get_logger
from competitive_intelligence.infrastructure.database.repositories import (
    SqlCompetitorRepository,
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlMonitoringLeaseRepository,
    SqlSourceRepository,
    SqlWorkflowStateStore,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import get_db_manager
from competitive_intelligence.monitoring.scheduler import MonitoringScheduler
from competitive_intelligence.monitoring.worker import MonitoringWorker
from competitive_intelligence.orchestration.orchestrator import (
    DeterministicWorkflowOrchestrator,
)
from competitive_intelligence.orchestration.stages import (
    DecisionSupportStage,
    EventAnalysisStage,
    InsightGenerationStage,
    KnowledgeProcessingStage,
    RetrievalStage,
    SignalDetectionStage,
)

logger = get_logger(__name__)


async def run_worker() -> None:
    """Initialize and execute the continuous monitoring scheduler."""
    settings = get_settings()
    configure_logging(log_level=settings.log_level, log_format=settings.log_format)

    logger.info(
        "Initializing Continuous Monitoring Worker",
        environment=settings.environment,
        poll_interval=settings.monitoring_poll_interval_seconds,
        batch_size=settings.monitoring_batch_size,
        max_concurrency=settings.monitoring_max_concurrency,
    )

    # Validate production configuration if running in production
    settings.validate_production_configuration(raise_exception=True)

    db_manager = get_db_manager(settings)
    sessionmaker = db_manager.sessionmaker

    # Wire worker dependencies
    async with sessionmaker() as session:
        # Repositories for scheduler and monitoring service
        source_repo = SqlSourceRepository(session)
        lease_repo = SqlMonitoringLeaseRepository(session)
        workspace_repo = SqlWorkspaceRepository(session)
        competitor_repo = SqlCompetitorRepository(session)
        state_store = SqlWorkflowStateStore(sessionmaker)
        doc_repo = SqlDocumentRepository(sessionmaker)
        intel_repo = SqlIntelligenceRepository(sessionmaker)
        dec_repo = SqlDecisionRepository(sessionmaker)

        # Build production stages
        stages = {
            # Knowledge Processing Stage
            KnowledgeProcessingStage(): KnowledgeProcessingStage(),
            # Retrieval Stage
            RetrievalStage(): RetrievalStage(),
            # Event Analysis Stage
            EventAnalysisStage(intelligence_repo=intel_repo): EventAnalysisStage(intelligence_repo=intel_repo),
            # Signal Detection Stage
            SignalDetectionStage(intelligence_repo=intel_repo): SignalDetectionStage(intelligence_repo=intel_repo),
            # Insight Generation Stage
            InsightGenerationStage(intelligence_repo=intel_repo): InsightGenerationStage(intelligence_repo=intel_repo),
            # Decision Support Stage
            DecisionSupportStage(decision_repo=dec_repo): DecisionSupportStage(decision_repo=dec_repo),
        }

        orchestrator = DeterministicWorkflowOrchestrator()
        workflow_service = WorkflowExecutionService(
            orchestrator=orchestrator,
            state_store=state_store,
            workspace_repo=workspace_repo,
            competitor_repo=competitor_repo,
        )

        monitoring_service = SourceMonitoringService(
            source_repo=source_repo,
            lease_repo=lease_repo,
            workflow_service=workflow_service,
            workspace_repo=workspace_repo,
        )

        worker = MonitoringWorker(
            monitoring_service=monitoring_service,
            lease_repo=lease_repo,
            heartbeat_interval_seconds=settings.monitoring_heartbeat_interval_seconds,
            lease_extension_seconds=settings.monitoring_lease_timeout_seconds,
        )

        scheduler = MonitoringScheduler(
            source_repo=source_repo,
            lease_repo=lease_repo,
            worker=worker,
            poll_interval_seconds=settings.monitoring_poll_interval_seconds,
            batch_size=settings.monitoring_batch_size,
            max_concurrency=settings.monitoring_max_concurrency,
            lease_timeout_seconds=settings.monitoring_lease_timeout_seconds,
        )

        stop_event = asyncio.Event()

        def handle_shutdown(sig: int, frame: Any) -> None:
            logger.info("Shutdown signal received, stopping scheduler...", signal=sig)
            stop_event.set()

        # Register signals if available on OS
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_shutdown)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, handle_shutdown)

        logger.info("Starting monitoring scheduler loop...")
        await scheduler.start()

        try:
            while not stop_event.is_set():
                await asyncio.sleep(1.0)
        finally:
            logger.info("Stopping monitoring scheduler...")
            await scheduler.stop()
            await db_manager.close()
            logger.info("Monitoring worker shutdown complete.")


def main() -> None:
    """Synchronous entrypoint."""
    try:
        asyncio.run(run_worker())
    except (KeyboardInterrupt, SystemExit):
        sys.exit(0)


if __name__ == "__main__":
    main()
