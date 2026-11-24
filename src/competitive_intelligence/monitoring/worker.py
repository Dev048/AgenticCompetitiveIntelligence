"""Monitoring worker executing single-source acquisition and heartbeat renewal."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
    BaseMonitoringWorker,
    BaseSourceMonitoringService,
)
from competitive_intelligence.orchestration.models import WorkflowRun

logger = get_logger(__name__)


class MonitoringWorker(BaseMonitoringWorker):
    """Executes claimed monitoring work with periodic background heartbeat lease renewal."""

    def __init__(
        self,
        monitoring_service: BaseSourceMonitoringService,
        lease_repo: BaseMonitoringLeaseRepository,
        heartbeat_interval_seconds: float = 60.0,
        lease_extension_seconds: int = 300,
    ) -> None:
        self._monitoring_service = monitoring_service
        self._lease_repo = lease_repo
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._lease_extension_seconds = lease_extension_seconds

    async def _heartbeat_loop(self, source_id: str, lease_owner: str) -> None:
        """Background coroutine periodically extending lease expiration while acquisition is active."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_seconds)
                renewed = await self._lease_repo.renew_source_lease(
                    source_id=source_id,
                    lease_owner=lease_owner,
                    extension_seconds=self._lease_extension_seconds,
                )
                if renewed:
                    logger.debug(
                        "LEASE_RENEWED",
                        source_id=source_id,
                        lease_owner=lease_owner,
                        extended_seconds=self._lease_extension_seconds,
                    )
                else:
                    logger.warning(
                        "Failed to renew lease - lease may have expired or been reclaimed",
                        source_id=source_id,
                        lease_owner=lease_owner,
                    )
                    break
        except asyncio.CancelledError:
            # Clean exit on cancellation
            pass
        except Exception as exc:
            logger.error(
                "Unexpected error in monitoring heartbeat loop",
                source_id=source_id,
                lease_owner=lease_owner,
                error=str(exc),
            )

    async def execute_monitoring_task(
        self,
        source_id: str,
        lease_owner: str,
        correlation_id: str | None = None,
    ) -> WorkflowRun | None:
        """Execute source monitoring with robust error containment and heartbeat lease renewal."""
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(source_id=source_id, lease_owner=lease_owner)
        )

        try:
            return await self._monitoring_service.monitor_source(
                source_id=source_id,
                lease_owner=lease_owner,
                correlation_id=correlation_id,
            )
        except Exception as exc:
            logger.error(
                "MonitoringWorker encountered unhandled failure during execution",
                source_id=source_id,
                lease_owner=lease_owner,
                error=str(exc),
                correlation_id=correlation_id,
            )
            # Ensure lease is released on crash/error
            try:
                await self._lease_repo.release_source_lease(source_id, lease_owner)
            except Exception:
                pass
            return None
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
