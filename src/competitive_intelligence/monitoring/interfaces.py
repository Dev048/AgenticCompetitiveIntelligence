"""Abstract interfaces for Continuous Monitoring subsystem."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from competitive_intelligence.monitoring.models import SourceMonitoringLease
from competitive_intelligence.orchestration.models import (
    WorkflowRun,
    WorkflowStageType,
)


class BaseSourceMonitoringService(ABC):
    """Abstract application service for executing single-source monitoring."""

    @abstractmethod
    async def monitor_source(
        self,
        source_id: str,
        lease_owner: str,
        correlation_id: str | None = None,
        target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
    ) -> WorkflowRun | None:
        """Execute single-source monitoring, acquisition, and pipeline dispatch."""
        pass


class BaseMonitoringLeaseRepository(ABC):
    """Abstract repository for durable source monitoring leases."""

    @abstractmethod
    async def claim_due_source(
        self,
        source_id: str,
        lease_owner: str,
        lease_duration_seconds: int = 300,
    ) -> SourceMonitoringLease | None:
        """Atomically claim a lease on a source if unleased or if previous lease expired.

        Returns:
            SourceMonitoringLease if successfully claimed, None if already leased by another active worker.
        """
        pass

    @abstractmethod
    async def renew_source_lease(
        self,
        source_id: str,
        lease_owner: str,
        extension_seconds: int = 300,
    ) -> bool:
        """Renew heartbeat and extend lease expiration for the active owner."""
        pass

    @abstractmethod
    async def release_source_lease(
        self,
        source_id: str,
        lease_owner: str,
    ) -> bool:
        """Release the lease held by the owner upon completion or failure."""
        pass

    @abstractmethod
    async def get_active_lease(
        self,
        source_id: str,
    ) -> SourceMonitoringLease | None:
        """Retrieve active lease on source if one exists."""
        pass

    @abstractmethod
    async def reclaim_expired_leases(
        self,
        as_of: datetime | None = None,
    ) -> int:
        """Clean up or reclaim expired leases whose heartbeat/lease has elapsed."""
        pass


class BaseMonitoringWorker(ABC):
    """Abstract worker interface for executing source monitoring tasks."""

    @abstractmethod
    async def execute_monitoring_task(
        self,
        source_id: str,
        lease_owner: str,
        correlation_id: str | None = None,
    ) -> WorkflowRun | None:
        """Execute acquisition and pipeline dispatch for a claimed source."""
        pass


class BaseMonitoringScheduler(ABC):
    """Abstract interface for the continuous monitoring polling scheduler."""

    @abstractmethod
    async def start(self) -> None:
        """Start background polling scheduler."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop background polling scheduler."""
        pass

    @abstractmethod
    async def run_once(self) -> int:
        """Execute a single polling iteration, returning the number of claimed sources."""
        pass
