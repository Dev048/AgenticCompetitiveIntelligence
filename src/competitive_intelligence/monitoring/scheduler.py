"""Continuous monitoring scheduler polling for due sources and dispatching to workers."""

import asyncio
from datetime import datetime, timezone
import os
import socket
from typing import Any
from uuid import uuid4

from competitive_intelligence.application.interfaces import BaseSourceRepository
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.core.metrics import set_active_monitoring_leases
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
    BaseMonitoringScheduler,
    BaseMonitoringWorker,
)

logger = get_logger(__name__)


class MonitoringScheduler(BaseMonitoringScheduler):
    """Periodic scheduler that queries due sources, claims atomic leases, and dispatches to workers."""

    def __init__(
        self,
        source_repo: BaseSourceRepository,
        lease_repo: BaseMonitoringLeaseRepository,
        worker: BaseMonitoringWorker,
        worker_id: str | None = None,
        poll_interval_seconds: float = 15.0,
        batch_size: int = 10,
        max_concurrency: int = 5,
        lease_timeout_seconds: int = 300,
    ) -> None:
        self._source_repo = source_repo
        self._lease_repo = lease_repo
        self._worker = worker
        self._worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{str(uuid4())[:8]}"
        self._poll_interval_seconds = poll_interval_seconds
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._lease_timeout_seconds = lease_timeout_seconds

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._running = False
        self._loop_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def is_running(self) -> bool:
        return self._running

    async def run_once(self) -> int:
        """Execute a single polling cycle.

        1. Reclaims expired leases.
        2. Retrieves due sources up to batch size.
        3. Atomically claims each due source.
        4. Dispatches claimed sources to workers under concurrency limits.
        """
        now = datetime.now(timezone.utc)
        claimed_count = 0

        try:
            # 1. Reclaim expired leases
            reclaimed = await self._lease_repo.reclaim_expired_leases(as_of=now)
            if reclaimed > 0:
                logger.info("LEASE_RECLAIMED", count=reclaimed, worker_id=self._worker_id)

            # 2. Query due sources
            due_sources = await self._source_repo.get_due_sources(
                limit=self._batch_size, as_of=now
            )
            if not due_sources:
                return 0

            # 3. Atomically claim and dispatch
            for source in due_sources:
                lease = await self._lease_repo.claim_due_source(
                    source_id=source.source_id,
                    lease_owner=self._worker_id,
                    lease_duration_seconds=self._lease_timeout_seconds,
                )

                if lease is None:
                    # Leased by another concurrent worker; skip
                    continue

                claimed_count += 1
                corr_id = str(uuid4())

                # 4. Dispatch to worker under semaphore bound
                task = asyncio.create_task(
                    self._dispatch_with_concurrency(
                        source_id=source.source_id,
                        lease_owner=self._worker_id,
                        correlation_id=corr_id,
                    )
                )
                self._active_tasks.add(task)
                set_active_monitoring_leases(len(self._active_tasks))
                def _on_task_done(t: asyncio.Task[Any]) -> None:
                    self._active_tasks.discard(t)
                    set_active_monitoring_leases(len(self._active_tasks))
                task.add_done_callback(_on_task_done)

        except Exception as exc:
            logger.error(
                "Error during MonitoringScheduler run_once cycle",
                worker_id=self._worker_id,
                error=str(exc),
            )

        set_active_monitoring_leases(len(self._active_tasks))
        return claimed_count


    async def _dispatch_with_concurrency(
        self,
        source_id: str,
        lease_owner: str,
        correlation_id: str,
    ) -> None:
        """Execute monitoring worker task within bounded concurrency semaphore."""
        async with self._semaphore:
            try:
                await self._worker.execute_monitoring_task(
                    source_id=source_id,
                    lease_owner=lease_owner,
                    correlation_id=correlation_id,
                )
            except Exception as exc:
                logger.error(
                    "Error executing dispatched monitoring task",
                    source_id=source_id,
                    error=str(exc),
                    correlation_id=correlation_id,
                )

    async def _scheduler_loop(self) -> None:
        """Main periodic polling loop."""
        logger.info(
            "MonitoringScheduler started",
            worker_id=self._worker_id,
            interval=self._poll_interval_seconds,
            batch_size=self._batch_size,
            max_concurrency=self._max_concurrency,
        )
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("Error in scheduler loop", error=str(exc))

            try:
                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                break

    async def start(self) -> None:
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the background scheduler loop and await active worker tasks."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):
                pass
            self._loop_task = None

        if self._active_tasks:
            # Await all in-flight worker tasks
            await asyncio.gather(*list(self._active_tasks), return_exceptions=True)

        logger.info("MonitoringScheduler stopped", worker_id=self._worker_id)
