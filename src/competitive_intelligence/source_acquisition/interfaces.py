"""Abstract interfaces for Source Acquisition and Monitoring subsystem.

Defines protocol contracts for acquisition clients, fallback policies, monitoring, and source persistence.
"""

from abc import ABC, abstractmethod
from typing import Any

from competitive_intelligence.source_acquisition.models import AcquisitionResult, Source


class AcquisitionClient(ABC):
    """Abstract interface for acquiring data from an external source."""

    @abstractmethod
    async def acquire(self, source: Source) -> AcquisitionResult:
        """Acquire content snapshot from the given source.

        Args:
            source: The Source domain model defining target location and constraints.

        Returns:
            AcquisitionResult representing the outcome of this acquisition attempt.
        """
        pass


class FallbackPolicy(ABC):
    """Abstract policy governing acquisition fallback decisions."""

    @abstractmethod
    def is_recoverable(self, result: AcquisitionResult) -> bool:
        """Determine whether an acquisition failure is eligible for fallback."""
        pass

    @abstractmethod
    def should_fallback(self, result: AcquisitionResult) -> bool:
        """Determine whether the orchestrator should attempt fallback to the next provider."""
        pass


# Backwards compatibility alias
BaseAcquisitionClient = AcquisitionClient


class BaseMonitor(ABC):
    """Abstract interface for monitoring scheduler."""

    @abstractmethod
    async def check_source(self, source: Source) -> Any:
        """Execute a scheduled check on a specific source."""
        pass

    @abstractmethod
    async def get_due_sources(self, limit: int = 50) -> list[Source]:
        """Retrieve sources that are due for acquisition based on monitoring policy."""
        pass


class BaseSourceRepository(ABC):
    """Abstract interface for persisting and querying Sources."""

    @abstractmethod
    async def get_source(self, source_id: str) -> Source | None:
        """Get source by ID."""
        pass

    @abstractmethod
    async def save_source(self, source: Source) -> None:
        """Create or update source."""
        pass
