"""Abstract interfaces for Strategic Analysis and Insight Generation."""

from abc import ABC, abstractmethod
from typing import Any

from competitive_intelligence.intelligence.models import (
    InferredSignal,
    Insight,
    IntelligenceEvent,
)


class RawInsightPayload:
    """Type forward reference for raw candidate insight payloads."""
    pass


class BaseStrategicInterpreter(ABC):
    """Abstract contract for interpreting signals and events into candidate raw insight payloads."""

    @abstractmethod
    async def interpret(
        self,
        workspace_id: str,
        company_id: str,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Extract candidate strategic insights from signals and events."""
        pass


class BaseInsightValidator(ABC):
    """Abstract contract for verifying structural grounding and provenance in candidate insights."""

    @abstractmethod
    def validate(
        self,
        payload: Any,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
    ) -> tuple[bool, list[str]]:
        """Validate candidate payload: returns (is_valid, validation_errors)."""
        pass


class BaseStrategicAnalyzer(ABC):
    """Abstract interface for Strategic Analysis & Insight Generation subsystem."""

    @abstractmethod
    async def generate_insights(
        self,
        workspace_id: str,
        company_id: str,
        signals: list[InferredSignal],
        events: list[IntelligenceEvent],
        existing_insights: list[Insight] | None = None,
    ) -> list[Insight]:
        """Synthesize verified, grounded strategic Insights."""
        pass
