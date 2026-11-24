"""Abstract interfaces for Decision Support and Recommendation subsystem."""

from abc import ABC, abstractmethod
from typing import Any

from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionOption,
    Recommendation,
)
from competitive_intelligence.intelligence.models import (
    Insight,
    IntelligenceEvent,
)


class BaseOptionGenerator(ABC):
    """Abstract contract for generating candidate decision options from insights and context."""

    @abstractmethod
    async def generate_options(
        self,
        context: DecisionContext,
        insights: list[Insight],
        events: list[IntelligenceEvent],
    ) -> list[DecisionOption]:
        """Generate feasible candidate decision options."""
        pass


class BaseRecommendationValidator(ABC):
    """Abstract contract for structurally validating recommendation payloads."""

    @abstractmethod
    def validate(
        self,
        payload: Any,
        context: DecisionContext,
        options: list[DecisionOption],
        insights: list[Insight],
        events: list[IntelligenceEvent],
    ) -> tuple[bool, list[str]]:
        """Validate recommendation payload: returns (is_valid, validation_errors)."""
        pass


class BaseDecisionAnalyzer(ABC):
    """Abstract interface for the Decision Support & Recommendation subsystem."""

    @abstractmethod
    async def analyze_and_recommend(
        self,
        context: DecisionContext,
        insights: list[Insight],
        events: list[IntelligenceEvent],
        existing_recommendations: list[Recommendation] | None = None,
    ) -> tuple[list[DecisionOption], Recommendation | None]:
        """Evaluate context and insights to generate candidate options and an optimal recommendation."""
        pass
