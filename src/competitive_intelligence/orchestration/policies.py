"""Policies governing retry behavior, execution budgets, and stage constraints."""

from pydantic import BaseModel, Field

from competitive_intelligence.orchestration.models import ErrorCategory


class RetryPolicy(BaseModel):
    """Configurable retry policy with exponential backoff and category filtering."""

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 2.0
    retryable_categories: list[ErrorCategory] = Field(
        default_factory=lambda: [ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT]
    )

    def is_retryable(self, category: ErrorCategory, attempt: int) -> bool:
        """Return True if the error category is configured as retryable and max attempts not exceeded."""
        return category in self.retryable_categories and attempt < self.max_attempts

    def get_backoff(self, attempt: int) -> float:
        """Compute exponential backoff delay in seconds."""
        delay = self.initial_backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff_seconds)


class ExecutionBudget(BaseModel):
    """Resource budgets and timeout limits propagated across workflow stages."""

    stage_timeout_seconds: float = 30.0
    workflow_timeout_seconds: float = 180.0
    max_retrieval_candidates: int = 50
    reranking_top_k: int = 10
    max_events_to_analyze: int = 100
