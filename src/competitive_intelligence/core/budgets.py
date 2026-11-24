"""LLM call and token budget management for task-level cost and rate controls."""

from dataclasses import dataclass, field
from typing import Any

from competitive_intelligence.core.errors import BudgetExceededError
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.core.metrics import track_llm_budget_exhaustion

logger = get_logger(__name__)


@dataclass
class BudgetUsage:
    """Current usage state for a task execution."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retrieval_hops: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.prompt_tokens + self.completion_tokens


class LLMBudgetTracker:
    """Tracks and enforces limits on LLM invocations and token consumption per task."""

    def __init__(
        self,
        task_id: str,
        max_llm_calls: int = 10,
        max_retrieval_hops: int = 3,
        max_tokens: int = 30000,
    ) -> None:
        self.task_id = task_id
        self.max_llm_calls = max_llm_calls
        self.max_retrieval_hops = max_retrieval_hops
        self.max_tokens = max_tokens
        self.usage = BudgetUsage()

    def record_llm_call(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record an LLM call and verify limits."""
        self.usage.llm_calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens

        logger.debug(
            "Recorded LLM call",
            task_id=self.task_id,
            llm_calls=self.usage.llm_calls,
            max_llm_calls=self.max_llm_calls,
            total_tokens=self.usage.total_tokens,
            max_tokens=self.max_tokens,
        )

        if self.usage.llm_calls > self.max_llm_calls:
            track_llm_budget_exhaustion("llm_calls")
            raise BudgetExceededError(
                f"Task '{self.task_id}' exceeded max LLM calls limit ({self.usage.llm_calls}/{self.max_llm_calls})"
            )

        if self.usage.total_tokens > self.max_tokens:
            track_llm_budget_exhaustion("token_budget")
            raise BudgetExceededError(
                f"Task '{self.task_id}' exceeded max token budget ({self.usage.total_tokens}/{self.max_tokens})"
            )

    def record_retrieval_hop(self) -> None:
        """Record a retrieval hop and verify limits."""
        self.usage.retrieval_hops += 1
        logger.debug(
            "Recorded retrieval hop",
            task_id=self.task_id,
            retrieval_hops=self.usage.retrieval_hops,
            max_retrieval_hops=self.max_retrieval_hops,
        )

        if self.usage.retrieval_hops > self.max_retrieval_hops:
            track_llm_budget_exhaustion("retrieval_hops")
            raise BudgetExceededError(
                f"Task '{self.task_id}' exceeded max retrieval hops ({self.usage.retrieval_hops}/{self.max_retrieval_hops})"
            )

