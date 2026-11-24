"""Abstract interface for LLM client integration in Agentic RAG."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from competitive_intelligence.core.budgets import LLMBudgetTracker

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract interface for calling LLMs with budget tracking and structured output validation."""

    @property
    def provider_name(self) -> str:
        """Name of the underlying LLM provider."""
        return "custom"

    @property
    def model_name(self) -> str:
        """Name of the configured model."""
        return "custom-model"


    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        """Generate structured response validated against a Pydantic model.
        
        Args:
            prompt: User/query prompt text.
            response_schema: Pydantic model class to validate and parse output into.
            system_prompt: Optional instructions for system prompt.
            temperature: Sampling temperature (default 0.0 for deterministic output).
            budget_tracker: Optional task-level budget tracker to record token usage.
            
        Returns:
            Validated instance of response_schema.
            
        Raises:
            LLMUnavailableError: If LLM service cannot be reached.
            LLMResponseMalformedError: If response fails schema validation.
            LLMTimeoutError: If LLM request times out.
            BudgetExceededError: If token or call limits are exceeded.
        """
        pass

    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        """Generate unstructured text completion.
        
        Args:
            prompt: User/query prompt text.
            system_prompt: Optional instructions for system prompt.
            temperature: Sampling temperature.
            budget_tracker: Optional task-level budget tracker to record token usage.
            
        Returns:
            Raw string response.
        """
        pass
