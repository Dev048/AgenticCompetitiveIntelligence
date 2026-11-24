"""Error classes for the LLM abstraction boundary."""

from competitive_intelligence.core.errors import CompetitiveIntelligenceError


class LLMError(CompetitiveIntelligenceError):
    """Base exception for all LLM client failures."""
    pass


class LLMUnavailableError(LLMError):
    """Raised when LLM service is unreachable or credentials are missing."""
    pass


class LLMResponseMalformedError(LLMError):
    """Raised when LLM output does not match expected Pydantic schema or JSON format."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request exceeds configured execution timeout."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM provider rate limits are exceeded (HTTP 429)."""
    pass


class LLMAuthError(LLMUnavailableError):
    """Raised when LLM authentication credentials are missing or invalid (HTTP 401/403)."""
    pass


class LLMBadRequestError(LLMError):
    """Raised when LLM request is rejected due to invalid parameters (HTTP 400)."""
    pass

