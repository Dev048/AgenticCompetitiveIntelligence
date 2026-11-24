"""Core shared components: configuration, structured logging, error types, and budget controls."""

from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.config import (
    ProductionConfigurationError,
    Settings,
    get_settings,
)
from competitive_intelligence.core.errors import (
    AcquisitionError,
    BudgetExceededError,
    CompetitiveIntelligenceError,
    ConfigurationError,
    PipelineError,
    RetrievalError,
    ValidationError,
)
from competitive_intelligence.core.logger import (
    bind_log_context,
    clear_log_context,
    get_log_context,
    get_logger,
    redact_sensitive_value,
)

__all__ = [
    "AcquisitionError",
    "BudgetExceededError",
    "CompetitiveIntelligenceError",
    "ConfigurationError",
    "LLMBudgetTracker",
    "PipelineError",
    "ProductionConfigurationError",
    "RetrievalError",
    "Settings",
    "ValidationError",
    "bind_log_context",
    "clear_log_context",
    "get_log_context",
    "get_logger",
    "get_settings",
    "redact_sensitive_value",
]

