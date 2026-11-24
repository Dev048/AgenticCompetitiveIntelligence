"""Core domain and system exceptions."""


class CompetitiveIntelligenceError(Exception):
    """Base exception for all domain errors within the Competitive Intelligence system."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(CompetitiveIntelligenceError):
    """Raised when system or environment configuration is invalid."""


class ValidationError(CompetitiveIntelligenceError):
    """Raised when domain validation fails."""


class AcquisitionError(CompetitiveIntelligenceError):
    """Raised when fetching content from an external source fails."""


class PipelineError(CompetitiveIntelligenceError):
    """Raised when transformation or indexing in the knowledge pipeline fails."""


class RetrievalError(CompetitiveIntelligenceError):
    """Raised when vector or lexical retrieval fails."""


class BudgetExceededError(CompetitiveIntelligenceError):
    """Raised when LLM call count or token budget limit is exceeded."""


class EntityNotFoundError(CompetitiveIntelligenceError):
    """Raised when a requested domain entity is not found."""


class DuplicateEntityError(CompetitiveIntelligenceError):
    """Raised when an entity with duplicate unique constraints already exists."""


class WorkspaceIsolationError(CompetitiveIntelligenceError):
    """Raised when attempting cross-workspace access or workspace boundary violation."""
