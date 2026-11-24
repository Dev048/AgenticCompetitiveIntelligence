"""Application and service layer."""

from typing import Any

from competitive_intelligence.application.interfaces import (
    BaseCompetitorRepository,
    BaseDecisionRepository,
    BaseDocumentRepository,
    BaseIntelligenceRepository,
    BaseSourceRepository,
    BaseUserRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)

__all__ = [
    "BaseUserRepository",
    "BaseWorkspaceRepository",
    "BaseCompetitorRepository",
    "BaseSourceRepository",
    "BaseDocumentRepository",
    "BaseIntelligenceRepository",
    "BaseDecisionRepository",
    "User",
    "Workspace",
    "TrackedCompetitor",
    "WorkspaceService",
    "CompetitorService",
    "SourceService",
    "DocumentService",
    "IntelligenceService",
    "DecisionService",
    "WorkflowExecutionService",
]

_SERVICE_EXPORTS = {
    "WorkspaceService",
    "CompetitorService",
    "SourceService",
    "DocumentService",
    "IntelligenceService",
    "DecisionService",
    "WorkflowExecutionService",
}


def __getattr__(name: str) -> Any:
    if name in _SERVICE_EXPORTS:
        import competitive_intelligence.application.services as services

        return getattr(services, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
