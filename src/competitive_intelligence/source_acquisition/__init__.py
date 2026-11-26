"""Source acquisition and monitoring subsystem."""

from typing import TYPE_CHECKING, Any

from competitive_intelligence.source_acquisition.interfaces import (
    AcquisitionClient,
    BaseAcquisitionClient,
    BaseMonitor,
    BaseSourceRepository,
    FallbackPolicy,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    DataSensitivity,
    FetchStatus,
    GovernanceState,
    LifecycleState,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    RateLimitConstraints,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToolChain,
    ToSStatus,
)
from competitive_intelligence.source_acquisition.orchestrator import (
    AcquisitionOrchestrator,
    DefaultFallbackPolicy,
    ProviderNotRegisteredError,
    ProviderRegistry,
)

if TYPE_CHECKING:
    from competitive_intelligence.source_acquisition.providers.crawl4ai_client import (
        Crawl4AIClient,
    )

__all__ = [
    "AccessPattern",
    "AcquisitionClient",
    "AcquisitionConfig",
    "AcquisitionMethod",
    "AcquisitionOrchestrator",
    "AcquisitionResult",
    "AcquisitionStatus",
    "BaseAcquisitionClient",
    "BaseMonitor",
    "BaseSourceRepository",
    "ContentType",
    "Crawl4AIClient",
    "DataSensitivity",
    "DefaultFallbackPolicy",
    "FallbackPolicy",
    "FetchStatus",
    "GovernanceState",
    "LifecycleState",
    "LifecycleStatus",
    "MonitoringPolicy",
    "PlatformType",
    "ProviderNotRegisteredError",
    "ProviderRegistry",
    "RateLimitConstraints",
    "Source",
    "SourceCategory",
    "SourceClassification",
    "SourceGovernance",
    "SourceIdentity",
    "SourceLifecycle",
    "SourceLocation",
    "SourceReliability",
    "SourceTier",
    "ToSStatus",
    "ToolChain",
]


def __getattr__(name: str) -> Any:
    # Crawl4AI pulls in Playwright/Chromium. Import it only when a crawler is actually
    # requested, so the API (and slim images without Chromium) can run without the
    # "acquisition" extra installed.
    if name == "Crawl4AIClient":
        from competitive_intelligence.source_acquisition.providers.crawl4ai_client import (
            Crawl4AIClient,
        )

        return Crawl4AIClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
