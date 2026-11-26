"""External acquisition provider adapters (Crawl4AI, Agent Reach, RSS, REST APIs).

Concrete adapters implementing AcquisitionClient to bridge external acquisition
tooling with the application domain.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from competitive_intelligence.source_acquisition.providers.crawl4ai_client import (
        Crawl4AIClient,
    )

__all__ = [
    "Crawl4AIClient",
]


def __getattr__(name: str) -> Any:
    # Crawl4AI pulls in Playwright/Chromium. Import it only when a crawler is actually
    # requested, so the API (and the read-only demo image) can run without the
    # "acquisition" extra installed.
    if name == "Crawl4AIClient":
        from competitive_intelligence.source_acquisition.providers.crawl4ai_client import (
            Crawl4AIClient,
        )

        return Crawl4AIClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
