"""External acquisition provider adapters (Crawl4AI, Agent Reach, RSS, REST APIs).

Concrete adapters implementing AcquisitionClient to bridge external acquisition
tooling with the application domain.
"""

from competitive_intelligence.source_acquisition.providers.crawl4ai_client import (
    Crawl4AIClient,
)

__all__ = [
    "Crawl4AIClient",
]
