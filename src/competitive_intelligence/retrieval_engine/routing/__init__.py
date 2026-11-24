"""Query routing subsystem for adaptive competitive intelligence retrieval."""

from competitive_intelligence.retrieval_engine.routing.adaptive_retriever import (
    AdaptiveRetriever,
)
from competitive_intelligence.retrieval_engine.routing.errors import (
    AdaptiveRetrievalError,
)
from competitive_intelligence.retrieval_engine.routing.feature_extractor import (
    QueryFeatureExtractor,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    QueryFeatures,
    RouterConfiguration,
    RoutingDecision,
)
from competitive_intelligence.retrieval_engine.routing.router import (
    DeterministicRetrievalRouter,
)

__all__ = [
    "AdaptiveRetrievalError",
    "AdaptiveRetriever",
    "DeterministicRetrievalRouter",
    "QueryFeatureExtractor",
    "QueryFeatures",
    "RouterConfiguration",
    "RoutingDecision",
]
