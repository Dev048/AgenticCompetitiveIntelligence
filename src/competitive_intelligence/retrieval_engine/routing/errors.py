"""Exception types for retrieval routing and adaptive retrieval execution."""

from competitive_intelligence.retrieval_engine.dense.semantic_retriever import (
    RetrievalError,
)


class AdaptiveRetrievalError(RetrievalError):
    """Exception raised when adaptive retrieval routing, pipeline resolution, or execution fails."""

    pass
