"""BM25 lexical retriever component for Retrieval Engine subsystem.

Coordinates lexical query execution against a BaseLexicalIndex (BM25Index),
applying metadata filtering and returning ranked RetrievalResult models.
"""

from competitive_intelligence.knowledge_pipeline.models import (
    VectorSearchFilter,
)
from competitive_intelligence.retrieval_engine.lexical.interfaces import (
    BaseLexicalIndex,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


class BM25Retriever:
    """BM25 lexical retrieval component.

    Responsibilities:
    - Validates lexical query strings.
    - Delegates lexical search to an injected BaseLexicalIndex.
    - Applies top_k and metadata filtering.
    - Preserves descending BM25 score ordering.
    - Returns clean application-level RetrievalResult items.
    """

    def __init__(
        self,
        index: BaseLexicalIndex,
        default_top_k: int = 10,
    ) -> None:
        """Initialize BM25Retriever.

        Args:
            index: Lexical index instance (e.g. BM25Index).
            default_top_k: Default number of matches to return if omitted in retrieve().

        Raises:
            ValueError: If default_top_k is non-positive.
        """
        if default_top_k <= 0:
            raise ValueError("default_top_k must be a positive integer")

        self._index = index
        self._default_top_k = default_top_k

    @property
    def index(self) -> BaseLexicalIndex:
        """Underlying lexical index instance."""
        return self._index

    @property
    def default_top_k(self) -> int:
        """Configured default top_k."""
        return self._default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Execute synchronous BM25 lexical retrieval.

        Args:
            query: Non-empty search query string.
            top_k: Optional maximum number of matches (defaults to default_top_k).
            filters: Optional metadata filters applied before top_k ranking.

        Returns:
            A list of RetrievalResult items ranked in descending BM25 score order.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        return self._index.search(
            query=query.strip(),
            top_k=effective_top_k,
            filters=filters,
        )

    async def aretrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Execute asynchronous BM25 lexical retrieval.

        Args:
            query: Non-empty search query string.
            top_k: Optional maximum number of matches (defaults to default_top_k).
            filters: Optional metadata filters applied before top_k ranking.

        Returns:
            A list of RetrievalResult items ranked in descending BM25 score order.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
        """
        return self.retrieve(query=query, top_k=top_k, filters=filters)
