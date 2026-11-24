"""Dense semantic retriever component for Retrieval Engine subsystem.

Coordinates query embedding via BaseEmbeddingProvider and vector similarity search
via BaseVectorStore using domain SearchConfiguration and VectorSearchFilter models.
"""

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseEmbeddingProvider,
    BaseVectorStore,
)
from competitive_intelligence.knowledge_pipeline.models import (
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


class RetrievalError(Exception):
    """Base exception for retrieval engine operations."""


class SemanticRetrievalError(RetrievalError):
    """Exception raised when dense semantic retrieval fails."""


class SemanticRetriever:
    """Dense semantic retrieval component.

    Responsibilities:
    - Validates query strings.
    - Uses an injected BaseEmbeddingProvider to generate query dense vectors.
    - Delegates similarity search to an injected BaseVectorStore.
    - Applies SearchConfiguration (top_k, hnsw_ef, exact_search) and VectorSearchFilter.
    - Preserves result ordering and exact float similarity scores.
    - Encapsulates errors into domain RetrievalError exceptions.
    - Operates strictly with domain abstractions (zero database/ML library leakage).
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        default_search_config: SearchConfiguration | None = None,
    ) -> None:
        """Initialize SemanticRetriever.

        Args:
            embedding_provider: Provider abstraction for computing query embeddings.
            vector_store: Vector store abstraction for similarity search.
            default_search_config: Optional fallback search configuration.
        """
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_search_config = default_search_config or SearchConfiguration()

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Underlying embedding provider abstraction."""
        return self._embedding_provider

    @property
    def vector_store(self) -> BaseVectorStore:
        """Underlying vector store abstraction."""
        return self._vector_store

    @property
    def default_search_config(self) -> SearchConfiguration:
        """Default search configuration."""
        return self._default_search_config

    def retrieve(
        self,
        query: str,
        search_config: SearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Execute synchronous dense semantic retrieval.

        Args:
            query: Non-empty search query string.
            search_config: Optional search parameters (overrides default_search_config).
            filters: Optional metadata payload filters.

        Returns:
            A list of VectorSearchResult items ranked by descending similarity score.

        Raises:
            ValueError: If query is empty or whitespace-only.
            SemanticRetrievalError: If embedding generation or vector store search fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")

        clean_query = query.strip()

        try:
            query_vector = self._embedding_provider.embed_query(clean_query)
        except Exception as exc:
            raise SemanticRetrievalError(
                f"Embedding generation failed for query '{clean_query}': {exc}"
            ) from exc

        effective_config = search_config or self._default_search_config

        try:
            results = self._vector_store.search(
                query_vector=query_vector,
                filters=filters,
                search_config=effective_config,
            )
        except Exception as exc:
            raise SemanticRetrievalError(
                f"Vector search failed for query '{clean_query}': {exc}"
            ) from exc

        return results

    async def aretrieve(
        self,
        query: str,
        search_config: SearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Execute asynchronous dense semantic retrieval.

        Args:
            query: Non-empty search query string.
            search_config: Optional search parameters (overrides default_search_config).
            filters: Optional metadata payload filters.

        Returns:
            A list of VectorSearchResult items ranked by descending similarity score.

        Raises:
            ValueError: If query is empty or whitespace-only.
            SemanticRetrievalError: If embedding generation or vector store search fails.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")

        clean_query = query.strip()

        try:
            query_vector = await self._embedding_provider.aembed_query(clean_query)
        except Exception as exc:
            raise SemanticRetrievalError(
                f"Async embedding generation failed for query '{clean_query}': {exc}"
            ) from exc

        effective_config = search_config or self._default_search_config

        try:
            results = await self._vector_store.asearch(
                query_vector=query_vector,
                filters=filters,
                search_config=effective_config,
            )
        except Exception as exc:
            raise SemanticRetrievalError(
                f"Async vector search failed for query '{clean_query}': {exc}"
            ) from exc

        return results


class SemanticRetrieverAdapter(BaseRetriever):
    """Adapter wrapping SemanticRetriever to conform to the BaseRetriever interface."""

    def __init__(self, semantic_retriever: SemanticRetriever) -> None:
        self.semantic_retriever = semantic_retriever

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        company_id = (
            query.metadata_filter.company_ids[0]
            if query.metadata_filter.company_ids
            else None
        )
        filters = VectorSearchFilter(
            workspace_id=query.metadata_filter.workspace_id,
            company_id=company_id,
        )
        search_cfg = SearchConfiguration(top_k=query.top_k)
        results = await self.semantic_retriever.aretrieve(
            query=query.query_text,
            search_config=search_cfg,
            filters=filters,
        )
        return [RetrievalResult.from_vector_search_result(r) for r in results]
