"""Unit tests for SemanticRetriever in Retrieval Engine subsystem.

Verifies query validation, embedding coordination, vector search delegation,
parameter propagation (SearchConfiguration, VectorSearchFilter), score & order preservation,
error handling, async retrieval, and strict dependency isolation.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
import uuid

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseEmbeddingProvider,
    BaseVectorStore,
    MockEmbeddingProvider,
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
)
from competitive_intelligence.retrieval_engine import (
    RetrievalError,
    SemanticRetrievalError,
    SemanticRetriever,
)


# ==============================================================================
# Mocks & Test Spies
# ==============================================================================


class SpyVectorStore(BaseVectorStore):
    """Spy implementation of BaseVectorStore to record calls and return canned results."""

    def __init__(self, return_results: list[VectorSearchResult] | None = None) -> None:
        self.return_results = return_results if return_results is not None else []
        self.search_calls: list[dict[str, Any]] = []

    def ensure_collection(self) -> None:
        pass

    def upsert_entries(self, entries: Sequence[Any]) -> None:
        pass

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
        search_config: SearchConfiguration | None = None,
    ) -> list[VectorSearchResult]:
        self.search_calls.append({
            "query_vector": list(query_vector),
            "top_k": top_k,
            "filters": filters,
            "search_config": search_config,
        })
        return self.return_results

    def get_by_id(self, chunk_id: str) -> VectorSearchResult | None:
        return None

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        pass

    def delete_by_filter(self, filters: VectorSearchFilter) -> None:
        pass


class FaultyEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider that raises an exception during query embedding."""

    @property
    def model_name(self) -> str:
        return "faulty-model"

    @property
    def model_version(self) -> str | None:
        return None

    @property
    def dimensions(self) -> int | None:
        return 16

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("Embedding service unavailable or timed out")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("Embedding service unavailable or timed out")


class FaultyVectorStore(BaseVectorStore):
    """Vector store that raises an exception during search."""

    def ensure_collection(self) -> None:
        pass

    def upsert_entries(self, entries: Sequence[Any]) -> None:
        pass

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
        search_config: SearchConfiguration | None = None,
    ) -> list[VectorSearchResult]:
        raise RuntimeError("Vector database node unreachable")

    def get_by_id(self, chunk_id: str) -> VectorSearchResult | None:
        return None

    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        pass

    def delete_by_filter(self, filters: VectorSearchFilter) -> None:
        pass


def create_sample_results() -> list[VectorSearchResult]:
    """Helper to create dummy scored search results."""
    return [
        VectorSearchResult(
            chunk_id=str(uuid.uuid4()),
            score=0.925,
            text="Competitor launched Enterprise SLA.",
            document_id="doc_1",
            version_id="ver_1",
            company_id="comp_acme",
            workspace_id="ws_main",
            chunk_level=0,
            collected_at=datetime.now(timezone.utc),
        ),
        VectorSearchResult(
            chunk_id=str(uuid.uuid4()),
            score=0.810,
            text="SLA uptime target is 99.99%.",
            document_id="doc_1",
            version_id="ver_1",
            company_id="comp_acme",
            workspace_id="ws_main",
            chunk_level=1,
            parent_chunk_id="parent_uuid",
            collected_at=datetime.now(timezone.utc),
        ),
    ]


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_semantic_retriever_initialization() -> None:
    """Test initializing SemanticRetriever with dependencies."""
    embedder = MockEmbeddingProvider(dimension=16)
    store = SpyVectorStore()
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    assert retriever.embedding_provider is embedder
    assert retriever.vector_store is store
    assert retriever.default_search_config.top_k == 10


def test_successful_retrieval_flow() -> None:
    """Test complete retrieve flow: query embedding -> vector store search -> results."""
    embedder = MockEmbeddingProvider(dimension=16)
    canned_results = create_sample_results()
    store = SpyVectorStore(return_results=canned_results)
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    query_text = "What is the new SLA uptime?"
    search_cfg = SearchConfiguration(top_k=5, hnsw_ef=64, exact_search=True)
    filters = VectorSearchFilter(company_id="comp_acme", workspace_id="ws_main")

    results = retriever.retrieve(query=query_text, search_config=search_cfg, filters=filters)

    # 1. Output verification
    assert len(results) == 2
    assert results == canned_results
    assert results[0].score == 0.925
    assert results[1].score == 0.810

    # 2. VectorStore call verification
    assert len(store.search_calls) == 1
    call = store.search_calls[0]
    expected_query_vec = embedder.embed_query(query_text)
    assert call["query_vector"] == expected_query_vec
    assert call["search_config"] == search_cfg
    assert call["filters"] == filters


def test_default_search_config_used_when_omitted() -> None:
    """Test that default_search_config is passed when search_config is None."""
    embedder = MockEmbeddingProvider(dimension=16)
    store = SpyVectorStore()
    default_cfg = SearchConfiguration(top_k=15, hnsw_ef=32)
    retriever = SemanticRetriever(
        embedding_provider=embedder,
        vector_store=store,
        default_search_config=default_cfg,
    )

    _ = retriever.retrieve("pricing strategy")

    assert len(store.search_calls) == 1
    assert store.search_calls[0]["search_config"] == default_cfg


@pytest.mark.parametrize("empty_query", ["", "   ", "\t\r\n  "])
def test_empty_query_raises_value_error(empty_query: str) -> None:
    """Test that empty or whitespace-only query raises ValueError."""
    embedder = MockEmbeddingProvider(dimension=16)
    store = SpyVectorStore()
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    with pytest.raises(ValueError, match="Query string cannot be empty or whitespace-only"):
        retriever.retrieve(empty_query)


def test_empty_results_handling() -> None:
    """Test that empty vector store results return [] cleanly."""
    embedder = MockEmbeddingProvider(dimension=16)
    store = SpyVectorStore(return_results=[])
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    results = retriever.retrieve("unknown query with no matches")
    assert results == []


def test_result_ordering_and_scores_strictly_preserved() -> None:
    """Test that scores and ranking order are untouched by the retriever."""
    embedder = MockEmbeddingProvider(dimension=16)
    raw_results = [
        VectorSearchResult(
            chunk_id="chunk_1", score=0.999, text="t1", document_id="d1", version_id="v1"
        ),
        VectorSearchResult(
            chunk_id="chunk_2", score=0.543, text="t2", document_id="d1", version_id="v1"
        ),
        VectorSearchResult(
            chunk_id="chunk_3", score=0.120, text="t3", document_id="d1", version_id="v1"
        ),
    ]
    store = SpyVectorStore(return_results=raw_results)
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    results = retriever.retrieve("ranking test")

    assert len(results) == 3
    assert [r.chunk_id for r in results] == ["chunk_1", "chunk_2", "chunk_3"]
    assert [r.score for r in results] == [0.999, 0.543, 0.120]


def test_embedding_failure_translated_to_retrieval_error() -> None:
    """Test that embedding provider exception is wrapped in SemanticRetrievalError."""
    faulty_embedder = FaultyEmbeddingProvider()
    store = SpyVectorStore()
    retriever = SemanticRetriever(embedding_provider=faulty_embedder, vector_store=store)

    with pytest.raises(SemanticRetrievalError, match="Embedding generation failed for query"):
        retriever.retrieve("failing query")


def test_vector_store_failure_translated_to_retrieval_error() -> None:
    """Test that vector store exception is wrapped in SemanticRetrievalError."""
    embedder = MockEmbeddingProvider(dimension=16)
    faulty_store = FaultyVectorStore()
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=faulty_store)

    with pytest.raises(SemanticRetrievalError, match="Vector search failed for query"):
        retriever.retrieve("search failure query")


@pytest.mark.anyio
async def test_async_retrieve_operation() -> None:
    """Test async aretrieve method."""
    embedder = MockEmbeddingProvider(dimension=16)
    canned_results = create_sample_results()
    store = SpyVectorStore(return_results=canned_results)
    retriever = SemanticRetriever(embedding_provider=embedder, vector_store=store)

    results = await retriever.aretrieve("async search query")

    assert len(results) == 2
    assert results[0].chunk_id == canned_results[0].chunk_id


def test_dependency_isolation() -> None:
    """Verify that SemanticRetriever does not import qdrant_client, sentence_transformers, torch."""
    import competitive_intelligence.retrieval_engine.dense.semantic_retriever as ret_mod

    mod_file = getattr(ret_mod, "__file__", "")
    assert mod_file != ""
    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "qdrant_client" not in code
    assert "sentence_transformers" not in code
    assert "SentenceTransformer" not in code
    assert "torch" not in code
    assert "bge_m3" not in code.lower()
