"""Unit tests for HybridRetriever using Reciprocal Rank Fusion (RRF).

Verifies disjoint candidate merging, overlapping duplicate consolidation,
exact mathematical RRF scoring, rank-based fusion independence from raw scores,
candidate pool sizing, configurable rrf_k, deterministic tie-breaking,
empty candidate handling, failure propagation, async execution, and library isolation.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
import uuid

import pytest

from competitive_intelligence.knowledge_pipeline import (
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
)
from competitive_intelligence.retrieval_engine import (
    HybridRetrievalError,
    HybridRetriever,
    HybridSearchConfiguration,
    RetrievalResult,
)


# ==============================================================================
# Mocks & Test Spies
# ==============================================================================


class FakeSemanticRetriever:
    """Mock SemanticRetriever returning pre-configured VectorSearchResult candidates."""

    def __init__(self, candidates: list[VectorSearchResult] | None = None) -> None:
        self.candidates = candidates if candidates is not None else []
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        search_config: SearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        self.calls.append({
            "query": query,
            "search_config": search_config,
            "filters": filters,
        })
        return self.candidates

    async def aretrieve(
        self,
        query: str,
        search_config: SearchConfiguration | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        return self.retrieve(query=query, search_config=search_config, filters=filters)


class FakeBM25Retriever:
    """Mock BM25Retriever returning pre-configured RetrievalResult candidates."""

    def __init__(self, candidates: list[RetrievalResult] | None = None) -> None:
        self.candidates = candidates if candidates is not None else []
        self.calls: list[dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
        })
        return self.candidates

    async def aretrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        return self.retrieve(query=query, top_k=top_k, filters=filters)


class CrashingRetriever:
    """Retriever mock that simulates unexpected internal crash."""

    def retrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("Service failure or timeout")

    async def aretrieve(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("Service failure or timeout")


def create_vector_result(chunk_id: str, score: float, text: str = "Semantic text") -> VectorSearchResult:
    """Helper to construct a mock VectorSearchResult."""
    return VectorSearchResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id="doc_101",
        version_id="ver_101",
        company_id="comp_alpha",
        workspace_id="ws_intel",
        collected_at=datetime.now(timezone.utc),
        metadata={"tag": "semantic"},
    )


def create_bm25_result(chunk_id: str, score: float, text: str = "BM25 text") -> RetrievalResult:
    """Helper to construct a mock BM25 RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id="doc_101",
        version_id="ver_101",
        company_id="comp_alpha",
        workspace_id="ws_intel",
        metadata={"tag": "lexical"},
    )


# ==============================================================================
# Unit Tests
# ==============================================================================


def test_hybrid_search_configuration_defaults_and_validation() -> None:
    """Test HybridSearchConfiguration defaults and pool calculation."""
    cfg = HybridSearchConfiguration()
    assert cfg.final_top_k == 10
    assert cfg.fusion_candidate_k == 50
    assert cfg.rrf_k == 60
    assert cfg.effective_semantic_k == 250
    assert cfg.effective_lexical_k == 250

    # Custom explicit candidate_k
    custom_k = HybridSearchConfiguration(final_top_k=5, candidate_k=30, rrf_k=40)
    assert custom_k.final_top_k == 5
    assert custom_k.effective_semantic_k == 30
    assert custom_k.effective_lexical_k == 30

    # Custom split pools
    split_k = HybridSearchConfiguration(
        final_top_k=8,
        semantic_candidate_k=40,
        lexical_candidate_k=25,
    )
    assert split_k.effective_semantic_k == 40
    assert split_k.effective_lexical_k == 25

    # Validation errors
    with pytest.raises(ValueError, match="Value must be a positive integer"):
        HybridSearchConfiguration(final_top_k=0)
    with pytest.raises(ValueError, match="Value must be a positive integer"):
        HybridSearchConfiguration(rrf_k=-1)
    with pytest.raises(ValueError, match="Candidate pool size must be a positive integer"):
        HybridSearchConfiguration(candidate_k=0)


def test_disjoint_candidates_rrf_scoring() -> None:
    """Test RRF calculation when semantic and BM25 candidates have no overlap."""
    sem_cands = [
        create_vector_result("chunk_sem_1", score=0.95),  # rank 1 -> 1/(60+1) = 1/61
        create_vector_result("chunk_sem_2", score=0.85),  # rank 2 -> 1/(60+2) = 1/62
    ]
    bm25_cands = [
        create_bm25_result("chunk_lex_1", score=12.5),    # rank 1 -> 1/(60+1) = 1/61
        create_bm25_result("chunk_lex_2", score=8.2),     # rank 2 -> 1/(60+2) = 1/62
    ]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = hybrid.retrieve("hybrid test query", config=HybridSearchConfiguration(final_top_k=4, rrf_k=60))

    assert len(results) == 4
    # Rank 1 semantic and Rank 1 BM25 both have score 1/61 (~0.01639344)
    # Tie-breaker sorts by chunk_id ("chunk_lex_1" < "chunk_sem_1")
    assert results[0].chunk_id == "chunk_lex_1"
    assert results[1].chunk_id == "chunk_sem_1"
    assert pytest.approx(results[0].score, rel=1e-6) == 1.0 / 61
    assert pytest.approx(results[1].score, rel=1e-6) == 1.0 / 61

    # Rank 2 items have score 1/62
    assert results[2].chunk_id == "chunk_lex_2"
    assert results[3].chunk_id == "chunk_sem_2"
    assert pytest.approx(results[2].score, rel=1e-6) == 1.0 / 62
    assert pytest.approx(results[3].score, rel=1e-6) == 1.0 / 62


def test_overlapping_chunk_combines_rrf_contributions() -> None:
    """Test that a chunk appearing in both retrievers gets summed RRF contributions."""
    # Chunk A is rank 1 in semantic and rank 1 in BM25
    # Chunk B is rank 2 in semantic only
    # Chunk C is rank 2 in BM25 only
    sem_cands = [
        create_vector_result("chunk_A", score=0.99),  # rank 1
        create_vector_result("chunk_B", score=0.70),  # rank 2
    ]
    bm25_cands = [
        create_bm25_result("chunk_A", score=15.0),   # rank 1
        create_bm25_result("chunk_C", score=6.0),    # rank 2
    ]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = hybrid.retrieve("overlapping test", config=HybridSearchConfiguration(final_top_k=5, rrf_k=60))

    assert len(results) == 3
    # Top result MUST be chunk_A with combined RRF: 1/61 + 1/61 = 2/61 (~0.03278688)
    assert results[0].chunk_id == "chunk_A"
    expected_score_a = (1.0 / 61) + (1.0 / 61)
    assert pytest.approx(results[0].score, rel=1e-6) == expected_score_a

    # Verify fusion metadata on chunk_A
    meta_a = results[0].metadata
    assert meta_a["semantic_rank"] == 1
    assert meta_a["semantic_score"] == 0.99
    assert meta_a["bm25_rank"] == 1
    assert meta_a["bm25_score"] == 15.0
    assert pytest.approx(meta_a["rrf_score"], rel=1e-6) == expected_score_a
    assert meta_a["fusion_method"] == "rrf"


def test_exact_mathematical_rrf_scoring() -> None:
    """Test precise multi-candidate RRF scores across unequal ranks."""
    # Chunk 1: rank 1 semantic, rank 3 BM25 -> 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.01639344 + 0.01587302 = 0.03226646
    # Chunk 2: rank 2 semantic, rank 1 BM25 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.01612903 + 0.01639344 = 0.03252248
    # Chunk 3: rank 3 semantic, rank 2 BM25 -> 1/(60+3) + 1/(60+2) = 1/63 + 1/62 = 0.01587302 + 0.01612903 = 0.03200205
    sem_cands = [
        create_vector_result("chunk_1", score=0.90),
        create_vector_result("chunk_2", score=0.80),
        create_vector_result("chunk_3", score=0.70),
    ]
    bm25_cands = [
        create_bm25_result("chunk_2", score=20.0),
        create_bm25_result("chunk_3", score=15.0),
        create_bm25_result("chunk_1", score=10.0),
    ]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = hybrid.retrieve("multi-rank query", config=HybridSearchConfiguration(final_top_k=3, rrf_k=60))

    assert len(results) == 3
    # Order should be Chunk 2 (0.032522) > Chunk 1 (0.032266) > Chunk 3 (0.032002)
    assert results[0].chunk_id == "chunk_2"
    assert results[1].chunk_id == "chunk_1"
    assert results[2].chunk_id == "chunk_3"

    assert pytest.approx(results[0].score, rel=1e-6) == (1.0 / 62) + (1.0 / 61)
    assert pytest.approx(results[1].score, rel=1e-6) == (1.0 / 61) + (1.0 / 63)
    assert pytest.approx(results[2].score, rel=1e-6) == (1.0 / 63) + (1.0 / 62)


def test_rank_based_not_raw_score_addition() -> None:
    """Verify that astronomical raw BM25 score does not override pure rank order."""
    # Semantic rank 1 (score 0.9) vs BM25 rank 1 (score 100000.0)
    sem_cands = [create_vector_result("chunk_sem", score=0.9)]
    bm25_cands = [create_bm25_result("chunk_bm25", score=100000.0)]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = hybrid.retrieve("test raw score", config=HybridSearchConfiguration(final_top_k=2, rrf_k=60))

    # Both items have identical RRF score (1/61)
    assert results[0].score == results[1].score


def test_candidate_pool_sizing_and_parameter_propagation() -> None:
    """Verify candidate pool sizes and filters are passed to both retrievers."""
    sem_mock = FakeSemanticRetriever([])
    bm25_mock = FakeBM25Retriever([])

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]

    filters = VectorSearchFilter(company_id="comp_test", workspace_id="ws_test")
    config = HybridSearchConfiguration(
        final_top_k=5,
        candidate_k=25,
    )

    _ = hybrid.retrieve("parameter test", config=config, filters=filters)

    assert len(sem_mock.calls) == 1
    sem_call = sem_mock.calls[0]
    assert sem_call["query"] == "parameter test"
    assert sem_call["search_config"].top_k == 25
    assert sem_call["filters"] == filters

    assert len(bm25_mock.calls) == 1
    bm25_call = bm25_mock.calls[0]
    assert bm25_call["query"] == "parameter test"
    assert bm25_call["top_k"] == 25
    assert bm25_call["filters"] == filters


def test_deterministic_tie_breaking_on_equal_rrf_score() -> None:
    """Test that equal RRF scores break ties deterministically by chunk_id ascending."""
    sem_cands = [
        create_vector_result("chunk_Z", score=0.9),  # rank 1 -> 1/61
        create_vector_result("chunk_A", score=0.9),  # rank 2 -> 1/62
    ]
    bm25_cands = [
        create_bm25_result("chunk_M", score=10.0),   # rank 1 -> 1/61
        create_bm25_result("chunk_B", score=8.0),    # rank 2 -> 1/62
    ]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = hybrid.retrieve("tie break test", config=HybridSearchConfiguration(final_top_k=4, rrf_k=60))

    # Rank 1 items (1/61): "chunk_M" < "chunk_Z"
    # Rank 2 items (1/62): "chunk_A" < "chunk_B"
    assert [r.chunk_id for r in results] == ["chunk_M", "chunk_Z", "chunk_A", "chunk_B"]


def test_semantic_only_and_bm25_only_results() -> None:
    """Test graceful handling when only one retriever returns candidates."""
    sem_cands = [create_vector_result("chunk_sem", score=0.9)]
    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_empty = FakeBM25Retriever([])

    hybrid_sem = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_empty)  # type: ignore[arg-type]
    res_sem = hybrid_sem.retrieve("query")
    assert len(res_sem) == 1
    assert res_sem[0].chunk_id == "chunk_sem"
    assert res_sem[0].metadata["semantic_rank"] == 1
    assert res_sem[0].metadata["bm25_rank"] is None

    # BM25 only
    sem_empty = FakeSemanticRetriever([])
    bm25_cands = [create_bm25_result("chunk_bm25", score=12.0)]
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid_bm25 = HybridRetriever(semantic_retriever=sem_empty, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    res_bm25 = hybrid_bm25.retrieve("query")
    assert len(res_bm25) == 1
    assert res_bm25[0].chunk_id == "chunk_bm25"
    assert res_bm25[0].metadata["bm25_rank"] == 1
    assert res_bm25[0].metadata["semantic_rank"] is None


def test_both_retrievers_empty_returns_empty_list() -> None:
    """Test graceful handling when both retrievers return []."""
    sem_mock = FakeSemanticRetriever([])
    bm25_mock = FakeBM25Retriever([])

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    assert hybrid.retrieve("empty query results") == []


@pytest.mark.parametrize("invalid_query", ["", "   ", "\t\n"])
def test_empty_query_raises_value_error(invalid_query: str) -> None:
    """Test that empty or whitespace query raises ValueError."""
    sem_mock = FakeSemanticRetriever([])
    bm25_mock = FakeBM25Retriever([])
    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Query string cannot be empty or whitespace-only"):
        hybrid.retrieve(invalid_query)


def test_semantic_retriever_failure_raises_hybrid_error() -> None:
    """Test that failure in semantic retriever raises HybridRetrievalError."""
    crashing_sem = CrashingRetriever()
    bm25_mock = FakeBM25Retriever([])
    hybrid = HybridRetriever(semantic_retriever=crashing_sem, bm25_retriever=bm25_mock)  # type: ignore[arg-type]

    with pytest.raises(HybridRetrievalError, match="Semantic retrieval failed during hybrid search"):
        hybrid.retrieve("valid query")


def test_bm25_retriever_failure_raises_hybrid_error() -> None:
    """Test that failure in BM25 retriever raises HybridRetrievalError."""
    sem_mock = FakeSemanticRetriever([])
    crashing_bm25 = CrashingRetriever()
    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=crashing_bm25)  # type: ignore[arg-type]

    with pytest.raises(HybridRetrievalError, match="BM25 retrieval failed during hybrid search"):
        hybrid.retrieve("valid query")


@pytest.mark.anyio
async def test_async_hybrid_retrieval() -> None:
    """Test asynchronous aretrieve executes concurrently and produces identical results."""
    sem_cands = [create_vector_result("chunk_A", score=0.9)]
    bm25_cands = [create_bm25_result("chunk_A", score=10.0)]

    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever(bm25_cands)

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    results = await hybrid.aretrieve("async test")

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_A"
    assert pytest.approx(results[0].score, rel=1e-6) == (1.0 / 61) + (1.0 / 61)


def test_dependency_isolation() -> None:
    """Verify that HybridRetriever does not import qdrant_client, rank_bm25, sentence_transformers."""
    import competitive_intelligence.retrieval_engine.hybrid.hybrid_retriever as hybrid_mod

    mod_file = getattr(hybrid_mod, "__file__", "")
    assert mod_file != ""
    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "qdrant_client" not in code
    assert "rank_bm25" not in code
    assert "sentence_transformers" not in code
    assert "SentenceTransformer" not in code
    assert "torch" not in code
    assert "bge_m3" not in code.lower()


def test_hybrid_retriever_returns_full_fusion_pool() -> None:
    """Test that HybridRetriever returns fusion_candidate_k candidates when final_top_k is None."""
    # Create 15 candidates
    sem_cands = [create_vector_result(f"chunk_sem_{i}", score=0.9) for i in range(15)]
    sem_mock = FakeSemanticRetriever(sem_cands)
    bm25_mock = FakeBM25Retriever([])

    hybrid = HybridRetriever(semantic_retriever=sem_mock, bm25_retriever=bm25_mock)  # type: ignore[arg-type]
    
    # 1. Default behavior (final_top_k = 10)
    config_default = HybridSearchConfiguration()
    res_default = hybrid.retrieve("query", config=config_default)
    assert len(res_default) == 10
    
    # 2. Expanded pool (final_top_k = None, fusion_candidate_k = 12)
    config_expanded = HybridSearchConfiguration(final_top_k=None, fusion_candidate_k=12)
    res_expanded = hybrid.retrieve("query", config=config_expanded)
    assert len(res_expanded) == 12
