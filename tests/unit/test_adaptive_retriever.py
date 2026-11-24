"""Unit tests for AdaptiveRetriever execution coordinator.

Tests strategy dispatch, budget propagation, filter preservation, provenance attachment,
error encapsulation, and isolation from underlying concrete storage libraries.
"""

from typing import Any
import pytest

from competitive_intelligence.retrieval_engine.interfaces import (
    BaseReranker,
    BaseRetrievalRouter,
    BaseRetriever,
)
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)
from competitive_intelligence.retrieval_engine.reranking.reranked_retriever import (
    RerankedRetriever,
)
from competitive_intelligence.retrieval_engine.routing.adaptive_retriever import (
    AdaptiveRetriever,
)
from competitive_intelligence.retrieval_engine.routing.errors import (
    AdaptiveRetrievalError,
)
from competitive_intelligence.retrieval_engine.routing.models import (
    QueryFeatures,
    RoutingDecision,
)


# ==============================================================================
# Test Fakes & Spies
# ==============================================================================


class FakeRetriever(BaseRetriever):
    """Mock retriever tracking execution calls and returning predetermined results."""

    def __init__(self, candidates: list[RetrievalResult] | None = None) -> None:
        self.candidates = candidates if candidates is not None else []
        self.calls: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        # Return copies so metadata modification doesn't mutate fixture
        return [c.model_copy(deep=True) for c in self.candidates]


class CrashingRetriever(BaseRetriever):
    """Mock retriever that simulates downstream failure."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        raise RuntimeError("Downstream service connection dropped")


class FakeReranker(BaseReranker):
    """Mock reranker tracking calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self,
        query: str,
        candidates: Any,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "candidates": list(candidates), "top_k": top_k})
        results = [c.model_copy(deep=True) for c in candidates]
        if top_k is not None:
            results = results[:top_k]
        return results


class FakeRouter(BaseRetrievalRouter):
    """Mock router returning configured decision and tracking call count."""

    def __init__(self, decision: RoutingDecision | None = None) -> None:
        self.decision = decision
        self.call_count = 0

    def route(self, query: str | RetrievalQuery) -> RoutingDecision:
        self.call_count += 1
        if self.decision is None:
            raise RuntimeError("Router failed to analyze query")
        return self.decision


def create_sample_result(chunk_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=f"Text for {chunk_id}",
        document_id="doc_1",
        version_id="ver_1",
        metadata={"existing_key": "val"},
    )


def create_sample_decision(
    strategy: RetrievalStrategy,
    use_reranker: bool = False,
    candidate_pool_size: int = 20,
    final_top_k: int = 10,
    rule_name: str = "TestRule",
) -> RoutingDecision:
    features = QueryFeatures(
        raw_query="sample query",
        token_count=2,
        char_count=12,
    )
    return RoutingDecision(
        query_text="sample query",
        strategy=strategy,
        use_reranker=use_reranker,
        candidate_pool_size=candidate_pool_size,
        final_top_k=final_top_k,
        rule_name=rule_name,
        rationale="Test decision rationale",
        features=features,
    )


@pytest.fixture
def sample_query() -> RetrievalQuery:
    return RetrievalQuery(
        query_text="sample query",
        metadata_filter=MetadataFilter(
            workspace_id="ws_1",
            company_ids=["comp_1"],
            custom_filters={"env": "prod"},
        ),
        top_k=10,
        strategy=RetrievalStrategy.HYBRID_RRF,
    )


# ==============================================================================
# Unit Tests
# ==============================================================================


@pytest.mark.anyio
async def test_bm25_route_executes_only_bm25(sample_query: RetrievalQuery) -> None:
    """Test SPARSE_BM25 decision invokes only the BM25 retriever."""
    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25, final_top_k=5)
    router = FakeRouter(decision)

    bm25 = FakeRetriever([create_sample_result("bm25_1")])
    semantic = FakeRetriever([create_sample_result("sem_1")])
    hybrid = FakeRetriever([create_sample_result("hyb_1")])

    adaptive = AdaptiveRetriever(
        router=router,
        bm25_retriever=bm25,
        semantic_retriever=semantic,
        hybrid_retriever=hybrid,
    )

    results = await adaptive.retrieve(sample_query)

    assert len(bm25.calls) == 1
    assert len(semantic.calls) == 0
    assert len(hybrid.calls) == 0

    assert bm25.calls[0].top_k == 5
    assert bm25.calls[0].strategy == RetrievalStrategy.SPARSE_BM25
    assert len(results) == 1
    assert results[0].chunk_id == "bm25_1"


@pytest.mark.anyio
async def test_semantic_route_executes_only_semantic(sample_query: RetrievalQuery) -> None:
    """Test DENSE_VECTOR decision invokes only the Semantic retriever."""
    decision = create_sample_decision(strategy=RetrievalStrategy.DENSE_VECTOR, final_top_k=8)
    router = FakeRouter(decision)

    bm25 = FakeRetriever([create_sample_result("bm25_1")])
    semantic = FakeRetriever([create_sample_result("sem_1")])

    adaptive = AdaptiveRetriever(
        router=router,
        bm25_retriever=bm25,
        semantic_retriever=semantic,
    )

    results = await adaptive.retrieve(sample_query)

    assert len(semantic.calls) == 1
    assert len(bm25.calls) == 0
    assert semantic.calls[0].top_k == 8
    assert results[0].chunk_id == "sem_1"


@pytest.mark.anyio
async def test_hybrid_unreranked_executes_hybrid(sample_query: RetrievalQuery) -> None:
    """Test HYBRID_RRF decision with use_reranker=False invokes un-reranked HybridRetriever."""
    decision = create_sample_decision(
        strategy=RetrievalStrategy.HYBRID_RRF,
        use_reranker=False,
        final_top_k=10,
    )
    router = FakeRouter(decision)

    hybrid = FakeRetriever([create_sample_result("hyb_1")])
    reranked = FakeRetriever([create_sample_result("rerank_1")])

    adaptive = AdaptiveRetriever(
        router=router,
        hybrid_retriever=hybrid,
        reranked_retriever=reranked,
    )

    results = await adaptive.retrieve(sample_query)

    assert len(hybrid.calls) == 1
    assert len(reranked.calls) == 0
    assert results[0].chunk_id == "hyb_1"


@pytest.mark.anyio
async def test_hybrid_reranked_executes_reranked_pipeline(sample_query: RetrievalQuery) -> None:
    """Test HYBRID_RRF decision with use_reranker=True delegates to RerankedRetriever."""
    decision = create_sample_decision(
        strategy=RetrievalStrategy.HYBRID_RRF,
        use_reranker=True,
        candidate_pool_size=30,
        final_top_k=5,
    )
    router = FakeRouter(decision)

    upstream = FakeRetriever([create_sample_result(f"cand_{i}") for i in range(10)])
    reranker = FakeReranker()
    reranked_pipeline = RerankedRetriever(upstream_retriever=upstream, reranker=reranker, candidate_pool_size=10)

    adaptive = AdaptiveRetriever(
        router=router,
        reranked_retriever=reranked_pipeline,
    )

    results = await adaptive.retrieve(sample_query)

    # Verify upstream was queried with candidate_pool_size=30 from the decision
    assert len(upstream.calls) == 1
    assert upstream.calls[0].top_k == 30

    # Verify reranker received candidates and final_top_k=5
    assert len(reranker.calls) == 1
    assert reranker.calls[0]["top_k"] == 5
    assert len(results) == 5


@pytest.mark.anyio
async def test_provenance_and_auditability_metadata_attached(sample_query: RetrievalQuery) -> None:
    """Test that decision metadata is attached to result metadata while preserving existing provenance."""
    decision = create_sample_decision(
        strategy=RetrievalStrategy.SPARSE_BM25,
        use_reranker=False,
        candidate_pool_size=10,
        final_top_k=5,
        rule_name="ShortExactLookupRule",
    )
    router = FakeRouter(decision)
    bm25 = FakeRetriever([create_sample_result("bm25_prov")])

    adaptive = AdaptiveRetriever(router=router, bm25_retriever=bm25)
    results = await adaptive.retrieve(sample_query)

    assert len(results) == 1
    meta = results[0].metadata
    # Preserves existing metadata
    assert meta["existing_key"] == "val"
    # Injects routing provenance
    assert meta["routing_rule"] == "ShortExactLookupRule"
    assert meta["routing_strategy"] == "sparse_bm25"
    assert meta["routing_use_reranker"] is False
    assert meta["routing_candidate_pool_size"] == 10
    assert meta["routing_final_top_k"] == 5

    # Check last_decision property on AdaptiveRetriever
    assert adaptive.last_decision == decision


@pytest.mark.anyio
async def test_filter_and_metadata_propagation(sample_query: RetrievalQuery) -> None:
    """Test that caller's filter constraints are passed cleanly to downstream retriever."""
    decision = create_sample_decision(strategy=RetrievalStrategy.DENSE_VECTOR)
    router = FakeRouter(decision)
    semantic = FakeRetriever([create_sample_result("sem_1")])

    adaptive = AdaptiveRetriever(router=router, semantic_retriever=semantic)
    _ = await adaptive.retrieve(sample_query)

    passed_query = semantic.calls[0]
    assert passed_query.metadata_filter == sample_query.metadata_filter
    assert passed_query.metadata_filter.workspace_id == "ws_1"
    assert passed_query.metadata_filter.company_ids == ["comp_1"]
    assert passed_query.metadata_filter.custom_filters == {"env": "prod"}


@pytest.mark.anyio
async def test_input_query_is_not_mutated(sample_query: RetrievalQuery) -> None:
    """Test that input RetrievalQuery is not mutated by AdaptiveRetriever."""
    original_top_k = sample_query.top_k
    original_strategy = sample_query.strategy

    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25, final_top_k=3)
    router = FakeRouter(decision)
    bm25 = FakeRetriever([create_sample_result("res_1")])

    adaptive = AdaptiveRetriever(router=router, bm25_retriever=bm25)
    _ = await adaptive.retrieve(sample_query)

    assert sample_query.top_k == original_top_k
    assert sample_query.strategy == original_strategy


@pytest.mark.anyio
async def test_single_router_invocation_per_request(sample_query: RetrievalQuery) -> None:
    """Test router is called exactly once per retrieve() request."""
    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25)
    router = FakeRouter(decision)
    bm25 = FakeRetriever([create_sample_result("bm25_1")])

    adaptive = AdaptiveRetriever(router=router, bm25_retriever=bm25)
    _ = await adaptive.retrieve(sample_query)

    assert router.call_count == 1


@pytest.mark.anyio
async def test_async_alias_aretrieve(sample_query: RetrievalQuery) -> None:
    """Test aretrieve() alias delegates to retrieve() asynchronously."""
    decision = create_sample_decision(strategy=RetrievalStrategy.DENSE_VECTOR)
    router = FakeRouter(decision)
    semantic = FakeRetriever([create_sample_result("sem_1")])

    adaptive = AdaptiveRetriever(router=router, semantic_retriever=semantic)
    results = await adaptive.aretrieve(sample_query)

    assert len(results) == 1
    assert results[0].chunk_id == "sem_1"


@pytest.mark.anyio
async def test_empty_results_handling(sample_query: RetrievalQuery) -> None:
    """Test graceful handling when downstream retriever returns empty list."""
    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25)
    router = FakeRouter(decision)
    bm25 = FakeRetriever([])

    adaptive = AdaptiveRetriever(router=router, bm25_retriever=bm25)
    results = await adaptive.retrieve(sample_query)

    assert results == []


@pytest.mark.anyio
async def test_router_failure_raises_adaptive_error(sample_query: RetrievalQuery) -> None:
    """Test that router crash raises AdaptiveRetrievalError."""
    router = FakeRouter(decision=None)
    adaptive = AdaptiveRetriever(router=router)

    with pytest.raises(AdaptiveRetrievalError, match="Query routing failed"):
        await adaptive.retrieve(sample_query)


@pytest.mark.anyio
async def test_missing_retriever_configuration_raises_adaptive_error(sample_query: RetrievalQuery) -> None:
    """Test that missing configured retriever raises AdaptiveRetrievalError."""
    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25)
    router = FakeRouter(decision)
    # bm25_retriever is omitted (None)
    adaptive = AdaptiveRetriever(router=router)

    with pytest.raises(AdaptiveRetrievalError, match="Missing retriever configuration: No BM25Retriever configured"):
        await adaptive.retrieve(sample_query)


@pytest.mark.anyio
async def test_downstream_retriever_failure_raises_adaptive_error(sample_query: RetrievalQuery) -> None:
    """Test that downstream retriever crash raises AdaptiveRetrievalError."""
    decision = create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25)
    router = FakeRouter(decision)
    crashing_bm25 = CrashingRetriever()

    adaptive = AdaptiveRetriever(router=router, bm25_retriever=crashing_bm25)

    with pytest.raises(AdaptiveRetrievalError, match="Downstream retrieval failed for strategy 'sparse_bm25'"):
        await adaptive.retrieve(sample_query)


@pytest.mark.anyio
async def test_invalid_query_type_raises_type_error() -> None:
    """Test passing invalid query object raises TypeError."""
    router = FakeRouter(create_sample_decision(strategy=RetrievalStrategy.SPARSE_BM25))
    adaptive = AdaptiveRetriever(router=router)

    with pytest.raises(TypeError, match="query must be an instance of RetrievalQuery"):
        await adaptive.retrieve("raw string instead of RetrievalQuery")  # type: ignore[arg-type]


def test_dependency_isolation_in_adaptive_retriever() -> None:
    """Verify AdaptiveRetriever does not import Qdrant, BM25Index, or SentenceTransformers directly."""
    import competitive_intelligence.retrieval_engine.routing.adaptive_retriever as mod

    mod_file = getattr(mod, "__file__", "")
    assert mod_file != ""
    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "qdrant_client" not in code
    assert "rank_bm25" not in code
    assert "sentence_transformers" not in code
    assert "SentenceTransformer" not in code
    assert "torch" not in code
    assert "bge_m3" not in code.lower()
