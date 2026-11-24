import asyncio
import pytest

from typing import Sequence
from datetime import datetime, timezone

from competitive_intelligence.retrieval_engine.interfaces import BaseReranker, BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
    MetadataFilter,
    RetrievalStrategy,
)
from competitive_intelligence.retrieval_engine.reranking.reranked_retriever import RerankedRetriever
from competitive_intelligence.retrieval_engine.reranking.errors import RerankedRetrievalError

class FakeRetriever(BaseRetriever):
    def __init__(self, candidates: list[RetrievalResult], fail: bool = False):
        self.candidates = candidates
        self.fail = fail
        self.last_query: RetrievalQuery | None = None

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        if self.fail:
            raise RuntimeError("Upstream failure")
        self.last_query = query
        return self.candidates


class FakeReranker(BaseReranker):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.last_query_text: str | None = None
        self.last_candidates: Sequence[RetrievalResult] | None = None
        self.last_top_k: int | None = None

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if self.fail:
            raise RuntimeError("Reranker failure")
        self.last_query_text = query
        self.last_candidates = candidates
        self.last_top_k = top_k
        
        # Simply return reversed candidates to simulate reranking changes
        if top_k:
            return list(reversed(candidates))[:top_k]
        return list(reversed(candidates))


@pytest.fixture
def fake_query() -> RetrievalQuery:
    filter_ = MetadataFilter(workspace_id="ws1")
    return RetrievalQuery(
        query_text="test query",
        metadata_filter=filter_,
        top_k=5,
        strategy=RetrievalStrategy.HYBRID_RRF
    )

@pytest.fixture
def fake_candidates() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"c{i}",
            score=0.9,
            text=f"text {i}",
            document_id="d1",
            version_id="v1"
        ) for i in range(3)
    ]


def test_reranked_retriever_success(fake_query: RetrievalQuery, fake_candidates: list[RetrievalResult]) -> None:
    upstream = FakeRetriever(candidates=fake_candidates)
    reranker = FakeReranker()
    retriever = RerankedRetriever(upstream_retriever=upstream, reranker=reranker)

    results = asyncio.run(retriever.retrieve(fake_query))

    # Verify upstream was called with modified query for pool size
    assert upstream.last_query is not None
    assert upstream.last_query.top_k == 50
    assert upstream.last_query.query_text == "test query"

    # Verify reranker received correct inputs
    assert reranker.last_query_text == "test query"
    assert reranker.last_candidates == fake_candidates
    assert reranker.last_top_k == 5

    # Verify results (FakeReranker reverses the list)
    assert len(results) == 3
    assert results[0].chunk_id == "c2"
    assert results[1].chunk_id == "c1"
    assert results[2].chunk_id == "c0"


def test_reranked_retriever_custom_pool_size(fake_query: RetrievalQuery, fake_candidates: list[RetrievalResult]) -> None:
    upstream = FakeRetriever(candidates=fake_candidates)
    reranker = FakeReranker()
    retriever = RerankedRetriever(upstream_retriever=upstream, reranker=reranker, candidate_pool_size=100)

    _ = asyncio.run(retriever.retrieve(fake_query))

    assert upstream.last_query is not None
    assert upstream.last_query.top_k == 100
    assert reranker.last_top_k == fake_query.top_k


def test_reranked_retriever_empty_upstream(fake_query: RetrievalQuery) -> None:
    upstream = FakeRetriever(candidates=[])
    reranker = FakeReranker()
    retriever = RerankedRetriever(upstream_retriever=upstream, reranker=reranker)

    results = asyncio.run(retriever.retrieve(fake_query))

    assert results == []
    # Verify reranker was skipped
    assert reranker.last_query_text is None


def test_reranked_retriever_upstream_failure(fake_query: RetrievalQuery, fake_candidates: list[RetrievalResult]) -> None:
    upstream = FakeRetriever(candidates=fake_candidates, fail=True)
    reranker = FakeReranker()
    retriever = RerankedRetriever(upstream_retriever=upstream, reranker=reranker)

    with pytest.raises(RerankedRetrievalError, match="Upstream retrieval failed for query 'test query': Upstream failure"):
        asyncio.run(retriever.retrieve(fake_query))


def test_reranked_retriever_reranker_failure(fake_query: RetrievalQuery, fake_candidates: list[RetrievalResult]) -> None:
    upstream = FakeRetriever(candidates=fake_candidates)
    reranker = FakeReranker(fail=True)
    retriever = RerankedRetriever(upstream_retriever=upstream, reranker=reranker)

    with pytest.raises(RerankedRetrievalError, match="Reranking failed for query 'test query': Reranker failure"):
        asyncio.run(retriever.retrieve(fake_query))
