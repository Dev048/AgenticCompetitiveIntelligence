import pytest

from competitive_intelligence.knowledge_pipeline.models import DocumentChunk, ChunkMetadata
from competitive_intelligence.retrieval_engine.models import RetrievalResult
from competitive_intelligence.retrieval_engine.reranking.bge_m3_reranker import BGEM3Reranker
from competitive_intelligence.retrieval_engine.reranking.errors import BGERerankerError

from typing import Any
from datetime import datetime, timezone

class FakeCrossEncoder:
    """Mock CrossEncoder that deterministically scores pairs for testing."""
    def __init__(self, fail_on_predict: bool = False) -> None:
        self.fail_on_predict = fail_on_predict
        self.last_pairs: list[Any] = []
        self.last_batch_size: int | None = None

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        if self.fail_on_predict:
            raise RuntimeError("Simulated inference failure")
        self.last_pairs = pairs
        self.last_batch_size = batch_size
        
        # Deterministic scoring: score = length of query + length of candidate text
        # If candidate is "tie", return exactly 0.5 to test deterministic sorting
        scores: list[float] = []
        for q, c in pairs:
            if c == "tie":
                scores.append(0.5)
            else:
                scores.append(float(len(q) + len(c)))
        return scores


import asyncio

@pytest.fixture
def fake_candidates() -> list[RetrievalResult]:
    # Construct 3 fake candidate chunks
    def make_res(cid: str, text: str, score: float) -> RetrievalResult:
        meta = ChunkMetadata(
            source_id="src_1",
            document_id="doc_1",
            version_id="v1",
            company_id="comp_1",
            workspace_id="ws_1",
            source_url="http://example.com",
            source_category="test",
            collected_at=datetime.now(timezone.utc)
        )
        chunk = DocumentChunk(
            chunk_id=cid,
            document_id="doc_1",
            version_id="v1",
            content=text,
            content_hash="abc",
            metadata=meta
        )
        return RetrievalResult.from_chunk(chunk, score=score)

    c1 = make_res("cid_1", "short", 0.9)
    c1.metadata["test_meta"] = "val_1"
    
    c2 = make_res("cid_2", "a much longer text", 0.8)
    
    c3 = make_res("cid_3", "medium text", 0.95)
    
    return [c1, c2, c3]


def test_reranker_basic_preservation(fake_candidates: list[RetrievalResult]) -> None:
    """Test that all original fields and metadata are preserved."""
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model)

    results = asyncio.run(reranker.rerank("query", fake_candidates))
    
    assert len(results) == 3
    # Check that original_rank is correct (1-indexed) based on input order before sorting
    # We passed c1, c2, c3.
    # Scores will be: 
    # c1 ("short" len 5) + q(5) = 10.0
    # c2 ("a much longer text" len 18) + q(5) = 23.0
    # c3 ("medium text" len 11) + q(5) = 16.0
    # Sorted order (descending): c2 (23), c3 (16), c1 (10)
    
    assert results[0].chunk_id == "cid_2"
    assert results[0].original_rank == 2
    assert results[0].reranker_score == 23.0
    assert results[0].score == 0.8  # Original score MUST NOT be overwritten
    assert results[0].reranker_model == "BAAI/bge-reranker-v2-m3"

    assert results[1].chunk_id == "cid_3"
    assert results[1].original_rank == 3
    assert results[1].reranker_score == 16.0
    assert results[1].score == 0.95

    assert results[2].chunk_id == "cid_1"
    assert results[2].original_rank == 1
    assert results[2].reranker_score == 10.0
    assert results[2].score == 0.9
    assert results[2].metadata.get("test_meta") == "val_1"


def test_reranker_batching(fake_candidates: list[RetrievalResult]) -> None:
    """Test that the configured batch size is passed to the underlying model."""
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model, batch_size=42)

    asyncio.run(reranker.rerank("query", fake_candidates))
    assert fake_model.last_batch_size == 42
    assert len(fake_model.last_pairs) == 3


def test_reranker_top_k(fake_candidates: list[RetrievalResult]) -> None:
    """Test that top_k correctly slices the results after sorting."""
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model)

    # We only want the top 2 out of the 3 candidates
    results = asyncio.run(reranker.rerank("query", fake_candidates, top_k=2))
    assert len(results) == 2
    assert results[0].chunk_id == "cid_2"
    assert results[1].chunk_id == "cid_3"


def test_reranker_tie_breaking() -> None:
    """Test that ties are broken deterministically by chunk_id."""
    def make_res(cid: str) -> RetrievalResult:
        meta = ChunkMetadata(
            source_id="s1", document_id="d1", version_id="v1", company_id="c1", workspace_id="w1", source_url="http://example.com",
            source_category="test", collected_at=datetime.now(timezone.utc)
        )
        chunk = DocumentChunk(chunk_id=cid, document_id="d1", version_id="v1", content="tie", content_hash="x", metadata=meta)
        return RetrievalResult.from_chunk(chunk, score=0.0)

    # Note: sort is descending on (score, chunk_id).
    # Since scores will all be 0.5 (see FakeCrossEncoder), they will be sorted purely on chunk_id descending.
    # Therefore, we expect cid_C > cid_B > cid_A.
    cands = [make_res("cid_A"), make_res("cid_C"), make_res("cid_B")]
    
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model)
    results = asyncio.run(reranker.rerank("q", cands))
    
    assert [r.chunk_id for r in results] == ["cid_C", "cid_B", "cid_A"]


def test_reranker_empty_inputs(fake_candidates: list[RetrievalResult]) -> None:
    """Test handling of empty query and empty candidates."""
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model)

    # Empty candidates returns empty list
    results = asyncio.run(reranker.rerank("query", []))
    assert results == []

    # Empty query raises ValueError
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        asyncio.run(reranker.rerank("   ", fake_candidates))


def test_reranker_invalid_top_k(fake_candidates: list[RetrievalResult]) -> None:
    """Test that invalid top_k is rejected."""
    fake_model = FakeCrossEncoder()
    reranker = BGEM3Reranker(cross_encoder=fake_model)

    with pytest.raises(ValueError, match="top_k must be > 0"):
        asyncio.run(reranker.rerank("query", fake_candidates, top_k=0))


def test_reranker_inference_failure(fake_candidates: list[RetrievalResult]) -> None:
    """Test that model exceptions are wrapped cleanly."""
    fake_model = FakeCrossEncoder(fail_on_predict=True)
    reranker = BGEM3Reranker(cross_encoder=fake_model)

    with pytest.raises(BGERerankerError, match="Failed to score candidates: Simulated inference failure"):
        asyncio.run(reranker.rerank("query", fake_candidates))


def test_reranker_lazy_load_failure() -> None:
    """Test that initialization failures are wrapped cleanly."""
    reranker = BGEM3Reranker(model_name="this-model-does-not-exist-12345")
    
    # We must pass at least one candidate to trigger model load
    meta = ChunkMetadata(source_id="s1", document_id="d1", version_id="v1", company_id="c1", workspace_id="w1", source_url="http://example.com", source_category="test", collected_at=datetime.now(timezone.utc))
    chunk = DocumentChunk(chunk_id="c", document_id="d1", version_id="v1", content="t", content_hash="x", metadata=meta)
    res = RetrievalResult.from_chunk(chunk, score=0.0)
    
    with pytest.raises(BGERerankerError, match="Failed to initialize model"):
        asyncio.run(reranker.rerank("query", [res]))
