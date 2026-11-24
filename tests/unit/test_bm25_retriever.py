"""Unit tests for LexicalTokenizer, BM25Index, and BM25Retriever.

Verifies exact-term search, CI entity preservation (pricing, versions, model names),
tokenization consistency, pre-ranking metadata filtering, top-k limiting,
empty/zero-match handling, score/provenance preservation, and strict library isolation.
"""

from datetime import datetime, timezone
import uuid

import pytest

from competitive_intelligence.knowledge_pipeline import (
    ChunkMetadata,
    DocumentChunk,
    VectorSearchFilter,
    compute_content_hash,
)
from competitive_intelligence.retrieval_engine import (
    BaseLexicalIndex,
    BM25Index,
    BM25Retriever,
    LexicalTokenizer,
    RetrievalResult,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_chunk(
    content: str,
    company_id: str = "comp_openai",
    workspace_id: str = "ws_ci",
    doc_id: str = "doc_news",
    version_id: str = "ver_1",
    source_id: str = "src_blog",
    source_category: str = "news",
    content_type: str = "html",
    parent_chunk_id: str | None = None,
    chunk_level: int = 0,
) -> DocumentChunk:
    """Helper to create a valid DocumentChunk with rich metadata."""
    cid = str(uuid.uuid4())
    meta = ChunkMetadata(
        source_id=source_id,
        document_id=doc_id,
        version_id=version_id,
        company_id=company_id,
        workspace_id=workspace_id,
        source_url=f"https://{company_id}.com/post",
        source_category=source_category,
        collected_at=datetime.now(timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=0,
        total_chunks=1,
        language="en",
        custom_tags={"content_type": content_type, "title": "Release Note"},
    )
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        version_id=version_id,
        content=content,
        chunk_index=0,
        metadata=meta,
        content_hash=compute_content_hash(content),
        parent_chunk_id=parent_chunk_id,
        chunk_level=chunk_level,
        created_at=datetime.now(timezone.utc),
    )


# ==============================================================================
# Tokenizer Tests
# ==============================================================================


def test_tokenizer_preserves_ci_entities() -> None:
    """Test that tokenizer preserves pricing, version strings, model identifiers, and percentages."""
    tokenizer = LexicalTokenizer()

    text = (
        "OpenAI announced GPT-4o with $20/month pricing and batch API at $0.015 per 1k tokens. "
        "Anthropic released Claude-3.5-Sonnet and v2.1.0-rc1 with 50% discount for BAAI/bge-m3."
    )
    tokens = tokenizer.tokenize(text)

    # Pricing & percentages
    assert "$20" in tokens
    assert "$0.015" in tokens
    assert "50%" in tokens

    # Model & version identifiers
    assert "gpt-4o" in tokens
    assert "claude-3.5-sonnet" in tokens
    assert "v2.1.0-rc1" in tokens
    assert "bge-m3" in tokens

    # General terms
    assert "openai" in tokens
    assert "pricing" in tokens
    assert "discount" in tokens


def test_tokenizer_boundary_punctuation_stripping() -> None:
    """Test that punctuation attached to words is stripped cleanly."""
    tokenizer = LexicalTokenizer()
    text = "Hello, world! (Enterprise) [Pricing]: tier_1; 'quotes'."
    tokens = tokenizer.tokenize(text)

    assert "hello" in tokens
    assert "world" in tokens
    assert "enterprise" in tokens
    assert "pricing" in tokens
    assert "tier_1" in tokens
    assert "quotes" in tokens


def test_tokenizer_empty_and_whitespace() -> None:
    """Test tokenizer on empty or whitespace strings."""
    tokenizer = LexicalTokenizer()
    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("   \n\t  ") == []


# ==============================================================================
# BM25Index & BM25Retriever Tests
# ==============================================================================


def test_bm25_index_implements_interface() -> None:
    """Test that BM25Index implements BaseLexicalIndex."""
    index = BM25Index()
    assert isinstance(index, BaseLexicalIndex)
    assert index.chunk_count == 0


def test_basic_exact_term_and_ranking() -> None:
    """Test BM25 retrieval ranks relevant chunk with matching terms highest."""
    chunk_1 = create_chunk("OpenAI GPT-4o enterprise subscription costs $20/month.")
    chunk_2 = create_chunk("Anthropic Claude-3.5-Sonnet released with advanced reasoning.")
    chunk_3 = create_chunk("OpenAI batch API pricing reduced with 50% discount for high volume.")

    index = BM25Index()
    index.build_index([chunk_1, chunk_2, chunk_3])
    assert index.chunk_count == 3

    retriever = BM25Retriever(index=index)

    # Search for Claude
    results_claude = retriever.retrieve("Claude-3.5-Sonnet")
    assert len(results_claude) == 1
    assert results_claude[0].chunk_id == chunk_2.chunk_id
    assert results_claude[0].score > 0.0
    assert "Claude-3.5-Sonnet" in results_claude[0].text

    # Search for OpenAI pricing
    results_pricing = retriever.retrieve("OpenAI pricing")
    assert len(results_pricing) == 2
    # Both chunk_1 and chunk_3 contain "OpenAI" and pricing concepts
    chunk_ids = [r.chunk_id for r in results_pricing]
    assert chunk_1.chunk_id in chunk_ids
    assert chunk_3.chunk_id in chunk_ids


def test_top_k_limiting() -> None:
    """Test that top_k restricts the number of returned results."""
    chunks = [
        create_chunk(f"Document segment {i} discussing cloud infrastructure pricing.")
        for i in range(10)
    ]
    index = BM25Index()
    index.build_index(chunks)

    retriever = BM25Retriever(index=index, default_top_k=3)

    # Default top_k = 3
    res_default = retriever.retrieve("infrastructure pricing")
    assert len(res_default) == 3

    # Explicit top_k = 5
    res_custom = retriever.retrieve("infrastructure pricing", top_k=5)
    assert len(res_custom) == 5


def test_metadata_filtering() -> None:
    """Test pre-ranking metadata filtering by company_id, workspace_id, version_id, etc."""
    chunk_openai_1 = create_chunk("Enterprise security audit.", company_id="comp_openai", workspace_id="ws_1")
    chunk_openai_2 = create_chunk("Enterprise API tokens.", company_id="comp_openai", workspace_id="ws_2")
    chunk_anthropic = create_chunk("Enterprise security audit.", company_id="comp_anthropic", workspace_id="ws_1")

    index = BM25Index()
    index.build_index([chunk_openai_1, chunk_openai_2, chunk_anthropic])
    retriever = BM25Retriever(index=index)

    # 1. Filter by company_id
    filter_openai = VectorSearchFilter(company_id="comp_openai")
    results_openai = retriever.retrieve("Enterprise security", filters=filter_openai)
    assert len(results_openai) == 2
    assert all(r.company_id == "comp_openai" for r in results_openai)

    # 2. Filter by company_id AND workspace_id
    filter_strict = VectorSearchFilter(company_id="comp_openai", workspace_id="ws_1")
    results_strict = retriever.retrieve("Enterprise security", filters=filter_strict)
    assert len(results_strict) == 1
    assert results_strict[0].chunk_id == chunk_openai_1.chunk_id


def test_parent_child_provenance_in_retrieval_result() -> None:
    """Test that RetrievalResult accurately reflects child chunk and parent_chunk_id."""
    parent_id = str(uuid.uuid4())
    child_chunk = create_chunk(
        content="Child section regarding SLA response times.",
        parent_chunk_id=parent_id,
        chunk_level=1,
    )
    index = BM25Index()
    index.build_index([child_chunk])
    retriever = BM25Retriever(index=index)

    results = retriever.retrieve("SLA response times")
    assert len(results) == 1
    res = results[0]
    assert isinstance(res, RetrievalResult)
    assert res.chunk_id == child_chunk.chunk_id
    assert res.parent_chunk_id == parent_id
    assert res.chunk_level == 1
    assert res.score > 0.0


@pytest.mark.parametrize("invalid_query", ["", "   ", "\t\r\n"])
def test_empty_query_raises_value_error(invalid_query: str) -> None:
    """Test that empty or whitespace query raises ValueError."""
    index = BM25Index()
    retriever = BM25Retriever(index=index)
    with pytest.raises(ValueError, match="Query string cannot be empty or whitespace-only"):
        retriever.retrieve(invalid_query)


@pytest.mark.parametrize("invalid_top_k", [0, -1, -10])
def test_invalid_top_k_raises_value_error(invalid_top_k: int) -> None:
    """Test that non-positive top_k raises ValueError."""
    index = BM25Index()
    retriever = BM25Retriever(index=index)
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("valid query", top_k=invalid_top_k)


def test_empty_index_and_zero_matches_returns_empty_list() -> None:
    """Test that querying an empty index or querying terms not in vocabulary returns []."""
    index = BM25Index()
    retriever = BM25Retriever(index=index)
    assert retriever.retrieve("any term") == []

    chunk = create_chunk("Quantum computing quantum entanglement.")
    index.build_index([chunk])
    # Term completely absent
    assert retriever.retrieve("pricing discount") == []


def test_deterministic_rebuild_and_input_immutability() -> None:
    """Test that rebuilding index is deterministic and inputs are not mutated."""
    chunk = create_chunk("Deterministic lexical test content.")
    original_dict = chunk.model_dump()

    index = BM25Index()
    index.build_index([chunk])
    res_1 = index.search("lexical test")

    # Rebuild
    index.build_index([chunk])
    res_2 = index.search("lexical test")

    assert len(res_1) == 1
    assert len(res_2) == 1
    assert res_1[0].score == res_2[0].score
    assert chunk.model_dump() == original_dict


@pytest.mark.anyio
async def test_async_bm25_retrieval() -> None:
    """Test asynchronous aretrieve method."""
    chunk = create_chunk("Async BM25 retrieval test.")
    index = BM25Index()
    index.build_index([chunk])
    retriever = BM25Retriever(index=index)

    results = await retriever.aretrieve("BM25 retrieval")
    assert len(results) == 1
    assert results[0].chunk_id == chunk.chunk_id


def test_dependency_isolation() -> None:
    """Verify that BM25Retriever, BaseLexicalIndex, and models do not leak rank_bm25."""
    import competitive_intelligence.retrieval_engine.lexical.bm25_retriever as ret_mod
    import competitive_intelligence.retrieval_engine.lexical.interfaces as iface_mod
    import competitive_intelligence.retrieval_engine.models as model_mod

    for mod in (ret_mod, iface_mod, model_mod):
        mod_file = getattr(mod, "__file__", "")
        assert mod_file != ""
        with open(mod_file, "r", encoding="utf-8") as f:
            code = f.read()

        assert "rank_bm25" not in code
        assert "BM25Okapi" not in code
