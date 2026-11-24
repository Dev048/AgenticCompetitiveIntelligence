"""Integration test for QdrantVectorStore and Knowledge Pipeline embedding pipeline.

Verifies end-to-end collection lifecycle, embedding record mapping, indexing,
filtered vector retrieval, and provenance tracking against a local/in-memory Qdrant instance.
"""

from datetime import datetime, timezone
import uuid

import pytest

from competitive_intelligence.knowledge_pipeline import (
    ChunkMetadata,
    DistanceMetric,
    DocumentChunk,
    EmbeddingRecord,
    MockEmbeddingProvider,
    QdrantVectorStore,
    VectorSearchFilter,
    VectorStoreConfig,
    VectorStoreEntry,
    compute_content_hash,
)


@pytest.mark.integration
def test_qdrant_end_to_end_indexing_and_search() -> None:
    """Test full integration: mock embedding -> VectorStoreEntry -> Qdrant indexing -> filtered search."""
    config = VectorStoreConfig(
        collection_name=f"ci_test_{uuid.uuid4().hex[:8]}",
        vector_dimensions=16,
        distance_metric=DistanceMetric.COSINE,
        location=":memory:",
    )
    store = QdrantVectorStore(config=config)
    store.ensure_collection()

    embedder = MockEmbeddingProvider(dimension=16)

    # 1. Create 3 document chunks across 2 competitors
    chunks: list[DocumentChunk] = []
    company_ids = ["comp_openai", "comp_anthropic", "comp_openai"]
    contents = [
        "OpenAI announced custom enterprise GPT models with dedicated capacity.",
        "Anthropic released Claude 3.5 Sonnet with enhanced coding and visual reasoning.",
        "OpenAI reduced batch API pricing by 50% for high-volume customers.",
    ]

    for i, (cid, text) in enumerate(zip(company_ids, contents, strict=True)):
        chunk_id = str(uuid.uuid4())
        meta = ChunkMetadata(
            source_id=f"src_{i}",
            document_id=f"doc_{cid}",
            version_id="ver_2026_q3",
            company_id=cid,
            workspace_id="ws_intel",
            source_url=f"https://{cid}.com/news",
            source_category="news",
            collected_at=datetime.now(timezone.utc),
            start_char=0,
            end_char=len(text),
            chunk_index=i,
            total_chunks=3,
            language="en",
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=f"doc_{cid}",
                version_id="ver_2026_q3",
                content=text,
                chunk_index=i,
                metadata=meta,
                content_hash=compute_content_hash(text),
                parent_chunk_id=None,
                chunk_level=0,
                created_at=datetime.now(timezone.utc),
            )
        )

    # 2. Generate embeddings
    texts = [c.content for c in chunks]
    vectors = embedder.embed_texts(texts)

    # 3. Create entries and upsert into Qdrant
    entries: list[VectorStoreEntry] = []
    for chunk, vec in zip(chunks, vectors, strict=True):
        emb_record = EmbeddingRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            vector=vec,
            model_name="mock-embedder",
            dimensions=16,
        )
        entries.append(VectorStoreEntry.from_chunk_and_embedding(chunk, emb_record))

    store.upsert_entries(entries)

    # 4. Search with query vector pointing to Anthropic chunk
    query_vec = embedder.embed_query(chunks[1].content)
    results = store.search(query_vector=query_vec, top_k=3)

    assert len(results) == 3
    # Top result should be the exact matching Anthropic chunk (score = 1.0)
    assert results[0].company_id == "comp_anthropic"
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0
    assert "Claude 3.5 Sonnet" in results[0].text

    # 5. Search with company filter for OpenAI
    openai_filter = VectorSearchFilter(company_id="comp_openai")
    openai_results = store.search(query_vector=query_vec, top_k=5, filters=openai_filter)

    assert len(openai_results) == 2
    assert all(r.company_id == "comp_openai" for r in openai_results)
