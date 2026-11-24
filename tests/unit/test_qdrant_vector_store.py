"""Unit tests for QdrantVectorStore adapter in Knowledge Pipeline subsystem.

Uses in-memory QdrantClient (location=':memory:') to verify collection creation,
entry upsert, vector dimension validation, similarity search, payload filtering,
historical version coexistence, parent-child provenance, and strict domain isolation.
"""

from datetime import datetime, timezone
import uuid

import pytest

from competitive_intelligence.knowledge_pipeline import (
    BaseVectorStore,
    ChunkMetadata,
    DistanceMetric,
    DocumentChunk,
    EmbeddingRecord,
    QdrantVectorStore,
    SearchConfiguration,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStore,
    VectorStoreConfig,
    VectorStoreEntry,
    VectorStoreError,
    compute_content_hash,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_test_chunk(
    chunk_id: str | None = None,
    content: str = "Competitor pricing increased to $49/mo.",
    chunk_index: int = 0,
    chunk_level: int = 0,
    parent_chunk_id: str | None = None,
    doc_id: str = "doc_test_101",
    version_id: str = "ver_test_101",
    source_id: str = "src_pricing",
    company_id: str = "comp_acme",
    workspace_id: str = "ws_default",
) -> DocumentChunk:
    """Helper to construct a valid DocumentChunk."""
    cid = chunk_id or str(uuid.uuid4())
    metadata = ChunkMetadata(
        source_id=source_id,
        document_id=doc_id,
        version_id=version_id,
        company_id=company_id,
        workspace_id=workspace_id,
        source_url="https://acme.com/pricing",
        source_category="pricing",
        collected_at=datetime.now(timezone.utc),
        start_char=0,
        end_char=len(content),
        chunk_index=chunk_index,
        total_chunks=1,
        language="en",
        custom_tags={"title": "Acme Pricing Page", "content_type": "html"},
    )
    return DocumentChunk(
        chunk_id=cid,
        document_id=doc_id,
        version_id=version_id,
        content=content,
        chunk_index=chunk_index,
        metadata=metadata,
        content_hash=compute_content_hash(content),
        parent_chunk_id=parent_chunk_id,
        chunk_level=chunk_level,
        created_at=datetime.now(timezone.utc),
    )


def create_test_embedding(
    chunk: DocumentChunk,
    dimensions: int = 1024,
    seed_val: float = 0.1,
) -> EmbeddingRecord:
    """Helper to construct a valid EmbeddingRecord for a chunk."""
    # Deterministic unit vector
    raw_vec = [(seed_val + (i * 0.01)) % 1.0 for i in range(dimensions)]
    norm = sum(x * x for x in raw_vec) ** 0.5 or 1.0
    unit_vec = [round(x / norm, 6) for x in raw_vec]

    return EmbeddingRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        version_id=chunk.version_id,
        vector=unit_vec,
        model_name="BAAI/bge-m3",
        model_version="1.0.0",
        dimensions=dimensions,
    )


# ==============================================================================
# Configuration & Domain Model Tests
# ==============================================================================


def test_vector_store_config_defaults() -> None:
    """Test default collection configuration parameters."""
    config = VectorStoreConfig()
    assert config.collection_name == "ci_knowledge_chunks"
    assert config.vector_dimensions == 1024
    assert config.distance_metric == DistanceMetric.COSINE
    assert config.embedding_model_name == "BAAI/bge-m3"
    assert config.embedding_model_version == "1.0.0"
    assert config.location == ":memory:"


def test_vector_store_config_validation() -> None:
    """Test validation of configuration parameters."""
    with pytest.raises(ValueError, match="vector_dimensions must be a positive integer"):
        VectorStoreConfig(vector_dimensions=0)

    with pytest.raises(ValueError, match="Field cannot be empty or whitespace-only"):
        VectorStoreConfig(collection_name="   ")


def test_vector_store_entry_from_chunk_and_embedding() -> None:
    """Test factory constructor linking DocumentChunk and EmbeddingRecord."""
    chunk = create_test_chunk()
    embedding = create_test_embedding(chunk)

    entry = VectorStoreEntry.from_chunk_and_embedding(chunk=chunk, embedding=embedding)

    assert entry.chunk_id == chunk.chunk_id
    assert entry.document_id == chunk.document_id
    assert entry.version_id == chunk.version_id
    assert entry.text == chunk.content
    assert entry.vector == embedding.vector
    assert entry.source_id == "src_pricing"
    assert entry.company_id == "comp_acme"
    assert entry.workspace_id == "ws_default"
    assert entry.canonical_url == "https://acme.com/pricing"
    assert entry.title == "Acme Pricing Page"


def test_vector_store_entry_mismatched_ids_rejected() -> None:
    """Test that mismatched chunk_id between chunk and embedding raises ValueError."""
    chunk = create_test_chunk(chunk_id=str(uuid.uuid4()))
    other_chunk = create_test_chunk(chunk_id=str(uuid.uuid4()))
    embedding = create_test_embedding(other_chunk)

    with pytest.raises(ValueError, match="Chunk ID mismatch"):
        VectorStoreEntry.from_chunk_and_embedding(chunk=chunk, embedding=embedding)


# ==============================================================================
# QdrantVectorStore Operations Tests (In-Memory)
# ==============================================================================


def test_store_implements_interface() -> None:
    """Test that QdrantVectorStore implements BaseVectorStore interface."""
    store = QdrantVectorStore()
    assert isinstance(store, BaseVectorStore)
    assert isinstance(store, VectorStore)


def test_ensure_collection_is_idempotent() -> None:
    """Test that creating / ensuring collection works and can be called repeatedly."""
    store = QdrantVectorStore()
    store.ensure_collection()
    # Second call should safely detect existing collection without crashing
    store.ensure_collection()


def test_upsert_and_retrieve_entry() -> None:
    """Test upserting a single entry and retrieving it by chunk_id."""
    store = QdrantVectorStore()
    store.ensure_collection()

    chunk = create_test_chunk()
    embedding = create_test_embedding(chunk)
    entry = VectorStoreEntry.from_chunk_and_embedding(chunk=chunk, embedding=embedding)

    store.upsert_entries([entry])

    result = store.get_by_id(chunk.chunk_id)
    assert result is not None
    assert isinstance(result, VectorSearchResult)
    assert result.chunk_id == chunk.chunk_id
    assert result.text == chunk.content
    assert result.document_id == chunk.document_id
    assert result.version_id == chunk.version_id
    assert result.company_id == "comp_acme"
    assert result.workspace_id == "ws_default"
    assert result.canonical_url == "https://acme.com/pricing"


def test_get_nonexistent_id_returns_none() -> None:
    """Test that retrieving an unknown chunk_id returns None."""
    store = QdrantVectorStore()
    store.ensure_collection()

    missing_id = str(uuid.uuid4())
    assert store.get_by_id(missing_id) is None


def test_dimension_mismatch_raises_value_error() -> None:
    """Test that upserting or querying with wrong vector dimension raises ValueError."""
    store = QdrantVectorStore(config=VectorStoreConfig(vector_dimensions=1024))
    store.ensure_collection()

    chunk = create_test_chunk()
    # 512 dimensions instead of 1024
    embedding_wrong_dim = create_test_embedding(chunk, dimensions=512)
    entry = VectorStoreEntry.from_chunk_and_embedding(chunk=chunk, embedding=embedding_wrong_dim)

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        store.upsert_entries([entry])

    with pytest.raises(ValueError, match="Query vector dimension mismatch"):
        store.search(query_vector=[0.1] * 512)


def test_vector_search_returns_ranked_results() -> None:
    """Test vector similarity search ranks closest vectors highest."""
    store = QdrantVectorStore()
    store.ensure_collection()

    # Entry A with vector [1.0, 0.0, ...]
    chunk_a = create_test_chunk(content="Passage A about pricing")
    vec_a = [1.0] + [0.0] * 1023
    emb_a = EmbeddingRecord(
        chunk_id=chunk_a.chunk_id,
        document_id=chunk_a.document_id,
        version_id=chunk_a.version_id,
        vector=vec_a,
        model_name="BAAI/bge-m3",
        dimensions=1024,
    )
    entry_a = VectorStoreEntry.from_chunk_and_embedding(chunk=chunk_a, embedding=emb_a)

    # Entry B with vector [0.0, 1.0, ...]
    chunk_b = create_test_chunk(content="Passage B about infrastructure")
    vec_b = [0.0, 1.0] + [0.0] * 1022
    emb_b = EmbeddingRecord(
        chunk_id=chunk_b.chunk_id,
        document_id=chunk_b.document_id,
        version_id=chunk_b.version_id,
        vector=vec_b,
        model_name="BAAI/bge-m3",
        dimensions=1024,
    )
    entry_b = VectorStoreEntry.from_chunk_and_embedding(chunk=chunk_b, embedding=emb_b)

    store.upsert_entries([entry_a, entry_b])

    # Search with query vector pointing towards A [1.0, 0.0, ...]
    query_vec = [1.0] + [0.0] * 1023
    results = store.search(query_vector=query_vec, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == chunk_a.chunk_id
    assert results[0].score > results[1].score
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0


def test_search_top_k_limiting() -> None:
    """Test that top_k restricts the number of returned results."""
    store = QdrantVectorStore()
    store.ensure_collection()

    entries: list[VectorStoreEntry] = []
    for i in range(5):
        chunk = create_test_chunk(content=f"Paragraph {i}")
        emb = create_test_embedding(chunk, seed_val=0.1 * (i + 1))
        entries.append(VectorStoreEntry.from_chunk_and_embedding(chunk, emb))

    store.upsert_entries(entries)

    results = store.search(query_vector=[0.1] * 1024, top_k=3)
    assert len(results) == 3


def test_payload_filtering() -> None:
    """Test filtering search results by company_id, workspace_id, version_id, etc."""
    store = QdrantVectorStore()
    store.ensure_collection()

    # Company A chunk
    chunk_a = create_test_chunk(company_id="comp_alpha", workspace_id="ws_main")
    emb_a = create_test_embedding(chunk_a)
    entry_a = VectorStoreEntry.from_chunk_and_embedding(chunk_a, emb_a)

    # Company B chunk
    chunk_b = create_test_chunk(company_id="comp_beta", workspace_id="ws_main")
    emb_b = create_test_embedding(chunk_b)
    entry_b = VectorStoreEntry.from_chunk_and_embedding(chunk_b, emb_b)

    store.upsert_entries([entry_a, entry_b])

    # Filter for Company Alpha only
    filter_alpha = VectorSearchFilter(company_id="comp_alpha")
    results_alpha = store.search(query_vector=emb_a.vector, top_k=10, filters=filter_alpha)

    assert len(results_alpha) == 1
    assert results_alpha[0].chunk_id == chunk_a.chunk_id
    assert results_alpha[0].company_id == "comp_alpha"


def test_historical_versions_coexist() -> None:
    """Test that multiple historical versions of a document coexist without deletion."""
    store = QdrantVectorStore()
    store.ensure_collection()

    doc_id = "doc_tracked_page"
    # Version 1 (past price $20)
    chunk_v1 = create_test_chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        version_id="ver_2026_q1",
        content="Product Pro tier is $20/month.",
    )
    entry_v1 = VectorStoreEntry.from_chunk_and_embedding(chunk_v1, create_test_embedding(chunk_v1))

    # Version 2 (new price $30)
    chunk_v2 = create_test_chunk(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        version_id="ver_2026_q2",
        content="Product Pro tier is $30/month.",
    )
    entry_v2 = VectorStoreEntry.from_chunk_and_embedding(chunk_v2, create_test_embedding(chunk_v2))

    store.upsert_entries([entry_v1, entry_v2])

    # Query all chunks for doc_id
    doc_filter = VectorSearchFilter(document_id=doc_id)
    all_doc_results = store.search(query_vector=[0.1] * 1024, top_k=10, filters=doc_filter)
    assert len(all_doc_results) == 2

    # Query specifically version 1
    v1_filter = VectorSearchFilter(document_id=doc_id, version_id="ver_2026_q1")
    v1_results = store.search(query_vector=[0.1] * 1024, top_k=10, filters=v1_filter)
    assert len(v1_results) == 1
    assert v1_results[0].version_id == "ver_2026_q1"
    assert v1_results[0].text == "Product Pro tier is $20/month."


def test_parent_child_provenance_preservation() -> None:
    """Test that parent_chunk_id and chunk_level are stored in payload and retrieved in search."""
    store = QdrantVectorStore()
    store.ensure_collection()

    parent_id = str(uuid.uuid4())
    child_chunk = create_test_chunk(
        chunk_level=1,
        parent_chunk_id=parent_id,
        content="Detailed child clause on SLA terms.",
    )
    entry_child = VectorStoreEntry.from_chunk_and_embedding(
        child_chunk, create_test_embedding(child_chunk)
    )

    store.upsert_entries([entry_child])

    # Search with parent filter
    parent_filter = VectorSearchFilter(parent_chunk_id=parent_id)
    results = store.search(query_vector=[0.1] * 1024, top_k=5, filters=parent_filter)

    assert len(results) == 1
    assert results[0].parent_chunk_id == parent_id
    assert results[0].chunk_level == 1


def test_delete_by_ids_and_delete_by_filter() -> None:
    """Test deleting points by IDs and by payload filter."""
    store = QdrantVectorStore()
    store.ensure_collection()

    chunk_1 = create_test_chunk(company_id="comp_to_delete")
    chunk_2 = create_test_chunk(company_id="comp_to_keep")
    entry_1 = VectorStoreEntry.from_chunk_and_embedding(chunk_1, create_test_embedding(chunk_1))
    entry_2 = VectorStoreEntry.from_chunk_and_embedding(chunk_2, create_test_embedding(chunk_2))

    store.upsert_entries([entry_1, entry_2])

    # Delete point 1 by id
    store.delete_by_ids([chunk_1.chunk_id])
    assert store.get_by_id(chunk_1.chunk_id) is None
    assert store.get_by_id(chunk_2.chunk_id) is not None

    # Delete point 2 by filter
    store.delete_by_filter(VectorSearchFilter(company_id="comp_to_keep"))
    assert store.get_by_id(chunk_2.chunk_id) is None


@pytest.mark.anyio
async def test_async_vector_store_operations() -> None:
    """Test asynchronous aensure_collection, aupsert_entries, asearch, aget_by_id."""
    store = QdrantVectorStore()
    await store.aensure_collection()

    chunk = create_test_chunk(content="Async vector point.")
    entry = VectorStoreEntry.from_chunk_and_embedding(chunk, create_test_embedding(chunk))

    await store.aupsert_entries([entry])

    result = await store.aget_by_id(chunk.chunk_id)
    assert result is not None
    assert result.chunk_id == chunk.chunk_id

    search_res = await store.asearch(query_vector=entry.vector, top_k=1)
    assert len(search_res) == 1
    assert search_res[0].chunk_id == chunk.chunk_id


def test_no_qdrant_types_leaked_outside_adapter() -> None:
    """Verify that domain interfaces and models do not import qdrant_client."""
    import competitive_intelligence.knowledge_pipeline.interfaces as iface_mod
    import competitive_intelligence.knowledge_pipeline.models as model_mod

    for mod in (iface_mod, model_mod):
        mod_file = getattr(mod, "__file__", "")
        assert mod_file != ""
        with open(mod_file, "r", encoding="utf-8") as f:
            code = f.read()

        assert "qdrant_client" not in code
        assert "PointStruct" not in code
        assert "ScoredPoint" not in code


# ==============================================================================
# SearchConfiguration & Parameter Translation Tests
# ==============================================================================


def test_search_configuration_defaults_and_validation() -> None:
    """Test default values and parameter validation for SearchConfiguration."""
    config = SearchConfiguration()
    assert config.top_k == 10
    assert config.hnsw_ef is None
    assert config.exact_search is False
    assert config.score_threshold is None

    # Custom valid config
    custom = SearchConfiguration(top_k=25, hnsw_ef=64, exact_search=True, score_threshold=0.75)
    assert custom.top_k == 25
    assert custom.hnsw_ef == 64
    assert custom.exact_search is True
    assert custom.score_threshold == 0.75

    # Validation errors
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        SearchConfiguration(top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        SearchConfiguration(top_k=-5)

    with pytest.raises(ValueError, match="hnsw_ef must be a positive integer if specified"):
        SearchConfiguration(hnsw_ef=0)

    with pytest.raises(ValueError, match="hnsw_ef must be a positive integer if specified"):
        SearchConfiguration(hnsw_ef=-10)


def test_search_with_search_configuration_top_k() -> None:
    """Test that search respects top_k specified in SearchConfiguration."""
    store = QdrantVectorStore()
    store.ensure_collection()

    entries: list[VectorStoreEntry] = []
    for i in range(10):
        chunk = create_test_chunk(content=f"Document paragraph {i}")
        emb = create_test_embedding(chunk, seed_val=0.05 * (i + 1))
        entries.append(VectorStoreEntry.from_chunk_and_embedding(chunk, emb))

    store.upsert_entries(entries)

    search_cfg = SearchConfiguration(top_k=4)
    results = store.search(query_vector=[0.1] * 1024, search_config=search_cfg)
    assert len(results) == 4


def test_search_params_translation_to_qdrant_client() -> None:
    """Test that QdrantVectorStore translates hnsw_ef and exact_search into Qdrant SearchParams."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.query_points.return_value = MagicMock(points=[])

    store = QdrantVectorStore(client=mock_client)

    # 1. Search with default (no search_config) -> search_params=None
    _ = store.search(query_vector=[0.1] * 1024, top_k=5)
    mock_client.query_points.assert_called_with(
        collection_name="ci_knowledge_chunks",
        query=[0.1] * 1024,
        limit=5,
        query_filter=None,
        search_params=None,
        score_threshold=None,
        with_payload=True,
    )

    # 2. Search with SearchConfiguration(hnsw_ef=128, exact_search=True, score_threshold=0.8)
    cfg = SearchConfiguration(top_k=7, hnsw_ef=128, exact_search=True, score_threshold=0.8)
    _ = store.search(query_vector=[0.1] * 1024, search_config=cfg)

    call_args = mock_client.query_points.call_args[1]
    assert call_args["limit"] == 7
    assert call_args["score_threshold"] == 0.8
    search_params = call_args["search_params"]
    assert search_params is not None
    assert search_params.hnsw_ef == 128
    assert search_params.exact is True

    # 3. Search with SearchConfiguration(hnsw_ef=64, exact_search=False)
    cfg_hnsw = SearchConfiguration(top_k=3, hnsw_ef=64, exact_search=False)
    _ = store.search(query_vector=[0.1] * 1024, search_config=cfg_hnsw)

    call_args_hnsw = mock_client.query_points.call_args[1]
    assert call_args_hnsw["limit"] == 3
    search_params_hnsw = call_args_hnsw["search_params"]
    assert search_params_hnsw is not None
    assert search_params_hnsw.hnsw_ef == 64
    assert search_params_hnsw.exact is False
