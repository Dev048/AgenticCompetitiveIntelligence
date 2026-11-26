"""Process-wide singletons for heavy/exclusive resources (embedding model, embedded Qdrant)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from competitive_intelligence.application.api import deps
from competitive_intelligence.core.config import Settings
from competitive_intelligence.knowledge_pipeline.embeddings import bge_m3_provider
from competitive_intelligence.knowledge_pipeline.embeddings.mock_provider import (
    MockEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    VectorSearchFilter,
    VectorStoreEntry,
)
from competitive_intelligence.knowledge_pipeline.vector_store.qdrant_store import (
    QdrantVectorStore,
)

CHUNK_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _clean_caches() -> Iterator[None]:
    deps.reset_resource_caches()
    yield
    deps.reset_resource_caches()


def _embedded_settings(path: Path) -> Settings:
    return Settings(environment="development", qdrant_path=str(path), qdrant_url=None)


def _entry(workspace_id: str = "ws_1") -> VectorStoreEntry:
    return VectorStoreEntry(
        chunk_id=CHUNK_ID,
        document_id="doc_1",
        version_id="ver_1",
        workspace_id=workspace_id,
        vector=[0.1] * 1024,
        text="Northwind cut its Starter plan from $49 to $29.",
    )


def test_embedded_qdrant_is_a_shared_singleton(tmp_path: Path) -> None:
    cfg = _embedded_settings(tmp_path / "qdrant")

    first = deps.get_vector_store(cfg)
    second = deps.get_vector_store(cfg)

    assert first is second
    assert isinstance(first, QdrantVectorStore)
    assert first.config.path == str(tmp_path / "qdrant")


def test_embedded_qdrant_data_survives_across_requests(tmp_path: Path) -> None:
    cfg = _embedded_settings(tmp_path / "qdrant")

    writer = deps.get_vector_store(cfg)
    writer.ensure_collection()
    writer.upsert_entries([_entry()])

    reader = deps.get_vector_store(cfg)
    hits = reader.search([0.1] * 1024, top_k=5, filters=VectorSearchFilter(workspace_id="ws_1"))
    assert [h.chunk_id for h in hits] == [CHUNK_ID]


def test_embedded_qdrant_persists_on_disk_after_reset(tmp_path: Path) -> None:
    cfg = _embedded_settings(tmp_path / "qdrant")
    store = deps.get_vector_store(cfg)
    store.ensure_collection()
    store.upsert_entries([_entry()])

    # Simulates a process restart: the folder lock is released and a new client reopens it.
    deps.reset_resource_caches()
    reopened = deps.get_vector_store(cfg)

    assert reopened is not store
    assert reopened.get_by_id(CHUNK_ID) is not None


def test_embedded_qdrant_keeps_workspace_isolation(tmp_path: Path) -> None:
    store = deps.get_vector_store(_embedded_settings(tmp_path / "qdrant"))
    store.ensure_collection()
    store.upsert_entries([_entry(workspace_id="ws_a")])

    hits = store.search([0.1] * 1024, top_k=5, filters=VectorSearchFilter(workspace_id="ws_b"))
    assert hits == []


def test_in_memory_store_behaviour_is_unchanged() -> None:
    cfg = Settings(environment="development", qdrant_location=":memory:", qdrant_url=None)

    assert deps.get_vector_store(cfg) is not deps.get_vector_store(cfg)


def test_bge_m3_model_is_loaded_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict[str, Any]] = []

    class _FakeProvider(MockEmbeddingProvider):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(dimension=1024)
            constructed.append(kwargs)

    monkeypatch.setattr(bge_m3_provider, "BGEM3EmbeddingProvider", _FakeProvider)
    cfg = Settings(environment="development", embedding_provider="bge_m3")

    first = deps.get_embedding_service(cfg)
    second = deps.get_embedding_service(cfg)

    assert len(constructed) == 1
    assert first.provider is second.provider


def test_production_rejects_embedded_qdrant(tmp_path: Path) -> None:
    cfg = Settings(
        environment="production",
        database_url="postgresql+asyncpg://u:p@db:5432/ci",
        llm_provider="openai",
        openai_api_key="sk-test-key-not-real",
        embedding_provider="bge_m3",
        qdrant_url="http://qdrant:6333",
        qdrant_path=str(tmp_path / "qdrant"),
    )

    errors = cfg.validate_production_configuration(raise_exception=False)

    assert any("QDRANT_PATH" in e for e in errors)
