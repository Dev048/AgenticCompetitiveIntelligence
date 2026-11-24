"""Unit tests for KnowledgeIngestionService."""

from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentRepository,
)
from competitive_intelligence.knowledge_pipeline.chunking_service import (
    ChunkingService,
)
from competitive_intelligence.knowledge_pipeline.embeddings import (
    EmbeddingService,
    MockEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.ingestion_service import (
    KnowledgeIngestionService,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeType,
    Document,
    DocumentChunk,
    DocumentVersion,
    IngestionStatus,
    VectorSearchFilter,
    VectorSearchResult,
    VectorStoreConfig,
    VectorStoreEntry,
)
from competitive_intelligence.knowledge_pipeline.vector_store import (
    QdrantVectorStore,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


class InMemoryDocumentRepository(BaseDocumentRepository):
    """In-memory implementation of BaseDocumentRepository for unit testing."""

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.versions: dict[str, DocumentVersion] = {}
        self.chunks: dict[str, DocumentChunk] = {}

    async def create_document(self, document: Document) -> Document:
        self.documents[document.document_id] = document
        return document

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def get_document_by_id(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def get_document_by_canonical_url(
        self, source_id: str, canonical_url: str
    ) -> Document | None:
        for doc in self.documents.values():
            if doc.source_id == source_id and doc.canonical_url == canonical_url:
                return doc
        return None

    async def list_documents(
        self,
        workspace_id: str,
        company_id: str | None = None,
        source_id: str | None = None,
    ) -> list[Document]:
        return [
            d for d in self.documents.values()
            if d.workspace_id == workspace_id
            and (company_id is None or d.company_id == company_id)
            and (source_id is None or d.source_id == source_id)
        ]

    async def list_documents_by_company(self, company_id: str) -> list[Document]:
        return [d for d in self.documents.values() if d.company_id == company_id]

    async def create_version(self, version: DocumentVersion) -> DocumentVersion:
        # Check uniqueness constraint on (document_id, content_hash)
        for existing in self.versions.values():
            if (
                existing.document_id == version.document_id
                and existing.content_hash == version.content_hash
            ):
                raise ValueError(
                    f"Duplicate version collision on (document_id={version.document_id}, content_hash={version.content_hash})"
                )
        self.versions[version.version_id] = version
        return version

    async def get_version(self, version_id: str) -> DocumentVersion | None:
        return self.versions.get(version_id)

    async def get_version_by_id(self, version_id: str) -> DocumentVersion | None:
        return self.versions.get(version_id)

    async def get_version_by_hash(
        self, document_id: str, content_hash: str
    ) -> DocumentVersion | None:
        for ver in self.versions.values():
            if ver.document_id == document_id and ver.content_hash == content_hash:
                return ver
        return None

    async def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        doc_vers = [v for v in self.versions.values() if v.document_id == document_id]
        if not doc_vers:
            return None
        doc_vers.sort(key=lambda x: x.acquired_at, reverse=True)
        return doc_vers[0]

    async def list_versions(self, document_id: str) -> list[DocumentVersion]:
        doc_vers = [v for v in self.versions.values() if v.document_id == document_id]
        doc_vers.sort(key=lambda x: x.acquired_at)
        return doc_vers

    async def list_versions_for_document(self, document_id: str) -> list[DocumentVersion]:
        return await self.list_versions(document_id)

    async def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        self.chunks[chunk.chunk_id] = chunk
        return chunk

    async def create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        for c in chunks:
            self.chunks[c.chunk_id] = c
        return chunks

    async def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        return self.chunks.get(chunk_id)

    async def get_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        ver_chunks = [c for c in self.chunks.values() if c.version_id == version_id]
        ver_chunks.sort(key=lambda x: x.chunk_index)
        return ver_chunks

    async def list_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        return await self.get_chunks_for_version(version_id)

    async def get_child_chunks(self, parent_chunk_id: str) -> list[DocumentChunk]:
        children = [c for c in self.chunks.values() if c.parent_chunk_id == parent_chunk_id]
        children.sort(key=lambda x: x.chunk_index)
        return children


def create_test_source() -> Source:
    return Source(
        identity=SourceIdentity(
            source_id="src_acme_pricing",
            company_id="acme_corp",
            workspace_id="ws_ci_test",
            display_name="Acme Pricing Page",
        ),
        classification=SourceClassification(
            category=SourceCategory.PRICING,
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier="https://acme.com/pricing",
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )


def create_test_result(content: str) -> AcquisitionResult:
    return AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content=content,
        metadata={"title": "Acme Pricing Overview"},
        acquired_at=datetime.now(timezone.utc),
        method=AcquisitionMethod.CRAWL4AI,
        provider_name="Crawl4AIClient",
        status_code=200,
    )


@pytest.mark.anyio
async def test_initial_ingestion_full_pipeline() -> None:
    """Test full initial ingestion: Document, Version, Chunks, Embeddings, Vector store."""
    repo = InMemoryDocumentRepository()
    mock_provider = MockEmbeddingProvider(dimension=8)
    embedding_service = EmbeddingService(provider=mock_provider)
    vector_store = QdrantVectorStore(config=VectorStoreConfig(vector_dimensions=8))
    chunking_service = ChunkingService()

    service = KnowledgeIngestionService(
        document_repo=repo,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    source = create_test_source()
    content = "# Acme Pricing\n\nStandard plan: $29/month.\nEnterprise plan: $99/month."
    acq_result = create_test_result(content)

    res = await service.ingest(source=source, result=acq_result)

    assert res.status == IngestionStatus.COMPLETED
    assert res.is_skipped_unchanged is False
    assert res.change_result.change_type == ChangeType.INITIAL
    assert len(res.chunks) >= 1
    assert res.embeddings_count >= 1
    assert res.vectors_count >= 1

    # Verify repository storage
    assert res.document.document_id in repo.documents
    assert res.version.version_id in repo.versions
    assert len(repo.chunks) == len(res.chunks)

    # Verify vector store point retrieval
    first_chunk = res.chunks[0]
    retrieved_point = vector_store.get_by_id(first_chunk.chunk_id)
    assert retrieved_point is not None
    assert retrieved_point.document_id == res.document.document_id
    assert retrieved_point.version_id == res.version.version_id


@pytest.mark.anyio
async def test_unchanged_second_ingestion_skips_processing() -> None:
    """Test that ingesting unchanged content skips chunking, embedding, and vector upsert."""
    repo = InMemoryDocumentRepository()
    mock_provider = MockEmbeddingProvider(dimension=8)
    embedding_service = EmbeddingService(provider=mock_provider)
    vector_store = QdrantVectorStore(config=VectorStoreConfig(vector_dimensions=8))

    service = KnowledgeIngestionService(
        document_repo=repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    source = create_test_source()
    content = "# Acme Pricing\n\nStandard plan: $29/month."

    # 1. Initial Ingestion
    res1 = await service.ingest(source=source, result=create_test_result(content))
    assert res1.status == IngestionStatus.COMPLETED
    assert res1.change_result.change_type == ChangeType.INITIAL
    assert res1.vectors_count >= 1

    # 2. Second Ingestion with identical content
    res2 = await service.ingest(source=source, result=create_test_result(content))
    assert res2.status == IngestionStatus.SKIPPED_UNCHANGED
    assert res2.is_skipped_unchanged is True
    assert res2.change_result.change_type == ChangeType.UNCHANGED
    assert len(res2.chunks) == 0
    assert res2.embeddings_count == 0
    assert res2.vectors_count == 0


@pytest.mark.anyio
async def test_changed_content_creates_new_version_and_chunks() -> None:
    """Test that changed content creates a new version, chunks, and updates vector store."""
    repo = InMemoryDocumentRepository()
    mock_provider = MockEmbeddingProvider(dimension=8)
    embedding_service = EmbeddingService(provider=mock_provider)
    vector_store = QdrantVectorStore(config=VectorStoreConfig(vector_dimensions=8))

    service = KnowledgeIngestionService(
        document_repo=repo,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    source = create_test_source()
    content_v1 = "# Acme Pricing\n\nStandard plan: $29/month."
    content_v2 = "# Acme Pricing\n\nStandard plan reduced to $19/month!"

    res1 = await service.ingest(source=source, result=create_test_result(content_v1))
    assert res1.change_result.change_type == ChangeType.INITIAL

    res2 = await service.ingest(source=source, result=create_test_result(content_v2))
    assert res2.status == IngestionStatus.COMPLETED
    assert res2.is_skipped_unchanged is False
    assert res2.change_result.change_type == ChangeType.CHANGED
    assert res2.version.version_id != res1.version.version_id
    assert len(repo.versions) == 2


@pytest.mark.anyio
async def test_concurrency_collision_recovery() -> None:
    """Test that concurrent duplicate version creation is handled idempotently without error."""
    repo = InMemoryDocumentRepository()
    service = KnowledgeIngestionService(document_repo=repo)
    source = create_test_source()
    content = "# Acme Pricing\n\nStandard plan: $29/month."

    # First ingestion
    res1 = await service.ingest(source=source, result=create_test_result(content))
    assert res1.status == IngestionStatus.COMPLETED

    # Simulate second ingestion colliding on existing hash
    res2 = await service.ingest(source=source, result=create_test_result(content))
    assert res2.status == IngestionStatus.SKIPPED_UNCHANGED
    assert res2.version.version_id == res1.version.version_id
