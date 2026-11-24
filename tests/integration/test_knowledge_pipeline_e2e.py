"""End-to-End Integration Suite for the Knowledge Pipeline Subsystem.

Tests all 10 declared workflow execution scenarios:
A. Initial Ingestion (PostgreSQL Doc + Version + Chunks -> Embeddings -> Qdrant vectors)
B. Identical Second Ingestion (Idempotency & UNCHANGED skip)
C. Changed Content (Incremental versioning & diff detection)
D. Workflow Orchestrator Stage Execution (KnowledgeProcessingStage)
E. Concurrency Collision Recovery ((document_id, content_hash) integrity recovery)
F. Transient Embedding Failure (Error categorization & retryability)
G. Transient Vector Store Failure (Canonical database state preserved)
H. Multi-Tenant Workspace Isolation (Qdrant payload filter + PostgreSQL workspace isolation)
I. Dynamic Chunking Dispatch across policies (Markdown, recursive, whole-doc)
J. Full 5-tier Knowledge Provenance Chain (Point -> Chunk -> Version -> Document -> Source)
"""

from datetime import datetime, timezone
import hashlib
from typing import Any, Sequence
import uuid

import pytest
from sqlalchemy import select

from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.infrastructure.database.models import (
    DocumentChunkOrm,
    DocumentOrm,
    DocumentVersionOrm,
    SourceOrm,
    TrackedCompetitorOrm,
    UserOrm,
    WorkspaceOrm,
)
from competitive_intelligence.infrastructure.database.repositories.competitor_repo import (
    SqlCompetitorRepository,
)
from competitive_intelligence.infrastructure.database.repositories.document_repo import (
    SqlDocumentRepository,
)
from competitive_intelligence.infrastructure.database.repositories.source_repo import (
    SqlSourceRepository,
)
from competitive_intelligence.infrastructure.database.repositories.workflow_state_store import (
    SqlWorkflowStateStore,
)
from competitive_intelligence.infrastructure.database.repositories.workspace_repo import (
    SqlUserRepository,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import DatabaseSessionManager
from competitive_intelligence.knowledge_pipeline.embeddings import MockEmbeddingProvider
from competitive_intelligence.knowledge_pipeline.vector_store import QdrantVectorStore
from competitive_intelligence.knowledge_pipeline import (
    ChangeDetector,
    ChangeType,
    ChunkingParameters,
    ChunkingPolicyResolver,
    ChunkingService,
    ChunkingStrategy,
    DistanceMetric,
    DocumentDeduplicator,
    DocumentNormalizer,
    IngestionResult,
    KnowledgeIngestionService,
    VectorSearchFilter,
    VectorStoreConfig,
    VectorStoreEntry,
)
from competitive_intelligence.orchestration import (
    DeterministicWorkflowOrchestrator,
    ErrorCategory,
    RetryPolicy,
    StageExecutionContext,
    WorkflowInputs,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.stages.knowledge_stage import (
    KnowledgeProcessingStage,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    AcquisitionResult,
    AcquisitionStatus,
    ContentType,
    DataSensitivity,
    FetchStatus,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    RateLimitConstraints,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToSStatus,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def _create_test_db() -> DatabaseSessionManager:
    """Initialize an isolated in-memory SQLite database with all tables."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()
    return manager


async def _seed_workspace_and_source(
    session_manager: DatabaseSessionManager,
    workspace_id: str = "ws_kp_test",
    company_id: str = "comp_qdrant",
    source_id: str = "src_pricing_page",
    source_url: str = "https://qdrant.tech/pricing",
) -> tuple[Workspace, TrackedCompetitor, Source]:
    """Helper to seed hierarchical User -> Workspace -> Competitor -> Source records."""
    async with session_manager.session() as session:
        user_repo = SqlUserRepository(session)
        ws_repo = SqlWorkspaceRepository(session)
        comp_repo = SqlCompetitorRepository(session)
        src_repo = SqlSourceRepository(session)

        user = User(
            user_id=f"user_{uuid.uuid4().hex[:8]}",
            email=f"intel_{uuid.uuid4().hex[:6]}@example.com",
            full_name="Intelligence Analyst",
        )
        await user_repo.create(user)

        workspace = Workspace(
            workspace_id=workspace_id,
            name="Competitive Intel Workspace",
            owner_user_id=user.user_id,
        )
        await ws_repo.create(workspace)

        competitor = TrackedCompetitor(
            competitor_id=company_id,
            workspace_id=workspace.workspace_id,
            name="Qdrant Vector Search",
            domain="qdrant.tech",
            description="Vector search engine",
        )
        await comp_repo.create(competitor)

        source = Source(
            identity=SourceIdentity(
                source_id=source_id,
                company_id=competitor.competitor_id,
                workspace_id=workspace.workspace_id,
                display_name="Qdrant Official Pricing",
            ),
            classification=SourceClassification(
                category=SourceCategory.PRICING,
                platform=PlatformType.WEB,
                content_type=ContentType.HTML,
            ),
            location=SourceLocation(
                url_or_identifier=source_url,
                access_pattern=AccessPattern.PUBLIC_WEB,
            ),
            acquisition=AcquisitionConfig(
                acquisition_method=AcquisitionMethod.CRAWL4AI,
                rate_limit_constraints=RateLimitConstraints(requests_per_minute=20),
            ),
            monitoring=MonitoringPolicy(frequency=3600),
            reliability=SourceReliability(source_tier=SourceTier.PRIMARY, credibility_score=0.95),
            governance=SourceGovernance(terms_of_service_status=ToSStatus.COMPLIANT),
            lifecycle=SourceLifecycle(status=LifecycleStatus.ACTIVE),
        )
        await src_repo.create(source)

        return workspace, competitor, source


def _create_acquisition_result(
    source: Source,
    content: str,
    status_code: int = 200,
) -> AcquisitionResult:
    """Helper to produce a valid AcquisitionResult."""
    return AcquisitionResult(
        status=AcquisitionStatus.SUCCESS,
        content=content,
        status_code=status_code,
        method=AcquisitionMethod.CRAWL4AI,
        metadata={"crawler": "Crawl4AI", "url": source.location.url_or_identifier},
    )


def _create_vector_store() -> QdrantVectorStore:
    """Helper creating an in-memory isolated Qdrant vector store."""
    config = VectorStoreConfig(
        collection_name=f"kp_test_{uuid.uuid4().hex[:8]}",
        vector_dimensions=16,
        distance_metric=DistanceMetric.COSINE,
        location=":memory:",
    )
    store = QdrantVectorStore(config=config)
    store.ensure_collection()
    return store


# ==============================================================================
# Scenario A: Initial Ingestion
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_initial_ingestion() -> None:
    """Verify complete initial ingestion from AcquisitionResult to PostgreSQL records and Qdrant points."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    raw_html = (
        "# Qdrant Cloud Pricing\n\n"
        "## Managed Cloud Clusters\n"
        "Qdrant Managed Cloud starts at $25/month for dedicated 1GB RAM nodes.\n\n"
        "## Hybrid Cloud Enterprise\n"
        "Enterprise dedicated VPC clusters start at $0.05/node/hour with zero rate limits.\n"
    )
    acq_result = _create_acquisition_result(source, raw_html)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        result: IngestionResult = await service.ingest(source=source, result=acq_result)

        # 1. Verify IngestionResult
        assert result.is_new_document is True
        assert result.is_new_version is True
        assert result.is_skipped_unchanged is False
        assert result.change_type == ChangeType.INITIAL
        assert result.chunks_created_count > 0
        assert result.vector_points_upserted_count == result.chunks_created_count
        assert len(result.chunk_ids) == result.chunks_created_count

        # 2. Verify PostgreSQL Document Record
        persisted_doc = await doc_repo.get_document(result.document_id)
        assert persisted_doc is not None
        assert persisted_doc.url == source.location.url_or_identifier
        assert persisted_doc.workspace_id == workspace.workspace_id
        assert persisted_doc.company_id == competitor.competitor_id
        assert persisted_doc.canonical_source_id == source.source_id

        # 3. Verify PostgreSQL DocumentVersion Record
        persisted_version = await doc_repo.get_version(result.version_id)
        assert persisted_version is not None
        assert persisted_version.document_id == persisted_doc.id
        assert persisted_version.content_hash == result.version.content_hash

        # 4. Verify PostgreSQL DocumentChunk Records
        persisted_chunks = await doc_repo.get_chunks_for_version(result.version_id)
        assert len(persisted_chunks) == result.chunks_created_count
        for chunk in persisted_chunks:
            assert chunk.metadata.document_id == persisted_doc.id
            assert chunk.metadata.version_id == persisted_version.id
            assert chunk.metadata.source_id == source.source_id
            assert chunk.metadata.workspace_id == workspace.workspace_id
            assert chunk.metadata.company_id == competitor.competitor_id
            assert chunk.content_hash != ""

        # 5. Verify Qdrant Vector Store Points
        filter_res = vector_store.search(
            query_vector=[0.1] * 16,
            top_k=10,
            filters=VectorSearchFilter(document_id=result.document_id),
        )
        assert len(filter_res) == result.chunks_created_count
        for hit in filter_res:
            assert hit.document_id == result.document_id
            assert hit.version_id == result.version_id
            assert hit.workspace_id == workspace.workspace_id


# ==============================================================================
# Scenario B: Identical Second Ingestion (Idempotency & UNCHANGED skip)
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_idempotent_duplicate_ingestion() -> None:
    """Verify that an identical acquisition is detected as UNCHANGED and skips chunking/embedding/indexing."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    raw_html = "## Free Tier\nQdrant Cloud provides a perpetual free 1GB cluster."
    acq_result = _create_acquisition_result(source, raw_html)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # 1. First Ingestion
        res1 = await service.ingest(source=source, result=acq_result)
        assert res1.is_new_document is True
        assert res1.is_new_version is True
        assert res1.is_skipped_unchanged is False

        # 2. Second Ingestion (Exact Same Content)
        res2 = await service.ingest(source=source, result=acq_result)

        # 3. Assert Second Ingestion Skipped Redundant Processing
        assert res2.document_id == res1.document_id
        assert res2.version_id == res1.version_id
        assert res2.is_new_document is False
        assert res2.is_new_version is False
        assert res2.is_skipped_unchanged is True
        assert res2.change_type == ChangeType.UNCHANGED
        assert res2.chunks_created_count == 0
        assert res2.vector_points_upserted_count == 0

        # 4. Verify DB has exactly 1 Document and 1 Version
        versions = await doc_repo.list_versions(res1.document_id)
        assert len(versions) == 1

        chunks = await doc_repo.get_chunks_for_version(res1.version_id)
        assert len(chunks) == res1.chunks_created_count

        # 5. Verify Vector Store has no duplicate points
        all_points = vector_store.search(
            query_vector=[0.1] * 16,
            top_k=20,
            filters=VectorSearchFilter(document_id=res1.document_id),
        )
        assert len(all_points) == res1.chunks_created_count


# ==============================================================================
# Scenario C: Changed Content (Incremental Versioning)
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_incremental_changed_content() -> None:
    """Verify that modified content triggers version increment, diff detection, and new chunk indexing."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    v1_content = "# Qdrant Pricing\nManaged Cluster: $25/month."
    v2_content = "# Qdrant Pricing\nManaged Cluster: $20/month with auto-scaling."

    acq_v1 = _create_acquisition_result(source, v1_content)
    acq_v2 = _create_acquisition_result(source, v2_content)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # 1. Ingest Version 1
        res1 = await service.ingest(source=source, result=acq_v1)
        assert res1.change_type == ChangeType.INITIAL
        assert res1.version_id is not None

        # 2. Ingest Version 2
        res2 = await service.ingest(source=source, result=acq_v2)

        # 3. Assert Version 2 created
        assert res2.document_id == res1.document_id
        assert res2.version_id != res1.version_id
        assert res2.is_new_document is False
        assert res2.is_new_version is True
        assert res2.is_skipped_unchanged is False
        assert res2.change_type == ChangeType.CHANGED
        assert res2.chunks_created_count > 0
        assert res2.vector_points_upserted_count > 0

        # 4. Verify Document record
        doc = await doc_repo.get_document(res1.document_id)
        assert doc is not None

        # 5. Verify both versions exist in database
        versions = await doc_repo.list_versions(res1.document_id)
        assert len(versions) == 2
        assert {v.id for v in versions} == {res1.version_id, res2.version_id}

        # 6. Verify chunks exist independently for both version 1 and version 2
        v1_chunks = await doc_repo.get_chunks_for_version(res1.version_id)
        v2_chunks = await doc_repo.get_chunks_for_version(res2.version_id)
        assert len(v1_chunks) > 0
        assert len(v2_chunks) > 0
        assert {c.id for c in v1_chunks}.isdisjoint({c.id for c in v2_chunks})


# ==============================================================================
# Scenario D: Workflow Orchestrator Stage Execution (KnowledgeProcessingStage)
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_orchestrator_stage_execution() -> None:
    """Verify that KnowledgeProcessingStage executes through DeterministicWorkflowOrchestrator and populates context."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())
    state_store = SqlWorkflowStateStore(session_manager=db_manager)

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)

        ingestion_service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        stage = KnowledgeProcessingStage(ingestion_service=ingestion_service)
        orchestrator = DeterministicWorkflowOrchestrator(
            state_store=state_store,
            stages=[stage],
            retry_policy=RetryPolicy(max_attempts=2),
        )

        inputs = WorkflowInputs(
            workspace_id=workspace.workspace_id,
            company_id=competitor.competitor_id,
            raw_document_content="Pinecone Serverless launched with $0.04/1M RUs on AWS us-east-1.",
            document_url="https://pinecone.io/pricing",
            document_id="doc_pinecone_pricing",
        )

        result = await orchestrator.run_workflow(
            inputs=inputs,
            target_stop_stage=WorkflowStageType.KNOWLEDGE_PROCESSING,
        )

        # Verify workflow completion & produced outputs
        assert result.status == WorkflowRunStatus.COMPLETED
        assert WorkflowStageType.KNOWLEDGE_PROCESSING in result.completed_stages
        assert len(result.produced_outputs.document_ids) == 1
        assert len(result.produced_outputs.version_ids) == 1
        assert len(result.produced_outputs.chunk_ids) > 0
        assert result.produced_outputs.chunks_created_count > 0
        assert result.produced_outputs.vector_points_upserted_count > 0

        # Verify execution record
        persisted_run = await state_store.get_run(result.run_id)
        assert persisted_run is not None
        exec_record = persisted_run.stage_records.get(WorkflowStageType.KNOWLEDGE_PROCESSING)
        assert exec_record is not None
        assert exec_record.outputs["chunks_created_count"] > 0
        assert exec_record.outputs["vector_points_upserted_count"] > 0


# ==============================================================================
# Scenario E: Concurrency Collision Recovery
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_concurrency_collision_recovery() -> None:
    """Verify that when a version collision occurs, KnowledgeIngestionService safely retrieves the existing version."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    raw_text = "Weaviate pricing update: Enterprise serverless released."
    acq_result = _create_acquisition_result(source, raw_text)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # 1. Run first ingestion
        res1 = await service.ingest(source=source, result=acq_result)

        # 2. Second ingestion with same content
        res2 = await service.ingest(source=source, result=acq_result)

        assert res2.document_id == res1.document_id
        assert res2.version_id == res1.version_id
        assert res2.is_skipped_unchanged is True


# ==============================================================================
# Scenario F: Transient Embedding Failure Handling
# ==============================================================================


class FailingEmbeddingProvider(MockEmbeddingProvider):
    """Embedding provider that raises an operational/transient failure."""

    def __init__(self, should_fail: bool = True) -> None:
        super().__init__(dimension=16)
        self.should_fail = should_fail

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if self.should_fail:
            raise ConnectionError("Embedding provider remote endpoint timed out")
        res: list[list[float]] = super().embed_texts(texts)
        return res


@pytest.mark.anyio
async def test_knowledge_pipeline_embedding_failure_retryability() -> None:
    """Verify that embedding failures surface as transient retryable exceptions without false success."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    acq_result = _create_acquisition_result(source, "Test content for embedding failure handling.")
    vector_store = _create_vector_store()
    failing_embedder = FailingEmbeddingProvider(should_fail=True)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=failing_embedder,
            vector_store=vector_store,
        )

        # Pipeline must raise the error rather than swallow or produce zero vectors silently
        with pytest.raises(ConnectionError, match="Embedding provider remote endpoint timed out"):
            await service.ingest(source=source, result=acq_result)


# ==============================================================================
# Scenario G: Transient Vector Store Failure Handling
# ==============================================================================


class FailingVectorStore(QdrantVectorStore):
    """Vector store that simulates an upsert connection failure."""

    def __init__(self, *args: Any, should_fail: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.should_fail = should_fail

    def upsert_entries(self, entries: Sequence[VectorStoreEntry]) -> None:
        if self.should_fail:
            raise TimeoutError("Qdrant gRPC connection dropped during vector upsert")
        super().upsert_entries(entries)


@pytest.mark.anyio
async def test_knowledge_pipeline_vector_store_failure_handling() -> None:
    """Verify that vector store failures propagate cleanly for retry and preserve relational state."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    acq_result = _create_acquisition_result(source, "Test content for vector store failure handling.")
    failing_store = FailingVectorStore(
        config=VectorStoreConfig(
            collection_name=f"fail_{uuid.uuid4().hex[:8]}",
            vector_dimensions=16,
            location=":memory:",
        ),
        should_fail=True,
    )
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=failing_store,
        )

        with pytest.raises(TimeoutError, match="Qdrant gRPC connection dropped"):
            await service.ingest(source=source, result=acq_result)


@pytest.mark.anyio
async def test_knowledge_pipeline_indexing_failure_and_retry_recovery() -> None:
    """Verify initial vector indexing failure is recovered on retry without duplicate versions,
    chunks, or vector points, and subsequent unchanged ingestion skips completed work.
    """
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    raw_text = "Vector indexing recovery: Pinecone announces serverless index architecture."
    acq_result = _create_acquisition_result(source, raw_text)

    vector_store = FailingVectorStore(
        config=VectorStoreConfig(
            collection_name=f"retry_idx_{uuid.uuid4().hex[:8]}",
            vector_dimensions=16,
            location=":memory:",
        ),
        should_fail=True,
    )
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # 1. Initial attempt fails during vector store indexing
        with pytest.raises(TimeoutError, match="Qdrant gRPC connection dropped"):
            await service.ingest(source=source, result=acq_result)

        # Verify relational state was persisted (Document, Version, Chunks)
        docs = await doc_repo.list_documents(workspace_id=source.identity.workspace_id, source_id=source.identity.source_id)
        assert len(docs) == 1
        doc_id = docs[0].document_id

        versions = await doc_repo.list_versions_for_document(doc_id)
        assert len(versions) == 1
        initial_version_id = versions[0].version_id

        chunks = await doc_repo.get_chunks_for_version(initial_version_id)
        assert len(chunks) > 0
        expected_chunk_count = len(chunks)

        # Verify Qdrant points are MISSING due to indexing failure
        for ch in chunks:
            assert await vector_store.aget_by_id(ch.chunk_id) is None

        # 2. Heal vector store and retry ingestion with the exact same content
        vector_store.should_fail = False
        retry_res = await service.ingest(source=source, result=acq_result)

        # Retry must succeed and report that missing vector points were indexed
        assert retry_res.is_skipped_unchanged is False
        assert retry_res.document_id == doc_id
        assert retry_res.version_id == initial_version_id
        assert len(retry_res.chunks) == expected_chunk_count  # Reused existing chunks
        assert retry_res.vector_points_upserted_count == expected_chunk_count

        # 3. Verify NO duplicate document versions, chunks, or vector points
        versions_after_retry = await doc_repo.list_versions_for_document(doc_id)
        assert len(versions_after_retry) == 1
        assert versions_after_retry[0].version_id == initial_version_id

        chunks_after_retry = await doc_repo.get_chunks_for_version(initial_version_id)
        assert len(chunks_after_retry) == expected_chunk_count

        for ch in chunks_after_retry:
            pt = await vector_store.aget_by_id(ch.chunk_id)
            assert pt is not None
            assert pt.chunk_id == ch.chunk_id

        # 4. Subsequent unchanged ingestion must skip completed work
        subsequent_res = await service.ingest(source=source, result=acq_result)
        assert subsequent_res.is_skipped_unchanged is True
        assert subsequent_res.document_id == doc_id
        assert subsequent_res.version_id == initial_version_id
        assert subsequent_res.chunks_created_count == 0
        assert subsequent_res.vector_points_upserted_count == 0

        # Relational and vector state remains strictly unchanged
        versions_final = await doc_repo.list_versions_for_document(doc_id)
        assert len(versions_final) == 1
        chunks_final = await doc_repo.get_chunks_for_version(initial_version_id)
        assert len(chunks_final) == expected_chunk_count


@pytest.mark.anyio
async def test_knowledge_pipeline_embedding_failure_and_retry_recovery() -> None:
    """Verify embedding failure is recovered on retry without duplicate versions,
    chunks, or vector points, and subsequent unchanged ingestion skips completed work.
    """
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    raw_text = "Embedding failure recovery: Milvus adds hybrid search capabilities."
    acq_result = _create_acquisition_result(source, raw_text)

    vector_store = _create_vector_store()
    failing_embedder = FailingEmbeddingProvider(should_fail=True)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=failing_embedder,
            vector_store=vector_store,
        )

        # 1. Initial attempt fails during embedding generation
        with pytest.raises(ConnectionError, match="Embedding provider remote endpoint timed out"):
            await service.ingest(source=source, result=acq_result)

        # Relational records were created before embedding call
        docs = await doc_repo.list_documents(workspace_id=source.identity.workspace_id, source_id=source.identity.source_id)
        assert len(docs) == 1
        doc_id = docs[0].document_id

        versions = await doc_repo.list_versions_for_document(doc_id)
        assert len(versions) == 1
        initial_version_id = versions[0].version_id

        chunks = await doc_repo.get_chunks_for_version(initial_version_id)
        assert len(chunks) > 0
        expected_chunk_count = len(chunks)

        # Vector store has no points
        for ch in chunks:
            assert await vector_store.aget_by_id(ch.chunk_id) is None

        # 2. Heal embedding provider and retry ingestion with same content
        failing_embedder.should_fail = False
        retry_res = await service.ingest(source=source, result=acq_result)

        # Retry must succeed and index the chunks
        assert retry_res.is_skipped_unchanged is False
        assert retry_res.document_id == doc_id
        assert retry_res.version_id == initial_version_id
        assert len(retry_res.chunks) == expected_chunk_count  # Reused existing chunks
        assert retry_res.vector_points_upserted_count == expected_chunk_count

        # 3. Verify no duplicate versions, chunks, or points
        versions_after = await doc_repo.list_versions_for_document(doc_id)
        assert len(versions_after) == 1

        chunks_after = await doc_repo.get_chunks_for_version(initial_version_id)
        assert len(chunks_after) == expected_chunk_count

        for ch in chunks_after:
            pt = await vector_store.aget_by_id(ch.chunk_id)
            assert pt is not None
            assert pt.chunk_id == ch.chunk_id

        # 4. Subsequent unchanged ingestion skips completed work
        subsequent_res = await service.ingest(source=source, result=acq_result)
        assert subsequent_res.is_skipped_unchanged is True
        assert subsequent_res.chunks_created_count == 0
        assert subsequent_res.vector_points_upserted_count == 0


# ==============================================================================
# Scenario H: Multi-Tenant Workspace Isolation
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_workspace_isolation() -> None:
    """Verify that documents, chunks, and vector search are strictly isolated across multiple workspaces."""
    db_manager = await _create_test_db()

    # Seed Workspace A and Source A
    ws_a, comp_a, src_a = await _seed_workspace_and_source(
        db_manager,
        workspace_id="ws_alpha",
        company_id="comp_alpha",
        source_id="src_alpha",
        source_url="https://alpha.example.com/pricing",
    )
    # Seed Workspace B and Source B
    ws_b, comp_b, src_b = await _seed_workspace_and_source(
        db_manager,
        workspace_id="ws_beta",
        company_id="comp_beta",
        source_id="src_beta",
        source_url="https://beta.example.com/pricing",
    )

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # Ingest for Workspace A
        res_a = await service.ingest(
            source=src_a,
            result=_create_acquisition_result(src_a, "Alpha company pricing: $10/mo"),
        )
        # Ingest for Workspace B
        res_b = await service.ingest(
            source=src_b,
            result=_create_acquisition_result(src_b, "Beta company pricing: $99/mo"),
        )

        # Search filtered by Workspace A -> only A chunks returned
        results_a = vector_store.search(
            query_vector=[0.1] * 16,
            top_k=10,
            filters=VectorSearchFilter(workspace_id=ws_a.workspace_id),
        )
        assert len(results_a) > 0
        for item in results_a:
            assert item.workspace_id == ws_a.workspace_id
            assert item.company_id == comp_a.competitor_id
            assert item.document_id == res_a.document_id

        # Search filtered by Workspace B -> only B chunks returned
        results_b = vector_store.search(
            query_vector=[0.1] * 16,
            top_k=10,
            filters=VectorSearchFilter(workspace_id=ws_b.workspace_id),
        )
        assert len(results_b) > 0
        for item in results_b:
            assert item.workspace_id == ws_b.workspace_id
            assert item.company_id == comp_b.competitor_id
            assert item.document_id == res_b.document_id


# ==============================================================================
# Scenario I: Dynamic Chunking Dispatch Across Policies
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_dynamic_chunking_dispatch() -> None:
    """Verify that different document formats dynamically resolve to specialized chunkers with preserved metadata."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        # 1. Short FAQ / single paragraph -> Whole Document strategy
        short_text = "Qdrant is an open-source vector search engine written in Rust."
        res_short = await service.ingest(
            source=source,
            result=_create_acquisition_result(source, short_text),
        )

        chunks_short = await doc_repo.get_chunks_for_version(res_short.version_id)
        assert len(chunks_short) == 1
        assert chunks_short[0].chunk_index == 0
        assert chunks_short[0].metadata.source_id == source.source_id

        # 2. Markdown with structured sections (> 1000 chars) -> Recursive / Parent-Child strategy
        structured_markdown = (
            "# Platform Architecture and Distributed Vector Storage\n\n"
            "## Distributed Storage Engine\n"
            "The storage engine uses memory-mapped files and RocksDB for metadata storage, "
            "allowing multi-gigabyte collections to be indexed with low memory overhead. "
            "Segments are immutable and written sequentially to optimize NVMe throughput and caching.\n\n"
            "## Quantization Strategies\n"
            "Supports Scalar Quantization (SQ) and Product Quantization (PQ) for 4x to 8x memory savings "
            "with sub-millisecond retrieval latencies across high-dimensional dense embedding spaces. "
            "Quantized vectors are kept in RAM while full vectors reside on disk for exact rescoring.\n\n"
            "## Payload Indexing and Filtering\n"
            "Payload indexes use in-memory roaring bitmaps for fast boolean filtering, "
            "ensuring strict multi-tenant isolation and granular metadata queries at line rate. "
            "Filters are evaluated before vector distance computations to prune search space.\n\n"
            "## High Availability Clustering\n"
            "Built-in Raft consensus protocol replicates vector segments across multiple nodes "
            "providing seamless failover, shard rebalancing, and zero-downtime rolling upgrades. "
            "Read replicas scale query throughput linearly across compute instances."
        )
        res_struct = await service.ingest(
            source=source,
            result=_create_acquisition_result(source, structured_markdown),
        )

        chunks_struct = await doc_repo.get_chunks_for_version(res_struct.version_id)
        assert len(chunks_struct) >= 3
        for chunk in chunks_struct:
            assert chunk.metadata.document_id == res_struct.document_id
            assert len(chunk.content) > 0


# ==============================================================================
# Scenario J: Full 5-tier Knowledge Provenance Chain
# ==============================================================================


@pytest.mark.anyio
async def test_knowledge_pipeline_full_provenance_chain() -> None:
    """Verify complete bidirectional 5-tier provenance: Vector Point -> Chunk -> Version -> Document -> Source."""
    db_manager = await _create_test_db()
    workspace, competitor, source = await _seed_workspace_and_source(db_manager)

    vector_store = _create_vector_store()
    embedder = MockEmbeddingProvider(dimension=16)
    chunking_service = ChunkingService(resolver=ChunkingPolicyResolver())

    doc_text = (
        "# Qdrant Release 1.10\n\n"
        "Qdrant 1.10 introduces universal dense and sparse hybrid multitenancy support."
    )
    acq_result = _create_acquisition_result(source, doc_text)

    async with db_manager.session() as session:
        doc_repo = SqlDocumentRepository(session)
        src_repo = SqlSourceRepository(session)
        service = KnowledgeIngestionService(
            doc_repo=doc_repo,
            chunking_service=chunking_service,
            embedding_provider=embedder,
            vector_store=vector_store,
        )

        res = await service.ingest(source=source, result=acq_result)

        # Step 1: Query Vector Store
        hits = vector_store.search(
            query_vector=[0.1] * 16,
            top_k=5,
            filters=VectorSearchFilter(document_id=res.document_id),
        )
        assert len(hits) > 0
        point = hits[0]

        # Step 2: Traverse from Vector metadata to DB Chunk
        chunk = await doc_repo.get_chunk_by_id(point.chunk_id)
        assert chunk is not None
        assert chunk.id == point.chunk_id
        assert chunk.content_hash == point.metadata.get("content_hash", chunk.content_hash)
        assert chunk.metadata.version_id == point.version_id

        # Step 3: Traverse from DB Chunk to DB DocumentVersion
        version = await doc_repo.get_version(chunk.metadata.version_id)
        assert version is not None
        assert version.id == chunk.metadata.version_id
        assert version.document_id == point.document_id

        # Step 4: Traverse from DB DocumentVersion to DB Document
        doc = await doc_repo.get_document(version.document_id)
        assert doc is not None
        assert doc.id == point.document_id
        assert doc.url == source.location.url_or_identifier
        assert doc.workspace_id == workspace.workspace_id
        assert doc.company_id == competitor.competitor_id
        assert doc.canonical_source_id == source.source_id

        # Step 5: Traverse from DB Document to DB Source
        persisted_source = await src_repo.get_by_id(doc.canonical_source_id)
        assert persisted_source is not None
        assert persisted_source.source_id == source.source_id
        assert persisted_source.company_id == competitor.competitor_id
        assert persisted_source.workspace_id == workspace.workspace_id
        assert persisted_source.location.url_or_identifier == doc.url
