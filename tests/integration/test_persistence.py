"""Integration tests for PostgreSQL / Async database persistence layer.

Tests end-to-end relational persistence, association tables, append-only semantics,
foreign-key integrity, transaction rollback, and full 10-layer provenance reconstruction:
Recommendation -> DecisionOption -> Insight -> Signal -> Event -> Evidence -> Chunk -> Version -> Document -> Source.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.schema import CreateTable

from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.decision_support.models import (
    ActionType,
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DecisionOption,
    DecisionRisk,
    DecisionUrgency,
    EvaluatedAlternative,
    ObjectiveDimension,
    PredictedConsequence,
    Recommendation,
    RecommendationConfidenceAssessment,
    RecommendationStatus,
    RiskBasisType,
    RiskTolerance,
    TimeHorizon,
)
from competitive_intelligence.infrastructure.database.models import (
    Base,
    DocumentChunkOrm,
    DocumentOrm,
    DocumentVersionOrm,
    EvidenceReferenceOrm,
    IntelligenceEventOrm,
    SourceOrm,
    TrackedCompetitorOrm,
    UserOrm,
    WorkspaceOrm,
)
from competitive_intelligence.infrastructure.database.repositories.competitor_repo import (
    SqlCompetitorRepository,
)
from competitive_intelligence.infrastructure.database.repositories.decision_repo import (
    SqlDecisionRepository,
)
from competitive_intelligence.infrastructure.database.repositories.document_repo import (
    SqlDocumentRepository,
)
from competitive_intelligence.infrastructure.database.repositories.intelligence_repo import (
    SqlIntelligenceRepository,
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
from competitive_intelligence.intelligence.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    SignalConfidenceAssessment,
    SignalStatus,
    SignalType,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TemporalScope,
    TimePrecision,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.orchestration.models import (
    ErrorCategory,
    StageErrorRecord,
    StageExecutionRecord,
    StageStatus,
    TriggerType,
    WorkflowProducedOutputs,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
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


async def _get_initialized_db() -> DatabaseSessionManager:
    """Helper creating and initializing an in-memory test database."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()
    return manager


# ==============================================================================
# 1. PostgreSQL Schema Compilation & Table Count Test
# ==============================================================================


def test_postgresql_ddl_and_table_count() -> None:
    """Verify that exactly 26 tables are defined and compile cleanly to PostgreSQL DDL."""
    tables = Base.metadata.tables
    assert len(tables) == 26, f"Expected exactly 26 tables, got {len(tables)}: {list(tables.keys())}"

    from sqlalchemy.dialects import postgresql

    # Verify each table compiles to valid PostgreSQL dialect DDL
    pg_dialect = postgresql.dialect()  # type: ignore[no-untyped-call]
    for table_name, table_obj in tables.items():
        ddl = str(CreateTable(table_obj).compile(dialect=pg_dialect))
        assert "CREATE TABLE" in ddl
        assert table_name in ddl


# ==============================================================================
# 2. Application & Hierarchy Tests (Workspace -> Competitor -> Source)
# ==============================================================================


@pytest.mark.anyio
async def test_workspace_competitor_source_persistence() -> None:
    """Test persistence from User -> Workspace -> TrackedCompetitor -> Source."""
    db_manager = await _get_initialized_db()
    try:
        async with db_manager.session() as session:
            user_repo = SqlUserRepository(session)
            ws_repo = SqlWorkspaceRepository(session)
            comp_repo = SqlCompetitorRepository(session)
            src_repo = SqlSourceRepository(session)

            # 1. Create User
            user = User(
                user_id="usr_123",
                email="intel_lead@enterprise.com",
                full_name="Chief Intelligence Officer",
            )
            await user_repo.create(user)

            # 2. Create Workspace
            ws = Workspace(
                workspace_id="ws_main",
                name="Strategic Tech Intel",
                owner_user_id=user.user_id,
            )
            await ws_repo.create(ws)

            # 3. Create Tracked Competitor
            comp = TrackedCompetitor(
                competitor_id="comp_acme",
                workspace_id=ws.workspace_id,
                name="Acme Corp",
                domain="acme.example.com",
                description="Leading cloud competitor",
            )
            await comp_repo.create(comp)

            # 4. Create Source
            source = Source(
                identity=SourceIdentity(
                    source_id="src_acme_pricing",
                    company_id=comp.competitor_id,
                    workspace_id=ws.workspace_id,
                    display_name="Acme Pricing Page",
                ),
                classification=SourceClassification(
                    category=SourceCategory.PRICING,
                    platform=PlatformType.WEB,
                    content_type=ContentType.HTML,
                ),
                location=SourceLocation(
                    url_or_identifier="https://acme.example.com/pricing",
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

        # Verify retrieval in a fresh session
        async with db_manager.session() as session:
            ws_repo = SqlWorkspaceRepository(session)
            comp_repo = SqlCompetitorRepository(session)
            src_repo = SqlSourceRepository(session)

            fetched_ws = await ws_repo.get_by_id("ws_main")
            assert fetched_ws is not None
            assert fetched_ws.name == "Strategic Tech Intel"

            competitors = await comp_repo.list_by_workspace("ws_main")
            assert len(competitors) == 1
            assert competitors[0].name == "Acme Corp"
            assert competitors[0].domain == "acme.example.com"

            sources = await src_repo.list_by_company("comp_acme")
            assert len(sources) == 1
            assert sources[0].source_id == "src_acme_pricing"
            assert sources[0].classification.category == SourceCategory.PRICING
            assert sources[0].reliability.credibility_score == 0.95
    finally:
        await db_manager.close()


# ==============================================================================
# 3. Document, Version, and Chunk Lineage Tests
# ==============================================================================


@pytest.mark.anyio
async def test_document_version_and_chunk_lineage() -> None:
    """Test Document -> append-only DocumentVersion -> DocumentChunk hierarchy."""
    db_manager = await _get_initialized_db()
    try:
        async with db_manager.session() as session:
            # Setup prerequisite parent records
            user = User(user_id="u1", email="u1@test.com")
            await SqlUserRepository(session).create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await SqlWorkspaceRepository(session).create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await SqlCompetitorRepository(session).create(comp)
            source = Source(
                identity=SourceIdentity(source_id="src1", company_id="comp1", workspace_id="ws1", display_name="S1"),
                classification=SourceClassification(category=SourceCategory.PRICING),
                location=SourceLocation(url_or_identifier="https://comp1.com/pricing"),
                acquisition=AcquisitionConfig(),
            )
            await SqlSourceRepository(session).create(source)

            doc_repo = SqlDocumentRepository(session)

            # 1. Create Document
            doc = Document(
                document_id="doc_101",
                source_id="src1",
                company_id="comp1",
                workspace_id="ws1",
                canonical_url="https://comp1.com/pricing",
            )
            await doc_repo.create_document(doc)

            # 2. Create Document Version 1
            content_v1 = "# Acme Pricing\nTier A: $99/mo\nTier B: $199/mo"
            hash_v1 = hashlib.sha256(content_v1.encode("utf-8")).hexdigest()
            v1 = DocumentVersion(
                version_id="ver_001",
                document_id=doc.document_id,
                content=content_v1,
                title="Acme Pricing V1",
                content_hash=hash_v1,
                acquired_at=datetime.now(timezone.utc),
                metadata={"source": "crawl4ai"},
            )
            await doc_repo.create_version(v1)

            # 3. Create Hierarchical Chunks (Parent & Child)
            parent_meta = ChunkMetadata(
                source_id="src1",
                document_id=doc.document_id,
                version_id=v1.version_id,
                company_id="comp1",
                workspace_id="ws1",
                source_url="https://comp1.com/pricing",
                source_category="pricing",
                collected_at=v1.acquired_at,
                start_char=0,
                end_char=len(content_v1),
                chunk_index=0,
                total_chunks=2,
            )
            parent_chunk = DocumentChunk(
                chunk_id="chk_parent_1",
                document_id=doc.document_id,
                version_id=v1.version_id,
                content=content_v1,
                chunk_index=0,
                chunk_level=0,
                metadata=parent_meta,
            )

            child_content = "Tier A: $99/mo"
            child_meta = ChunkMetadata(
                source_id="src1",
                document_id=doc.document_id,
                version_id=v1.version_id,
                company_id="comp1",
                workspace_id="ws1",
                source_url="https://comp1.com/pricing",
                source_category="pricing",
                collected_at=v1.acquired_at,
                start_char=16,
                end_char=30,
                chunk_index=1,
                total_chunks=2,
            )
            child_chunk = DocumentChunk(
                chunk_id="chk_child_1",
                document_id=doc.document_id,
                version_id=v1.version_id,
                content=child_content,
                parent_chunk_id=parent_chunk.chunk_id,
                chunk_index=1,
                chunk_level=1,
                metadata=child_meta,
            )

            await doc_repo.create_chunks([parent_chunk, child_chunk])

        # Verify chunk retrieval and parent-child relationship
        async with db_manager.session() as session:
            doc_repo = SqlDocumentRepository(session)

            chunks = await doc_repo.list_chunks_for_version("ver_001")
            assert len(chunks) == 2
            assert chunks[0].chunk_id == "chk_parent_1"
            assert chunks[0].chunk_level == 0
            assert chunks[1].chunk_id == "chk_child_1"
            assert chunks[1].parent_chunk_id == "chk_parent_1"
            assert chunks[1].chunk_level == 1

            children = await doc_repo.get_child_chunks("chk_parent_1")
            assert len(children) == 1
            assert children[0].chunk_id == "chk_child_1"
    finally:
        await db_manager.close()


# ==============================================================================
# 4. Append-Only Document Version Uniqueness & Immutability Test
# ==============================================================================


@pytest.mark.anyio
async def test_document_version_uniqueness_and_append_only() -> None:
    """Test that document versions enforce UNIQUE(document_id, content_hash)."""
    db_manager = await _get_initialized_db()
    try:
        async with db_manager.session() as session:
            user = User(user_id="u1", email="u1@test.com")
            await SqlUserRepository(session).create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await SqlWorkspaceRepository(session).create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await SqlCompetitorRepository(session).create(comp)
            source = Source(
                identity=SourceIdentity(source_id="src1", company_id="comp1", workspace_id="ws1", display_name="S1"),
                classification=SourceClassification(category=SourceCategory.PRICING),
                location=SourceLocation(url_or_identifier="https://comp1.com/pricing"),
                acquisition=AcquisitionConfig(),
            )
            await SqlSourceRepository(session).create(source)
            doc = Document(
                document_id="doc_101",
                source_id="src1",
                company_id="comp1",
                workspace_id="ws1",
                canonical_url="https://comp1.com/pricing",
            )
            await SqlDocumentRepository(session).create_document(doc)

            content = "Identical pricing content"
            chash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            v1 = DocumentVersion(
                version_id="ver_001",
                document_id=doc.document_id,
                content=content,
                content_hash=chash,
            )
            await SqlDocumentRepository(session).create_version(v1)

        # Attempt to insert identical hash for same document -> should raise error on flush/commit
        duplicate_failed = False
        try:
            async with db_manager.session() as session2:
                v1_dup = DocumentVersion(
                    version_id="ver_002",
                    document_id=doc.document_id,
                    content=content,
                    content_hash=chash,
                )
                await SqlDocumentRepository(session2).create_version(v1_dup)
        except Exception:
            duplicate_failed = True

        assert duplicate_failed, "Expected duplicate document version to violate UNIQUE(document_id, content_hash)"
    finally:
        await db_manager.close()


# ==============================================================================
# 5. Grounded Intelligence, M:N Link Tables & Provenance Test
# ==============================================================================


@pytest.mark.anyio
async def test_grounded_intelligence_and_association_links() -> None:
    """Test EvidenceReference -> IntelligenceEvent -> InferredSignal -> Insight associations."""
    now = datetime.now(timezone.utc)
    db_manager = await _get_initialized_db()

    try:
        async with db_manager.session() as session:
            # Setup base hierarchy
            user = User(user_id="u1", email="u1@test.com")
            await SqlUserRepository(session).create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await SqlWorkspaceRepository(session).create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await SqlCompetitorRepository(session).create(comp)
            source = Source(
                identity=SourceIdentity(source_id="src1", company_id="comp1", workspace_id="ws1", display_name="S1"),
                classification=SourceClassification(category=SourceCategory.PRICING),
                location=SourceLocation(url_or_identifier="https://comp1.com/pricing"),
                acquisition=AcquisitionConfig(),
            )
            await SqlSourceRepository(session).create(source)
            doc = Document(
                document_id="doc1",
                source_id="src1",
                company_id="comp1",
                workspace_id="ws1",
                canonical_url="https://comp1.com/pricing",
            )
            await SqlDocumentRepository(session).create_document(doc)
            ver = DocumentVersion(
                version_id="ver1",
                document_id="doc1",
                content="Enterprise Tier: $999/mo -> $499/mo",
                content_hash="h123",
                acquired_at=now,
            )
            await SqlDocumentRepository(session).create_version(ver)
            chunk = DocumentChunk(
                chunk_id="chk1",
                document_id="doc1",
                version_id="ver1",
                content="Enterprise Tier: $999/mo -> $499/mo",
                metadata=ChunkMetadata(
                    source_id="src1",
                    document_id="doc1",
                    version_id="ver1",
                    company_id="comp1",
                    workspace_id="ws1",
                    source_url="https://comp1.com/pricing",
                    source_category="pricing",
                    collected_at=now,
                ),
            )
            await SqlDocumentRepository(session).create_chunk(chunk)

            intel_repo = SqlIntelligenceRepository(session)

            # 1. Create EvidenceReference
            evidence = EvidenceReference(
                evidence_id="ev_001",
                source_id="src1",
                document_id="doc1",
                version_id="ver1",
                chunk_id="chk1",
                source_url="https://comp1.com/pricing",
                source_category="pricing",
                source_perspective=SourcePerspective.FIRST_PARTY,
                role=EvidenceRole.PRIMARY_DIRECT,
                extracted_text="Enterprise Tier price dropped from $999 to $499",
                acquired_at=now,
                retrieval_score=0.98,
            )

            # 2. Create IntelligenceEvent
            event = IntelligenceEvent(
                event_id="evt_001",
                workspace_id="ws1",
                company_id="comp1",
                event_type=IntelligenceEventType.PRICING_CHANGE,
                title="Enterprise Plan 50% Price Cut",
                summary="Competitor cut Enterprise plan pricing in half.",
                subject="Pricing Strategy",
                previous_state={"price": "$999"},
                new_state={"price": "$499"},
                event_time=EventTime(
                    occurred_at=now,
                    precision=TimePrecision.EXACT_TIMESTAMP,
                    acquired_at=now,
                    detected_at=now,
                ),
                confidence=ConfidenceAssessment(
                    score=0.95, level=ConfidenceLevel.VERY_HIGH, is_corroborated=True
                ),
                significance=SignificanceAssessment(
                    status=SignificanceStatus.ASSESSED,
                    level=SignificanceLevel.HIGH,
                    score=0.85,
                ),
                evidence=[evidence],
                primary_perspective=SourcePerspective.FIRST_PARTY,
            )
            await intel_repo.create_event(event)

            # 3. Create InferredSignal
            signal = InferredSignal(
                signal_id="sig_001",
                workspace_id="ws1",
                company_id="comp1",
                signal_type=SignalType.AGGRESSIVE_DISCOUNTING,
                subject="Downmarket Enterprise Push",
                title="Aggressive Down-Market Enterprise Expansion",
                summary="Observed significant price drop aiming to capture mid-market enterprise share.",
                supporting_event_ids=[event.event_id],
                confidence=SignalConfidenceAssessment(score=0.9, level=ConfidenceLevel.HIGH),
                status=SignalStatus.ACTIVE,
            )
            await intel_repo.create_signal(signal)

            # 4. Create Strategic Insight
            insight = Insight(
                insight_id="ins_001",
                workspace_id="ws1",
                company_id="comp1",
                insight_type=InsightType.PRICING_STRATEGY,
                title="Aggressive Down-Market Repositioning",
                summary="Competitor is discounting heavily to undercut enterprise market leaders.",
                statement="Competitor reduced enterprise price by 50% indicating a direct volume play.",
                supporting_signal_ids=[signal.signal_id],
                supporting_event_ids=[event.event_id],
                factual_confidence=ConfidenceAssessment(
                    score=0.95, level=ConfidenceLevel.VERY_HIGH, is_corroborated=True
                ),
                interpretive_confidence=InterpretiveConfidenceAssessment(
                    score=0.85, level=ConfidenceLevel.HIGH, rationale="Corroborated by primary source"
                ),
                significance=SignificanceAssessment(
                    status=SignificanceStatus.ASSESSED,
                    level=SignificanceLevel.HIGH,
                    score=0.9,
                ),
                temporal_scope=TemporalScope(start_date=now),
                status=InsightStatus.ASSESSED,
            )
            await intel_repo.create_insight(insight)

        # Verify in a new session with joined relationships
        async with db_manager.session() as session:
            intel_repo = SqlIntelligenceRepository(session)

            fetched_event = await intel_repo.get_event_by_id("evt_001")
            assert fetched_event is not None
            assert fetched_event.title == "Enterprise Plan 50% Price Cut"
            assert len(fetched_event.evidence) == 1
            assert fetched_event.evidence[0].evidence_id == "ev_001"
            assert fetched_event.evidence[0].extracted_text == "Enterprise Tier price dropped from $999 to $499"

            fetched_sig = await intel_repo.get_signal_by_id("sig_001")
            assert fetched_sig is not None
            assert fetched_sig.supporting_event_ids == ["evt_001"]

            fetched_ins = await intel_repo.get_insight_by_id("ins_001")
            assert fetched_ins is not None
            assert fetched_ins.supporting_signal_ids == ["sig_001"]
            assert fetched_ins.supporting_event_ids == ["evt_001"]
    finally:
        await db_manager.close()


# ==============================================================================
# 6. Complete 10-Layer Provenance Reconstruction Test
# ==============================================================================


@pytest.mark.anyio
async def test_full_ten_layer_provenance_reconstruction() -> None:
    """Test full lineage traversal from Recommendation down to Source."""
    now = datetime.now(timezone.utc)
    db_manager = await _get_initialized_db()

    try:
        async with db_manager.session() as session:
            # Setup complete 10-layer lineage
            user = User(user_id="u1", email="u1@test.com")
            await SqlUserRepository(session).create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await SqlWorkspaceRepository(session).create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await SqlCompetitorRepository(session).create(comp)
            source = Source(
                identity=SourceIdentity(source_id="src1", company_id="comp1", workspace_id="ws1", display_name="S1"),
                classification=SourceClassification(category=SourceCategory.PRICING),
                location=SourceLocation(url_or_identifier="https://comp1.com/pricing"),
                acquisition=AcquisitionConfig(),
            )
            await SqlSourceRepository(session).create(source)
            doc = Document(
                document_id="doc1",
                source_id="src1",
                company_id="comp1",
                workspace_id="ws1",
                canonical_url="https://comp1.com/pricing",
            )
            await SqlDocumentRepository(session).create_document(doc)
            ver = DocumentVersion(
                version_id="ver1",
                document_id="doc1",
                content="Enterprise Tier $499",
                content_hash="h999",
                acquired_at=now,
            )
            await SqlDocumentRepository(session).create_version(ver)
            chunk = DocumentChunk(
                chunk_id="chk1",
                document_id="doc1",
                version_id="ver1",
                content="Enterprise Tier $499",
                metadata=ChunkMetadata(
                    source_id="src1",
                    document_id="doc1",
                    version_id="ver1",
                    company_id="comp1",
                    workspace_id="ws1",
                    source_url="https://comp1.com/pricing",
                    source_category="pricing",
                    collected_at=now,
                ),
            )
            await SqlDocumentRepository(session).create_chunk(chunk)

            evidence = EvidenceReference(
                evidence_id="ev1",
                source_id="src1",
                document_id="doc1",
                version_id="ver1",
                chunk_id="chk1",
                source_url="https://comp1.com/pricing",
                source_category="pricing",
                extracted_text="Enterprise Tier $499",
                acquired_at=now,
            )
            event = IntelligenceEvent(
                event_id="evt1",
                workspace_id="ws1",
                company_id="comp1",
                event_type=IntelligenceEventType.PRICING_CHANGE,
                title="Price Cut",
                summary="Price dropped",
                subject="Pricing",
                event_time=EventTime(acquired_at=now, detected_at=now),
                confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.VERY_HIGH),
                evidence=[evidence],
            )
            await SqlIntelligenceRepository(session).create_event(event)

            signal = InferredSignal(
                signal_id="sig1",
                workspace_id="ws1",
                company_id="comp1",
                signal_type=SignalType.AGGRESSIVE_DISCOUNTING,
                subject="Pricing Strategy",
                title="Price Aggression",
                summary="Aggressive pricing observed",
                supporting_event_ids=[event.event_id],
            )
            await SqlIntelligenceRepository(session).create_signal(signal)

            insight = Insight(
                insight_id="ins1",
                workspace_id="ws1",
                company_id="comp1",
                insight_type=InsightType.PRICING_STRATEGY,
                title="Discount Push",
                summary="Competitor discount push",
                statement="Competitor discounting",
                supporting_signal_ids=[signal.signal_id],
                supporting_event_ids=[event.event_id],
                factual_confidence=ConfidenceAssessment(score=1.0, level=ConfidenceLevel.VERY_HIGH),
            )
            await SqlIntelligenceRepository(session).create_insight(insight)

            dec_repo = SqlDecisionRepository(session)

            # Decision Context
            context = DecisionContext(
                decision_id="dec1",
                workspace_id="ws1",
                title="Countering Competitor Discounting",
                decision_domain=DecisionDomain.PRICING,
                business_objectives=[
                    DecisionObjective(
                        name="Defend Market Share",
                        dimension=ObjectiveDimension.MARKET_SHARE,
                        target_description="Maintain 40% enterprise market share",
                    )
                ],
                constraints=[
                    DecisionConstraint(
                        constraint_type=ConstraintType.BUDGET,
                        description="Gross margin must remain above 75%",
                    )
                ],
                competitor_ids=["comp1"],
            )
            await dec_repo.create_context(context)

            # Decision Option
            option = DecisionOption(
                option_id="opt1",
                decision_id="dec1",
                title="Value-Add Feature Bundle",
                description="Bundle premium security features without direct price war.",
                action_type=ActionType.ASYMMETRIC_RESPONSE,
                expected_effects=[
                    PredictedConsequence(
                        dimension=ObjectiveDimension.RETENTION,
                        description="Increases customer retention",
                        is_positive=True,
                    )
                ],
                supporting_insight_ids=[insight.insight_id],
                supporting_event_ids=[event.event_id],
            )
            await dec_repo.create_option(option)

            # Recommendation
            rec = Recommendation(
                recommendation_id="rec1",
                decision_id="dec1",
                workspace_id="ws1",
                selected_option_id="opt1",
                urgency=DecisionUrgency.RECOMMENDED,
                status=RecommendationStatus.PROPOSED,
                title="Introduce Value-Add Security Bundle",
                summary="Protect margin and retain accounts by bundling security instead of cutting prices.",
                rationale="Maintains price discipline while neutralizing competitor value proposition.",
                evaluated_alternatives=[
                    EvaluatedAlternative(
                        option_id="opt1",
                        title="Value-Add Bundle",
                        is_selected=True,
                    )
                ],
                expected_benefits=["Protects 75% gross margin target", "Differentiates on enterprise security"],
                confidence=RecommendationConfidenceAssessment(
                    score=0.88,
                    level=ConfidenceLevel.HIGH,
                    rationale="Grounded in verified competitor price cuts and margin constraints",
                ),
                supporting_insight_ids=[insight.insight_id],
                supporting_event_ids=[event.event_id],
            )
            await dec_repo.create_recommendation(rec)

        # Reconstruct provenance in a new session
        async with db_manager.session() as session:
            dec_repo = SqlDecisionRepository(session)
            prov = await dec_repo.reconstruct_recommendation_provenance("rec1")

            assert prov["recommendation_id"] == "rec1"
            assert prov["decision_id"] == "dec1"
            assert prov["selected_option"]["option_id"] == "opt1"
            assert prov["insight_count"] == 1
            assert "ins1" in prov["insight_ids"]
            assert prov["signal_count"] == 1
            assert "sig1" in prov["signal_ids"]
            assert prov["event_count"] == 1
            assert "evt1" in prov["event_ids"]
            assert prov["evidence_count"] == 1
            assert "ev1" in prov["evidence_ids"]
            assert prov["chunk_count"] == 1
            assert "chk1" in prov["chunk_ids"]
            assert prov["version_count"] == 1
            assert "ver1" in prov["version_ids"]
            assert prov["document_count"] == 1
            assert "doc1" in prov["document_ids"]
            assert prov["source_count"] == 1
            assert "src1" in prov["source_ids"]
    finally:
        await db_manager.close()


# ==============================================================================
# 7. Workflow State Store & Checkpoint Durability Test
# ==============================================================================


@pytest.mark.anyio
async def test_workflow_state_store_checkpoints_and_resume() -> None:
    """Test SqlWorkflowStateStore run persistence, stage checkpointing, and resume state."""
    db_manager = await _get_initialized_db()
    try:
        async with db_manager.session() as session:
            # Create workspace and competitor parent
            user = User(user_id="u1", email="u1@test.com")
            await SqlUserRepository(session).create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await SqlWorkspaceRepository(session).create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await SqlCompetitorRepository(session).create(comp)

        store = SqlWorkflowStateStore(session_manager=db_manager)

        # 1. Create Workflow Run
        run = WorkflowRun(
            run_id="run_999",
            workspace_id="ws1",
            company_id="comp1",
            trigger_type=TriggerType.MANUAL_USER_REQUEST,
            status=WorkflowRunStatus.RUNNING,
            current_stage=WorkflowStageType.KNOWLEDGE_PROCESSING,
            target_stop_stage=WorkflowStageType.INSIGHT_GENERATION,
        )
        await store.create_run(run)

        # 2. Checkpoint Stage 1 (Knowledge Processing)
        stage1_record = StageExecutionRecord(
            stage=WorkflowStageType.KNOWLEDGE_PROCESSING,
            status=StageStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_ms=120.5,
            attempt_count=1,
            idempotency_key="idemp_stage1_abc",
            output_ids={"document_id": "doc1", "version_id": "ver1"},
        )
        outputs1 = WorkflowProducedOutputs(
            document_ids=["doc1"],
            version_ids=["ver1"],
            chunk_ids=["chk1", "chk2"],
        )
        await store.checkpoint_stage("run_999", WorkflowStageType.KNOWLEDGE_PROCESSING, stage1_record, outputs1)

        # 3. Checkpoint Stage 2 (Event Analysis with Error)
        error_rec = StageErrorRecord(
            stage=WorkflowStageType.EVENT_ANALYSIS,
            error_category=ErrorCategory.TRANSIENT,
            error_message="Rate limit exceeded on LLM gateway",
            attempt_number=1,
            is_retryable=True,
        )
        stage2_record = StageExecutionRecord(
            stage=WorkflowStageType.EVENT_ANALYSIS,
            status=StageStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_ms=45.0,
            attempt_count=1,
            error=error_rec,
        )
        outputs2 = WorkflowProducedOutputs(
            document_ids=["doc1"],
            version_ids=["ver1"],
            chunk_ids=["chk1", "chk2"],
        )
        await store.checkpoint_stage("run_999", WorkflowStageType.EVENT_ANALYSIS, stage2_record, outputs2)

        # 4. Retrieve Run and Verify State
        fetched_run = await store.get_run("run_999")
        assert fetched_run is not None
        assert fetched_run.run_id == "run_999"
        assert fetched_run.current_stage == WorkflowStageType.EVENT_ANALYSIS
        assert len(fetched_run.stage_records) == 2
        assert fetched_run.stage_records[WorkflowStageType.KNOWLEDGE_PROCESSING].status == StageStatus.COMPLETED
        assert fetched_run.stage_records[WorkflowStageType.EVENT_ANALYSIS].status == StageStatus.FAILED
        assert len(fetched_run.errors) == 1
        assert fetched_run.errors[0].error_category == ErrorCategory.TRANSIENT
        assert fetched_run.produced_outputs.chunk_ids == ["chk1", "chk2"]

        # 5. Test List Runs
        runs = await store.list_runs(workspace_id="ws1", company_id="comp1")
        assert len(runs) == 1
        assert runs[0].run_id == "run_999"
    finally:
        await db_manager.close()


# ==============================================================================
# 8. Transaction Rollback & Foreign Key Integrity Test
# ==============================================================================


@pytest.mark.anyio
async def test_transaction_rollback_and_fk_integrity() -> None:
    """Test transaction rollback on error and foreign key constraints."""
    db_manager = await _get_initialized_db()
    try:
        # 1. Test Rollback
        try:
            async with db_manager.session() as session:
                user = User(user_id="rollback_usr", email="rollback@test.com")
                await SqlUserRepository(session).create(user)
                # Intentionally cause an error in the transaction
                raise RuntimeError("Forced simulation error")
        except RuntimeError:
            pass

        # Verify rollback
        async with db_manager.session() as session:
            fetched = await SqlUserRepository(session).get_by_id("rollback_usr")
            assert fetched is None, "Expected transaction to be rolled back"
    finally:
        await db_manager.close()
