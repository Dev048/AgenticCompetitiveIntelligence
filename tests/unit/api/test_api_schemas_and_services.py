"""Unit tests for Application use-case services, schemas, and error handlers."""

from datetime import datetime, timezone
from typing import Any
import pytest

from competitive_intelligence.application.models import (
    TrackedCompetitor,
    User,
    Workspace,
)
from competitive_intelligence.application.services import (
    CompetitorService,
    DecisionService,
    DocumentService,
    IntelligenceService,
    SourceService,
    WorkflowExecutionService,
    WorkspaceService,
)
from competitive_intelligence.core.errors import (
    DuplicateEntityError,
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    DecisionOption,
    DecisionRisk,
    Recommendation,
)
from competitive_intelligence.infrastructure.database.repositories import (
    SqlCompetitorRepository,
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlSourceRepository,
    SqlUserRepository,
    SqlWorkflowStateStore,
    SqlWorkspaceRepository,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
)
from competitive_intelligence.knowledge_pipeline.models import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowInputs,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.source_acquisition.models import (
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_workspace_and_competitor_services() -> None:
    """Test WorkspaceService and CompetitorService behavior and boundary validation."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()

    try:
        async with manager.session() as session:
            ws_service = WorkspaceService(
                workspace_repo=SqlWorkspaceRepository(session),
                user_repo=SqlUserRepository(session),
            )
            comp_service = CompetitorService(
                competitor_repo=SqlCompetitorRepository(session),
                workspace_repo=SqlWorkspaceRepository(session),
            )

            # 1. Workspace creation validation
            with pytest.raises(ValidationError):
                await ws_service.create_workspace(name="", owner_user_id="u1")

            ws1 = await ws_service.create_workspace(name="Workspace Alpha", owner_user_id="u1")
            assert ws1.workspace_id is not None
            assert ws1.name == "Workspace Alpha"

            # 2. Competitor registration & boundary
            with pytest.raises(ValidationError):
                await comp_service.register_competitor(workspace_id=ws1.workspace_id, name="", domain="")

            comp1 = await comp_service.register_competitor(
                workspace_id=ws1.workspace_id,
                name="Competitor One",
                domain="comp1.com",
                description="Primary rival",
            )
            assert comp1.competitor_id is not None

            # 3. Workspace isolation check
            with pytest.raises(WorkspaceIsolationError):
                await comp_service.get_competitor(workspace_id="other_ws", competitor_id=comp1.competitor_id)

            # 4. Competitor update
            updated = await comp_service.update_competitor(
                workspace_id=ws1.workspace_id,
                competitor_id=comp1.competitor_id,
                name="Competitor One Updated",
            )
            assert updated.name == "Competitor One Updated"
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_source_and_document_services() -> None:
    """Test SourceService and DocumentService query and isolation rules."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()

    try:
        async with manager.session() as session:
            user_repo = SqlUserRepository(session)
            ws_repo = SqlWorkspaceRepository(session)
            comp_repo = SqlCompetitorRepository(session)
            src_repo = SqlSourceRepository(session)
            doc_repo = SqlDocumentRepository(session)

            user = User(user_id="u1", email="u1@test.com")
            await user_repo.create(user)
            ws = Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1")
            await ws_repo.create(ws)
            comp = TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            await comp_repo.create(comp)

            src_service = SourceService(source_repo=src_repo, competitor_repo=comp_repo, workspace_repo=ws_repo)
            doc_service = DocumentService(document_repo=doc_repo, workspace_repo=ws_repo)

            # Register source
            src = await src_service.register_source(
                workspace_id="ws1",
                company_id="comp1",
                display_name="Comp1 Pricing",
                url_or_identifier="https://comp1.com/pricing",
                category=SourceCategory.PRICING,
            )
            assert src.identity.source_id is not None

            # Document lineage
            doc = Document(
                document_id="doc1",
                workspace_id="ws1",
                company_id="comp1",
                source_id=src.identity.source_id,
                canonical_url="https://comp1.com/pricing",
            )
            await doc_repo.create_document(doc)

            ver = DocumentVersion(
                version_id="ver1",
                document_id="doc1",
                content="Sample content",
                content_hash="hash123",
            )
            await doc_repo.create_version(ver)

            # Verify DocumentService operations
            fetched_doc = await doc_service.get_document(workspace_id="ws1", document_id="doc1")
            assert fetched_doc.canonical_url == "https://comp1.com/pricing"

            versions = await doc_service.list_document_versions(workspace_id="ws1", document_id="doc1")
            assert len(versions) == 1

            # Verify workspace isolation on document
            with pytest.raises(WorkspaceIsolationError):
                await doc_service.get_document(workspace_id="ws_other", document_id="doc1")
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_workflow_execution_service_dispatch() -> None:
    """Test WorkflowExecutionService asynchronous dispatch and state persistence."""
    manager = DatabaseSessionManager(database_url="sqlite+aiosqlite:///:memory:")
    await manager.create_tables()

    try:
        async with manager.session() as session:
            await SqlUserRepository(session).create(User(user_id="u1", email="u1@test.com"))
            await SqlWorkspaceRepository(session).create(Workspace(workspace_id="ws1", name="WS1", owner_user_id="u1"))
            await SqlCompetitorRepository(session).create(
                TrackedCompetitor(competitor_id="comp1", workspace_id="ws1", name="Comp1", domain="comp1.com")
            )

        dispatched_tasks: list[Any] = []

        def mock_dispatcher(coro: Any) -> None:
            dispatched_tasks.append(coro)

        state_store = SqlWorkflowStateStore(manager.sessionmaker)
        wf_service = WorkflowExecutionService(
            state_store=state_store,
            task_dispatcher=mock_dispatcher,
        )

        inputs = WorkflowInputs(
            workspace_id="ws1",
            company_id="comp1",
            query_text="Evaluate competitive pricing movements",
        )

        run = await wf_service.submit_workflow_run(
            workspace_id="ws1",
            inputs=inputs,
            target_stop_stage=WorkflowStageType.INSIGHT_GENERATION,
        )

        assert run.run_id is not None
        assert run.status == WorkflowRunStatus.PENDING
        assert len(dispatched_tasks) == 1, "Expected background execution task to be dispatched"

        # Verify durable run can be retrieved
        fetched = await wf_service.get_workflow_run(workspace_id="ws1", run_id=run.run_id)
        assert fetched.run_id == run.run_id
        assert fetched.company_id == "comp1"
    finally:
        await manager.close()
