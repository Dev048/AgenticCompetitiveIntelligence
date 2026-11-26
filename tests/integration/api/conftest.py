"""Shared fixtures for API integration tests: a tiny workspace seeded through the real pipeline.

One fictional competitor and one pricing page with two versions are run through the same
stage wiring the API uses (mock LLM, mock embeddings): the first version is only indexed,
the second runs to decision support, so change detection, insights, recommendations and
the full provenance chain all exist.
"""

import asyncio
from collections.abc import Iterator

import pytest

from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.application.models import TrackedCompetitor, User, Workspace
from competitive_intelligence.core.config import get_settings
from competitive_intelligence.decision_support.models import (
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    RiskTolerance,
    TimeHorizon,
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
    get_db_manager,
    reset_db_managers,
)
from competitive_intelligence.orchestration.models import (
    WorkflowInputs,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    MonitoringPolicy,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)

TEST_WORKSPACE_ID = "test-workspace"
_OWNER = "test-operator"
_COMPANY = "comp-acme"
_SOURCE = "src-acme-pricing"
_URL = "https://acme-analytics.example/pricing"
_VERSIONS = [
    "Acme Analytics Pricing.\n"
    "Starter plan: $49 per month. Includes 2 seats and 50,000 tracked events.\n"
    "Growth plan: $199 per month. Includes 10 seats and funnels.\n"
    "There is no free plan; a 14-day trial is available.",
    "Acme Analytics Pricing.\n"
    "New Free plan: $0 forever. Includes up to 3 seats and 10,000 tracked events.\n"
    "Starter plan: now $29 per month (previously $49). Includes 5 seats.\n"
    "Growth plan: $199 per month. Includes 10 seats and funnels.",
]
QUERY = "Acme Analytics pricing change"


def _decision_context() -> DecisionContext:
    return DecisionContext(
        decision_id="dc-acme-pricing",
        workspace_id=TEST_WORKSPACE_ID,
        organization_id="test-org",
        title="Respond to Acme Analytics' entry-tier price cut and new Free plan",
        decision_domain=DecisionDomain.PRICING,
        business_objectives=[
            DecisionObjective(
                objective_id="obj-acquisition",
                name="Protect self-serve acquisition",
                dimension=ObjectiveDimension.GROWTH,
                target_description="Keep monthly signups within 10% of last quarter",
                priority_weight=1.0,
            )
        ],
        risk_tolerance=RiskTolerance.MODERATE,
        time_horizon=TimeHorizon.IMMEDIATE_SHORT_TERM,
        competitor_ids=[_COMPANY],
    )


async def _seed(manager: DatabaseSessionManager) -> None:
    await manager.create_tables()
    async with manager.session() as session:
        await SqlUserRepository(session).create(User(user_id=_OWNER, email="op@test.example"))
        await SqlWorkspaceRepository(session).create(
            Workspace(workspace_id=TEST_WORKSPACE_ID, name="Test", owner_user_id=_OWNER)
        )
        await SqlCompetitorRepository(session).create(
            TrackedCompetitor(
                competitor_id=_COMPANY,
                workspace_id=TEST_WORKSPACE_ID,
                name="Acme Analytics",
                domain="acme-analytics.example",
            )
        )
        await SqlSourceRepository(session).create(
            Source(
                identity=SourceIdentity(
                    source_id=_SOURCE,
                    workspace_id=TEST_WORKSPACE_ID,
                    company_id=_COMPANY,
                    display_name="Acme pricing page",
                ),
                classification=SourceClassification(
                    category=SourceCategory.PRICING,
                    platform=PlatformType.WEB,
                    content_type=ContentType.TEXT,
                ),
                location=SourceLocation(
                    url_or_identifier=_URL, access_pattern=AccessPattern.PUBLIC_WEB
                ),
                acquisition=AcquisitionConfig(
                    acquisition_method=AcquisitionMethod.CRAWL4AI,
                    tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
                ),
                monitoring=MonitoringPolicy(enabled=False),
            )
        )

    llm = MockLLMClient()
    intelligence_repo = SqlIntelligenceRepository(manager.sessionmaker)
    adaptive = deps.get_adaptive_retriever()
    orchestrator = deps.get_workflow_orchestrator(
        state_store=SqlWorkflowStateStore(manager.sessionmaker),
        knowledge_stage=deps.get_knowledge_processing_stage(
            ingestion_service=deps.get_knowledge_ingestion_service(
                document_repo=SqlDocumentRepository(manager.sessionmaker),
                chunking_service=deps.get_chunking_service(),
                embedding_service=deps.get_embedding_service(),
                vector_store=deps.get_vector_store(),
            )
        ),
        retrieval_stage=deps.get_retrieval_stage(
            retriever=deps.get_production_retriever(
                adaptive_retriever=adaptive,
                agentic_retriever=deps.get_agentic_retriever(
                    adaptive_retriever=adaptive, llm_client=llm
                ),
            )
        ),
        event_stage=deps.get_event_analysis_stage(intelligence_repo=intelligence_repo, llm_client=llm),
        signal_stage=deps.get_signal_detection_stage(intelligence_repo=intelligence_repo),
        insight_stage=deps.get_insight_generation_stage(
            intelligence_repo=intelligence_repo, llm_client=llm
        ),
        decision_stage=deps.get_decision_support_stage(
            decision_repo=SqlDecisionRepository(manager.sessionmaker), llm_client=llm
        ),
    )
    steps = [
        (0, WorkflowStageType.KNOWLEDGE_PROCESSING, None, None),
        (1, WorkflowStageType.DECISION_SUPPORT, QUERY, _decision_context()),
    ]
    for index, stop, query, context in steps:
        result = await orchestrator.run_workflow(
            inputs=WorkflowInputs(
                workspace_id=TEST_WORKSPACE_ID,
                company_id=_COMPANY,
                raw_document_content=_VERSIONS[index],
                source_url=_URL,
                source_id=_SOURCE,
                query_text=query,
                decision_context=context,
                enable_decision_support=context is not None,
                metadata={"source_id": _SOURCE, "source_category": SourceCategory.PRICING.value},
            ),
            target_stop_stage=stop,
            run_id=f"test-seed-{index}",
        )
        assert result.status == WorkflowRunStatus.COMPLETED, result


@pytest.fixture(scope="session")
def seeded_workspace(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, str]]:
    """Seed once per session; yields the env the API must run with to see that data."""
    root = tmp_path_factory.mktemp("seeded")
    env = {
        "ENVIRONMENT": "development",
        "LLM_PROVIDER": "mock",
        "EMBEDDING_PROVIDER": "mock",
        "QDRANT_PATH": str(root / "qdrant"),
        "DATABASE_URL": f"sqlite+aiosqlite:///{root / 'seed.db'}",
    }
    patch = pytest.MonkeyPatch()
    for key, value in env.items():
        patch.setenv(key, value)
    get_settings.cache_clear()
    deps.reset_resource_caches()

    async def _run() -> None:
        await _seed(get_db_manager())
        await reset_db_managers()

    try:
        asyncio.run(_run())
    finally:
        # Release the embedded Qdrant folder and restore the environment so the seeding env
        # never leaks into unrelated tests; users of this fixture apply ``env`` themselves.
        deps.reset_resource_caches()
        patch.undo()
        get_settings.cache_clear()
    yield env
