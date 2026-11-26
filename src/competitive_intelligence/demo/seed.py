"""Idempotent seeding of the public demo workspace.

Runs the real pipeline (normalize, version, chunk, embed, index, retrieve, events,
signals, insights, decision support) over the fictional corpus in ``corpus.py``:
- No crawling: every document is supplied as pre-acquired text.
- No live LLM calls by default: a ReplayLLMClient serves responses recorded once against
  the real provider (``scripts/seed_demo.py --record``). Fixture misses fall back to each
  stage's deterministic path and are reported in the summary.
- Idempotent: fixed IDs for workspace, competitors and sources, and a fixed ``run_id`` per
  pipeline step. Completed steps are skipped; failed or missing steps are (re)run.
"""

import time
from pathlib import Path

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.application.models import TrackedCompetitor, User, Workspace
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.demo.corpus import (
    COMPETITORS,
    DEMO_OWNER_USER_ID,
    DEMO_WORKSPACE_ID,
    DEMO_WORKSPACE_NAME,
    DOCUMENTS,
    DemoDocument,
    DemoPipelineStep,
    document_by_source_id,
    pipeline_steps,
)
from competitive_intelligence.demo.replay_llm import (
    LLMFixtureStore,
    ReplayLLMClient,
    ReplayStats,
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
)
from competitive_intelligence.orchestration.interfaces import BaseWorkflowOrchestrator
from competitive_intelligence.orchestration.models import (
    TriggerType,
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
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)

logger = get_logger(__name__)

DEFAULT_FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "llm_replay.json"


class StepOutcome(BaseModel):
    run_id: str
    status: str
    skipped: bool = False
    error: str | None = None


class SeedSummary(BaseModel):
    workspace_id: str
    steps_run: int = 0
    steps_skipped: int = 0
    steps_failed: int = 0
    duration_seconds: float = 0.0
    llm: ReplayStats = Field(default_factory=ReplayStats)
    steps: list[StepOutcome] = Field(default_factory=list)


def load_replay_client(fixtures_path: Path | None = None) -> ReplayLLMClient:
    """Replay-only client over the committed fixtures."""
    return ReplayLLMClient(LLMFixtureStore.load(fixtures_path or DEFAULT_FIXTURES_PATH))


async def seed_demo_workspace(
    llm_client: BaseLLMClient | None = None,
    db_manager: DatabaseSessionManager | None = None,
) -> SeedSummary:
    """Seed (or complete) the demo workspace. Safe to call on every boot."""
    started = time.perf_counter()
    manager = db_manager or get_db_manager()
    llm = llm_client or load_replay_client()
    await manager.create_tables()

    await _ensure_reference_data(manager)

    orchestrator = _build_orchestrator(llm, manager)
    summary = SeedSummary(workspace_id=DEMO_WORKSPACE_ID)
    state_store = SqlWorkflowStateStore(manager.sessionmaker)

    for step in pipeline_steps():
        existing = await state_store.get_run(step.run_id)
        if existing is not None and existing.status == WorkflowRunStatus.COMPLETED:
            summary.steps_skipped += 1
            summary.steps.append(StepOutcome(run_id=step.run_id, status="completed", skipped=True))
            continue

        if isinstance(llm, ReplayLLMClient):
            llm.begin_step(step.run_id)
        outcome = await _run_step(orchestrator, step)
        summary.steps.append(outcome)
        if outcome.status == WorkflowRunStatus.COMPLETED.value:
            summary.steps_run += 1
        else:
            summary.steps_failed += 1
            logger.warning("Demo seed step did not complete", **outcome.model_dump())

    if isinstance(llm, ReplayLLMClient):
        summary.llm = llm.stats
    summary.duration_seconds = round(time.perf_counter() - started, 2)
    return summary


async def _ensure_reference_data(manager: DatabaseSessionManager) -> None:
    async with manager.session() as session:
        users = SqlUserRepository(session)
        if await users.get_by_id(DEMO_OWNER_USER_ID) is None:
            await users.create(User(user_id=DEMO_OWNER_USER_ID, email="demo-operator@demo.example"))

        workspaces = SqlWorkspaceRepository(session)
        if await workspaces.get_by_id(DEMO_WORKSPACE_ID) is None:
            await workspaces.create(
                Workspace(
                    workspace_id=DEMO_WORKSPACE_ID,
                    name=DEMO_WORKSPACE_NAME,
                    owner_user_id=DEMO_OWNER_USER_ID,
                )
            )

        competitors = SqlCompetitorRepository(session)
        for comp in COMPETITORS:
            if await competitors.get_by_id(comp.competitor_id) is None:
                await competitors.create(
                    TrackedCompetitor(
                        competitor_id=comp.competitor_id,
                        workspace_id=DEMO_WORKSPACE_ID,
                        name=comp.name,
                        domain=comp.domain,
                        description=comp.description,
                    )
                )

        sources = SqlSourceRepository(session)
        for doc in DOCUMENTS:
            if await sources.get_by_id(doc.source_id) is None:
                await sources.create(_source_for(doc))


def _source_for(doc: DemoDocument) -> Source:
    return Source(
        identity=SourceIdentity(
            source_id=doc.source_id,
            workspace_id=DEMO_WORKSPACE_ID,
            company_id=doc.competitor_id,
            display_name=doc.display_name,
        ),
        classification=SourceClassification(
            category=doc.category,
            platform=PlatformType.WEB,
            content_type=ContentType.TEXT,
        ),
        location=SourceLocation(
            url_or_identifier=doc.url,
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
        # Demo sources are never polled: content is pre-acquired and the scheduler is off.
        monitoring=MonitoringPolicy(enabled=False),
    )


def _build_orchestrator(
    llm: BaseLLMClient, manager: DatabaseSessionManager
) -> BaseWorkflowOrchestrator:
    """Production stage wiring (same providers as the API), with the given LLM client."""
    document_repo = SqlDocumentRepository(manager.sessionmaker)
    intelligence_repo = SqlIntelligenceRepository(manager.sessionmaker)
    decision_repo = SqlDecisionRepository(manager.sessionmaker)

    ingestion = deps.get_knowledge_ingestion_service(
        document_repo=document_repo,
        chunking_service=deps.get_chunking_service(),
        embedding_service=deps.get_embedding_service(),
        vector_store=deps.get_vector_store(),
    )
    adaptive = deps.get_adaptive_retriever()
    retriever = deps.get_production_retriever(
        adaptive_retriever=adaptive,
        agentic_retriever=deps.get_agentic_retriever(adaptive_retriever=adaptive, llm_client=llm),
    )
    return deps.get_workflow_orchestrator(
        state_store=SqlWorkflowStateStore(manager.sessionmaker),
        knowledge_stage=deps.get_knowledge_processing_stage(ingestion_service=ingestion),
        retrieval_stage=deps.get_retrieval_stage(retriever=retriever),
        event_stage=deps.get_event_analysis_stage(intelligence_repo=intelligence_repo, llm_client=llm),
        signal_stage=deps.get_signal_detection_stage(intelligence_repo=intelligence_repo),
        insight_stage=deps.get_insight_generation_stage(
            intelligence_repo=intelligence_repo, llm_client=llm
        ),
        decision_stage=deps.get_decision_support_stage(decision_repo=decision_repo, llm_client=llm),
    )


async def _run_step(orchestrator: BaseWorkflowOrchestrator, step: DemoPipelineStep) -> StepOutcome:
    doc = document_by_source_id(step.source_id)
    inputs = WorkflowInputs(
        workspace_id=DEMO_WORKSPACE_ID,
        company_id=doc.competitor_id,
        raw_document_content=doc.versions[step.version_index],
        source_url=doc.url,
        source_id=doc.source_id,
        query_text=step.query_text,
        decision_context=step.decision_context,
        enable_decision_support=step.decision_context is not None,
        metadata={
            "source_id": doc.source_id,
            "source_category": doc.category.value,
            "trigger": "demo_seed",
            "trigger_type": TriggerType.MANUAL_USER_REQUEST.value,
        },
    )
    try:
        result = await orchestrator.run_workflow(
            inputs=inputs,
            target_stop_stage=WorkflowStageType(step.target_stop_stage),
            run_id=step.run_id,
        )
    except Exception as err:  # keep seeding the remaining steps
        return StepOutcome(run_id=step.run_id, status="failed", error=f"{type(err).__name__}: {err}")
    return StepOutcome(run_id=step.run_id, status=result.status.value, error=result.error_summary)
