"""FastAPI Dependency Injection providers."""

from collections.abc import AsyncGenerator
from fastapi import Depends, Path
from fastapi.params import Depends as DependsClass
from sqlalchemy.ext.asyncio import AsyncSession

from competitive_intelligence.agentic_rag.llm import (
    BaseLLMClient,
    create_llm_client,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.agentic_rag.selective.controller import (
    SelectiveAgenticRetriever,
)
from competitive_intelligence.core.config import Settings, get_settings
from competitive_intelligence.decision_support import (
    DeterministicDecisionAnalyzer,
    LLMOptionGenerator,
)
from competitive_intelligence.application.interfaces import (
    BaseCompetitorRepository,
    BaseDecisionRepository,
    BaseDocumentRepository,
    BaseIntelligenceRepository,
    BaseSourceRepository,
    BaseUserRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services import (
    CompetitorService,
    DecisionService,
    DocumentService,
    IntelligenceService,
    SourceMonitoringService,
    SourceService,
    WorkflowExecutionService,
    WorkspaceService,
)
from competitive_intelligence.infrastructure.database.repositories import (
    SqlCompetitorRepository,
    SqlDecisionRepository,
    SqlDocumentRepository,
    SqlIntelligenceRepository,
    SqlMonitoringLeaseRepository,
    SqlSourceRepository,
    SqlUserRepository,
    SqlWorkflowStateStore,
    SqlWorkspaceRepository,
)
from competitive_intelligence.intelligence import (
    DeterministicStrategicAnalyzer,
    LLMEvidenceInterpreter,
    LLMStrategicInterpreter,
    RetrievedEvidenceAnalyzer,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.knowledge_pipeline.chunking_service import (
    ChunkingService,
)
from competitive_intelligence.knowledge_pipeline.embeddings.embedding_service import (
    EmbeddingService,
)
from competitive_intelligence.knowledge_pipeline.embeddings.mock_provider import (
    MockEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.ingestion_service import (
    KnowledgeIngestionService,
)
from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseVectorStore,
)
from competitive_intelligence.knowledge_pipeline.models import (
    DistanceMetric,
    VectorStoreConfig,
)
from competitive_intelligence.knowledge_pipeline.vector_store.qdrant_store import (
    QdrantVectorStore,
)
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
)
from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowOrchestrator,
    BaseWorkflowStateStore,
)
from competitive_intelligence.orchestration.models import WorkflowStageType
from competitive_intelligence.orchestration.orchestrator import (
    DeterministicWorkflowOrchestrator,
)
from competitive_intelligence.orchestration.stages import (
    DecisionSupportStage,
    EventAnalysisStage,
    InsightGenerationStage,
    KnowledgeProcessingStage,
    RetrievalStage,
    SignalDetectionStage,
)
from competitive_intelligence.retrieval_engine.dense.semantic_retriever import (
    SemanticRetriever,
    SemanticRetrieverAdapter,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.routing.adaptive_retriever import (
    AdaptiveRetriever,
)
from competitive_intelligence.retrieval_engine.routing.router import (
    DeterministicRetrievalRouter,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a scoped async database session for the request."""
    manager = get_db_manager()
    async with manager.session() as session:
        yield session


def get_user_repo(session: AsyncSession = Depends(get_session)) -> BaseUserRepository:
    return SqlUserRepository(session)


def get_workspace_repo(session: AsyncSession = Depends(get_session)) -> BaseWorkspaceRepository:
    return SqlWorkspaceRepository(session)


def get_competitor_repo(session: AsyncSession = Depends(get_session)) -> BaseCompetitorRepository:
    return SqlCompetitorRepository(session)


def get_source_repo(session: AsyncSession = Depends(get_session)) -> BaseSourceRepository:
    return SqlSourceRepository(session)


def get_document_repo(session: AsyncSession = Depends(get_session)) -> BaseDocumentRepository:
    return SqlDocumentRepository(session)


def get_intelligence_repo(session: AsyncSession = Depends(get_session)) -> BaseIntelligenceRepository:
    return SqlIntelligenceRepository(session)


def get_decision_repo(session: AsyncSession = Depends(get_session)) -> BaseDecisionRepository:
    return SqlDecisionRepository(session)


def get_monitoring_lease_repo(session: AsyncSession = Depends(get_session)) -> BaseMonitoringLeaseRepository:
    return SqlMonitoringLeaseRepository(session)


def get_state_store() -> BaseWorkflowStateStore:
    return SqlWorkflowStateStore(get_db_manager().sessionmaker)


def get_workspace_service(
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
    user_repo: BaseUserRepository = Depends(get_user_repo),
) -> WorkspaceService:
    return WorkspaceService(workspace_repo=workspace_repo, user_repo=user_repo)


def get_competitor_service(
    competitor_repo: BaseCompetitorRepository = Depends(get_competitor_repo),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> CompetitorService:
    return CompetitorService(competitor_repo=competitor_repo, workspace_repo=workspace_repo)


def get_source_service(
    source_repo: BaseSourceRepository = Depends(get_source_repo),
    competitor_repo: BaseCompetitorRepository = Depends(get_competitor_repo),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> SourceService:
    return SourceService(
        source_repo=source_repo, competitor_repo=competitor_repo, workspace_repo=workspace_repo
    )


def get_document_service(
    document_repo: BaseDocumentRepository = Depends(get_document_repo),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> DocumentService:
    return DocumentService(document_repo=document_repo, workspace_repo=workspace_repo)


def get_intelligence_service(
    intelligence_repo: BaseIntelligenceRepository = Depends(get_intelligence_repo),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> IntelligenceService:
    return IntelligenceService(intelligence_repo=intelligence_repo, workspace_repo=workspace_repo)


def get_decision_service(
    decision_repo: BaseDecisionRepository = Depends(get_decision_repo),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> DecisionService:
    return DecisionService(decision_repo=decision_repo, workspace_repo=workspace_repo)


def get_workflow_document_repo() -> BaseDocumentRepository:
    return SqlDocumentRepository(get_db_manager().sessionmaker)


def get_workflow_intelligence_repo() -> BaseIntelligenceRepository:
    return SqlIntelligenceRepository(get_db_manager().sessionmaker)


def get_workflow_decision_repo() -> BaseDecisionRepository:
    return SqlDecisionRepository(get_db_manager().sessionmaker)


def get_chunking_service() -> ChunkingService:
    return ChunkingService()


def get_embedding_service(
    settings: Settings = Depends(get_settings),
) -> EmbeddingService:
    """Provide configured embedding service. Uses BGEM3 in production, mock in test/dev."""
    cfg = get_settings() if isinstance(settings, DependsClass) else (settings or get_settings())
    if cfg.environment == "production" or cfg.embedding_provider == "bge_m3":
        if cfg.environment == "production" and cfg.embedding_provider == "mock":
            raise ValueError("Mock embedding provider is not permitted in production environment")
        from competitive_intelligence.knowledge_pipeline.embeddings.bge_m3_provider import (
            BGEM3EmbeddingProvider,
        )

        provider = BGEM3EmbeddingProvider(
            model_name=cfg.embedding_model,
            device=cfg.embedding_device,
            batch_size=cfg.embedding_batch_size,
        )
        return EmbeddingService(provider=provider)

    # Lightweight mock embedding provider for fast testing/dev
    return EmbeddingService(provider=MockEmbeddingProvider(dimension=1024))


def get_vector_store(
    settings: Settings = Depends(get_settings),
) -> BaseVectorStore:
    """Provide configured vector store. Connects to real Qdrant service in production."""
    cfg = get_settings() if isinstance(settings, DependsClass) else (settings or get_settings())
    if cfg.environment == "production":
        store_config = VectorStoreConfig(
            collection_name=cfg.qdrant_collection_name,
            vector_dimensions=1024,
            distance_metric=DistanceMetric.COSINE,
            url=cfg.effective_qdrant_url,
            api_key=cfg.qdrant_api_key,
            timeout=cfg.qdrant_timeout,
            location=None,
        )
        return QdrantVectorStore(config=store_config)

    # In development, connect to remote Qdrant if explicit URL or remote host is set
    if cfg.qdrant_url or (cfg.qdrant_host != "localhost" and cfg.qdrant_location != ":memory:"):
        store_config = VectorStoreConfig(
            collection_name=cfg.qdrant_collection_name,
            vector_dimensions=1024,
            distance_metric=DistanceMetric.COSINE,
            url=cfg.effective_qdrant_url,
            api_key=cfg.qdrant_api_key,
            timeout=cfg.qdrant_timeout,
            location=None,
        )
        return QdrantVectorStore(config=store_config)

    # Default in-memory Qdrant store for fast isolated tests
    return QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name=cfg.qdrant_collection_name,
            vector_dimensions=1024,
            location=cfg.qdrant_location or ":memory:",
        )
    )


def get_knowledge_ingestion_service(
    document_repo: BaseDocumentRepository = Depends(get_workflow_document_repo),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> KnowledgeIngestionService:
    d_repo = get_workflow_document_repo() if isinstance(document_repo, DependsClass) else document_repo
    c_svc = get_chunking_service() if isinstance(chunking_service, DependsClass) else chunking_service
    e_svc = get_embedding_service() if isinstance(embedding_service, DependsClass) else embedding_service
    v_store = get_vector_store() if isinstance(vector_store, DependsClass) else vector_store
    return KnowledgeIngestionService(
        document_repo=d_repo,
        chunking_service=c_svc,
        embedding_service=e_svc,
        vector_store=v_store,
    )


def get_knowledge_processing_stage(
    ingestion_service: KnowledgeIngestionService = Depends(get_knowledge_ingestion_service),
) -> KnowledgeProcessingStage:
    ing_svc = get_knowledge_ingestion_service() if isinstance(ingestion_service, DependsClass) else ingestion_service
    return KnowledgeProcessingStage(ingestion_service=ing_svc)


def get_semantic_retriever(
    vector_store: BaseVectorStore = Depends(get_vector_store),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> BaseRetriever:
    v_store = get_vector_store() if isinstance(vector_store, DependsClass) else vector_store
    emb_svc = get_embedding_service() if isinstance(embedding_service, DependsClass) else embedding_service
    semantic_core = SemanticRetriever(
        vector_store=v_store,
        embedding_provider=emb_svc.provider,
    )
    return SemanticRetrieverAdapter(semantic_retriever=semantic_core)


def get_adaptive_retriever(
    semantic_retriever: BaseRetriever = Depends(get_semantic_retriever),
) -> AdaptiveRetriever:
    ret = get_semantic_retriever() if isinstance(semantic_retriever, DependsClass) else semantic_retriever
    router = DeterministicRetrievalRouter()
    return AdaptiveRetriever(
        router=router,
        bm25_retriever=ret,
        semantic_retriever=ret,
        hybrid_retriever=ret,
        reranked_retriever=ret,
    )


def get_llm_client() -> BaseLLMClient:
    return create_llm_client()


def get_agentic_retriever(
    adaptive_retriever: AdaptiveRetriever = Depends(get_adaptive_retriever),
    llm_client: BaseLLMClient = Depends(get_llm_client),
) -> BaseRetriever:
    ret = get_adaptive_retriever() if isinstance(adaptive_retriever, DependsClass) else adaptive_retriever
    client = get_llm_client() if isinstance(llm_client, DependsClass) else llm_client
    return AgenticRetriever(executor_retriever=ret, llm_client=client)


def get_production_retriever(
    adaptive_retriever: AdaptiveRetriever = Depends(get_adaptive_retriever),
    agentic_retriever: BaseRetriever = Depends(get_agentic_retriever),
) -> BaseRetriever:
    ad_ret = get_adaptive_retriever() if isinstance(adaptive_retriever, DependsClass) else adaptive_retriever
    ag_ret = get_agentic_retriever() if isinstance(agentic_retriever, DependsClass) else agentic_retriever
    return SelectiveAgenticRetriever(
        adaptive_retriever=ad_ret,
        agentic_retriever=ag_ret,
    )


def get_retrieval_stage(
    retriever: BaseRetriever = Depends(get_production_retriever),
) -> RetrievalStage:
    ret = get_production_retriever() if isinstance(retriever, DependsClass) else retriever
    return RetrievalStage(retriever=ret, allow_mock_fallback=False)


def get_event_analysis_stage(
    intelligence_repo: BaseIntelligenceRepository = Depends(get_workflow_intelligence_repo),
    llm_client: BaseLLMClient = Depends(get_llm_client),
) -> EventAnalysisStage:
    repo = get_workflow_intelligence_repo() if isinstance(intelligence_repo, DependsClass) else intelligence_repo
    client = get_llm_client() if isinstance(llm_client, DependsClass) else llm_client
    evidence_analyzer = RetrievedEvidenceAnalyzer(
        evidence_interpreter=LLMEvidenceInterpreter(llm_client=client)
    )
    return EventAnalysisStage(
        evidence_analyzer=evidence_analyzer,
        intelligence_repo=repo,
    )


def get_signal_detection_stage(
    intelligence_repo: BaseIntelligenceRepository = Depends(get_workflow_intelligence_repo),
) -> SignalDetectionStage:
    repo = get_workflow_intelligence_repo() if isinstance(intelligence_repo, DependsClass) else intelligence_repo
    return SignalDetectionStage(intelligence_repo=repo)


def get_insight_generation_stage(
    intelligence_repo: BaseIntelligenceRepository = Depends(get_workflow_intelligence_repo),
    llm_client: BaseLLMClient = Depends(get_llm_client),
) -> InsightGenerationStage:
    repo = get_workflow_intelligence_repo() if isinstance(intelligence_repo, DependsClass) else intelligence_repo
    client = get_llm_client() if isinstance(llm_client, DependsClass) else llm_client
    strategic_analyzer = DeterministicStrategicAnalyzer(
        strategic_interpreter=LLMStrategicInterpreter(llm_client=client)
    )
    return InsightGenerationStage(
        strategic_analyzer=strategic_analyzer,
        intelligence_repo=repo,
    )


def get_decision_support_stage(
    decision_repo: BaseDecisionRepository = Depends(get_workflow_decision_repo),
    llm_client: BaseLLMClient = Depends(get_llm_client),
) -> DecisionSupportStage:
    repo = get_workflow_decision_repo() if isinstance(decision_repo, DependsClass) else decision_repo
    client = get_llm_client() if isinstance(llm_client, DependsClass) else llm_client
    decision_analyzer = DeterministicDecisionAnalyzer(
        option_generator=LLMOptionGenerator(llm_client=client)
    )
    return DecisionSupportStage(
        decision_analyzer=decision_analyzer,
        decision_repo=repo,
    )


def get_workflow_orchestrator(
    state_store: BaseWorkflowStateStore = Depends(get_state_store),
    knowledge_stage: KnowledgeProcessingStage = Depends(get_knowledge_processing_stage),
    retrieval_stage: RetrievalStage = Depends(get_retrieval_stage),
    event_stage: EventAnalysisStage = Depends(get_event_analysis_stage),
    signal_stage: SignalDetectionStage = Depends(get_signal_detection_stage),
    insight_stage: InsightGenerationStage = Depends(get_insight_generation_stage),
    decision_stage: DecisionSupportStage = Depends(get_decision_support_stage),
) -> BaseWorkflowOrchestrator:
    s_store = get_state_store() if isinstance(state_store, DependsClass) else state_store
    k_stage = get_knowledge_processing_stage() if isinstance(knowledge_stage, DependsClass) else knowledge_stage
    r_stage = get_retrieval_stage() if isinstance(retrieval_stage, DependsClass) else retrieval_stage
    e_stage = get_event_analysis_stage() if isinstance(event_stage, DependsClass) else event_stage
    sig_stage = get_signal_detection_stage() if isinstance(signal_stage, DependsClass) else signal_stage
    ins_stage = get_insight_generation_stage() if isinstance(insight_stage, DependsClass) else insight_stage
    dec_stage = get_decision_support_stage() if isinstance(decision_stage, DependsClass) else decision_stage
    return DeterministicWorkflowOrchestrator(
        state_store=s_store,
        stages={
            WorkflowStageType.KNOWLEDGE_PROCESSING: k_stage,
            WorkflowStageType.RETRIEVAL: r_stage,
            WorkflowStageType.EVENT_ANALYSIS: e_stage,
            WorkflowStageType.SIGNAL_DETECTION: sig_stage,
            WorkflowStageType.INSIGHT_GENERATION: ins_stage,
            WorkflowStageType.DECISION_SUPPORT: dec_stage,
        },
    )


def get_workflow_service(
    state_store: BaseWorkflowStateStore = Depends(get_state_store),
    orchestrator: BaseWorkflowOrchestrator = Depends(get_workflow_orchestrator),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
    competitor_repo: BaseCompetitorRepository = Depends(get_competitor_repo),
) -> WorkflowExecutionService:
    s_store = get_state_store() if isinstance(state_store, DependsClass) else state_store
    orch = get_workflow_orchestrator() if isinstance(orchestrator, DependsClass) else orchestrator
    ws_repo = get_workspace_repo() if isinstance(workspace_repo, DependsClass) else workspace_repo
    comp_repo = get_competitor_repo() if isinstance(competitor_repo, DependsClass) else competitor_repo
    return WorkflowExecutionService(
        state_store=s_store,
        orchestrator=orch,
        workspace_repo=ws_repo,
        competitor_repo=comp_repo,
    )



def get_monitoring_service(
    source_repo: BaseSourceRepository = Depends(get_source_repo),
    lease_repo: BaseMonitoringLeaseRepository = Depends(get_monitoring_lease_repo),
    workflow_service: WorkflowExecutionService = Depends(get_workflow_service),
    workspace_repo: BaseWorkspaceRepository = Depends(get_workspace_repo),
) -> SourceMonitoringService:
    return SourceMonitoringService(
        source_repo=source_repo,
        lease_repo=lease_repo,
        workflow_service=workflow_service,
        workspace_repo=workspace_repo,
    )


async def get_current_workspace(
    workspace_id: str = Path(..., description="Target workspace ID"),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
) -> Workspace:
    """Dependency validating that the specified workspace exists."""
    return await workspace_service.verify_workspace_exists(workspace_id)
