"""Knowledge Processing stage wrapper coordinating full knowledge ingestion lifecycle."""

from datetime import datetime, timezone
import hashlib
import time

from competitive_intelligence.application.interfaces import BaseDocumentRepository
from competitive_intelligence.infrastructure.database.repositories.in_memory_doc_repo import (
    InMemoryDocumentRepository,
)
from competitive_intelligence.knowledge_pipeline.change_detector import ChangeDetector
from competitive_intelligence.knowledge_pipeline.chunking_service import (
    ChunkingService,
)
from competitive_intelligence.knowledge_pipeline.deduplicator import (
    DocumentDeduplicator,
)
from competitive_intelligence.knowledge_pipeline.embeddings.embedding_service import (
    EmbeddingService,
)
from competitive_intelligence.knowledge_pipeline.ingestion_service import (
    KnowledgeIngestionService,
)
from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseChangeDetector,
    BaseDocumentDeduplicator,
    BaseDocumentNormalizer,
    BaseVectorStore,
)
from competitive_intelligence.knowledge_pipeline.normalizer import (
    DocumentNormalizer,
)
from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowStage,
    StageExecutionContext,
)
from competitive_intelligence.orchestration.models import (
    ErrorCategory,
    StageErrorRecord,
    StageExecutionRecord,
    StageStatus,
    WorkflowStageType,
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


class KnowledgeProcessingStage(BaseWorkflowStage):
    """Executes normalization, hashing, deduplication, change detection, chunking, embedding, and vector indexing."""

    def __init__(
        self,
        ingestion_service: KnowledgeIngestionService | None = None,
        document_repo: BaseDocumentRepository | None = None,
        chunking_service: ChunkingService | None = None,
        normalizer: BaseDocumentNormalizer | None = None,
        change_detector: BaseChangeDetector | None = None,
        deduplicator: BaseDocumentDeduplicator | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        if ingestion_service is not None:
            self.ingestion_service = ingestion_service
        else:
            repo = document_repo or InMemoryDocumentRepository()
            self.ingestion_service = KnowledgeIngestionService(
                document_repo=repo,
                chunking_service=chunking_service,
                normalizer=normalizer,
                change_detector=change_detector,
                deduplicator=deduplicator,
                embedding_service=embedding_service,
                vector_store=vector_store,
            )

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.KNOWLEDGE_PROCESSING

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        content = context.inputs.raw_document_content or ""
        doc_id = context.inputs.document_id or "default_doc"
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{doc_id}:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={
                "document_id": context.inputs.document_id,
                "has_content": bool(context.inputs.raw_document_content),
            },
        )

        try:
            content = context.inputs.raw_document_content
            if not content or not content.strip():
                err = StageErrorRecord(
                    stage=self.stage_type,
                    error_category=ErrorCategory.DATA_QUALITY,
                    error_message="Raw document content is empty.",
                    is_retryable=False,
                )
                record.status = StageStatus.FAILED
                record.error = err
                record.completed_at = datetime.now(timezone.utc)
                record.duration_ms = (time.perf_counter() - t0) * 1000.0
                return record

            source_url = context.inputs.source_url or "https://example.com/source"
            source_id = context.inputs.source_id or context.inputs.metadata.get("source_id", f"src_{context.inputs.company_id}")

            source = Source(
                identity=SourceIdentity(
                    source_id=str(source_id),
                    company_id=context.inputs.company_id,
                    workspace_id=context.inputs.workspace_id,
                    display_name=f"{context.inputs.company_id} Monitored Source",
                ),
                classification=SourceClassification(
                    category=SourceCategory.WEBSITE,
                    platform=PlatformType.WEB,
                    content_type=ContentType.TEXT,
                ),
                location=SourceLocation(
                    url_or_identifier=source_url,
                    access_pattern=AccessPattern.PUBLIC_WEB,
                ),
                acquisition=AcquisitionConfig(
                    acquisition_method=AcquisitionMethod.CRAWL4AI,
                    tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
                ),
            )

            acq_result = AcquisitionResult(
                status=AcquisitionStatus.SUCCESS,
                content=content,
                acquired_at=datetime.now(timezone.utc),
            )

            # Execute end-to-end ingestion
            ingestion_result = await self.ingestion_service.ingest(
                source=source, result=acq_result
            )

            # Update context
            context.current_version = ingestion_result.version
            context.change_result = ingestion_result.change_result

            # Record produced output IDs
            context.run.produced_outputs.document_ids.append(
                ingestion_result.document.document_id
            )
            context.run.produced_outputs.version_ids.append(
                ingestion_result.version.version_id
            )
            for chunk in ingestion_result.chunks:
                context.run.produced_outputs.chunk_ids.append(chunk.chunk_id)
            context.run.produced_outputs.chunks_created_count += ingestion_result.chunks_created_count
            context.run.produced_outputs.vector_points_upserted_count += ingestion_result.vector_points_upserted_count

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "document_id": ingestion_result.document.document_id,
                "version_id": ingestion_result.version.version_id,
                "change_type": ingestion_result.change_result.change_type.value,
                "chunks_count": len(ingestion_result.chunks),
                "embeddings_count": ingestion_result.embeddings_count,
                "vectors_count": ingestion_result.vectors_count,
                "chunks_created_count": ingestion_result.chunks_created_count,
                "vector_points_upserted_count": ingestion_result.vector_points_upserted_count,
                "skipped_unchanged": ingestion_result.is_skipped_unchanged,
            }
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record

        except Exception as exc:
            # Check if this is a transient vector store / provider error or permanent
            is_transient = "VectorStoreError" in type(exc).__name__ or "EmbeddingProviderError" in type(exc).__name__
            err_cat = ErrorCategory.TRANSIENT if is_transient else ErrorCategory.PERMANENT
            record.status = StageStatus.FAILED
            record.error = StageErrorRecord(
                stage=self.stage_type,
                error_category=err_cat,
                error_message=str(exc),
                is_retryable=is_transient,
            )
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record
