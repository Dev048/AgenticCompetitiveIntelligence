"""Retrieval stage wrapper."""

from datetime import datetime, timezone
import hashlib
import time

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
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class RetrievalStage(BaseWorkflowStage):
    """Executes search retrieval to gather relevant evidence context."""

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        allow_mock_fallback: bool = True,
    ) -> None:
        self.retriever = retriever
        self.allow_mock_fallback = allow_mock_fallback

    @property
    def stage_type(self) -> WorkflowStageType:
        return WorkflowStageType.RETRIEVAL

    def compute_idempotency_key(self, context: StageExecutionContext) -> str:
        q_text = context.inputs.query_text or (
            context.current_version.content[:100] if context.current_version else "default_query"
        )
        return f"{self.stage_type.value}:{context.inputs.workspace_id}:{context.inputs.company_id}:{hashlib.sha256(q_text.encode('utf-8')).hexdigest()}"

    async def execute(self, context: StageExecutionContext) -> StageExecutionRecord:
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc)
        record = StageExecutionRecord(
            stage=self.stage_type,
            status=StageStatus.RUNNING,
            started_at=started_at,
            idempotency_key=self.compute_idempotency_key(context),
            input_summary={
                "has_query_text": bool(context.inputs.query_text),
                "has_current_version": bool(context.current_version),
            },
        )

        try:
            q_text = context.inputs.query_text
            if not q_text and context.current_version:
                q_text = context.current_version.content[:200]
            if not q_text:
                q_text = f"Latest developments and updates regarding {context.inputs.company_id}"

            retrieval_query = RetrievalQuery(
                query_text=q_text,
                metadata_filter=MetadataFilter(
                    workspace_id=context.inputs.workspace_id,
                    company_ids=[context.inputs.company_id],
                ),
                top_k=context.budget.reranking_top_k,
            )

            results: list[RetrievalResult] = []
            retrieval_mode = "mock"
            if self.retriever:
                results = await self.retriever.retrieve(retrieval_query)
                retrieval_mode = "production"
            elif self.allow_mock_fallback:
                # Deterministic baseline mock retrieval when external retriever not configured
                chunk_id = f"chunk_{context.inputs.company_id}_1"
                doc_id = context.inputs.document_id or f"doc_{context.inputs.company_id}"
                ver_id = context.current_version.version_id if context.current_version else "ver_1"
                ret_item = RetrievalResult(
                    chunk_id=chunk_id,
                    score=0.92,
                    text=q_text,
                    document_id=doc_id,
                    version_id=ver_id,
                    company_id=context.inputs.company_id,
                    workspace_id=context.inputs.workspace_id,
                    canonical_url=context.inputs.source_url or "https://example.com",
                )
                results = [ret_item]
            else:
                raise RuntimeError(
                    "RetrievalStage has no configured retriever and mock fallback is disabled."
                )

            context.retrieval_results = results

            # Record produced output IDs
            for res in results:
                context.run.produced_outputs.chunk_ids.append(res.chunk_id)

            record.status = StageStatus.COMPLETED
            record.output_ids = {
                "retrieved_chunks_count": len(results),
                "chunk_ids": [r.chunk_id for r in results],
                "retrieval_mode": retrieval_mode,
            }
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record

        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = StageErrorRecord(
                stage=self.stage_type,
                error_category=ErrorCategory.TRANSIENT,
                error_message=str(exc),
                is_retryable=True,
            )
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = (time.perf_counter() - t0) * 1000.0
            return record
