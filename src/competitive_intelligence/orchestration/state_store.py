"""Thread-safe reference in-memory state store for workflow execution state."""

from datetime import datetime, timezone

import anyio

from competitive_intelligence.orchestration.interfaces import BaseWorkflowStateStore
from competitive_intelligence.orchestration.models import (
    StageExecutionRecord,
    WorkflowProducedOutputs,
    WorkflowRun,
    WorkflowStageType,
)


class InMemoryWorkflowStateStore(BaseWorkflowStateStore):
    """Thread-safe in-memory state store implementing BaseWorkflowStateStore.

    Persists workflow run state, stage execution checkpoints, and domain output references.
    Does NOT store full domain entity content (preserves domain repository boundary).
    """

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}
        self._lock = anyio.Lock()

    async def create_run(self, run: WorkflowRun) -> None:
        async with self._lock:
            self._runs[run.run_id] = run.model_copy(deep=True)

    async def update_run(self, run: WorkflowRun) -> None:
        async with self._lock:
            existing = self._runs.get(run.run_id)
            run_copy = run.model_copy(deep=True)
            if existing:
                for stage_type, rec in existing.stage_records.items():
                    if stage_type not in run_copy.stage_records:
                        run_copy.stage_records[stage_type] = rec
            self._runs[run.run_id] = run_copy

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run else None

    async def get_run_by_idempotency_key(
        self,
        workspace_id: str,
        idempotency_key: str,
    ) -> WorkflowRun | None:
        async with self._lock:
            for r in self._runs.values():
                if (
                    r.workspace_id == workspace_id
                    and r.metadata.get("idempotency_key") == idempotency_key
                ):
                    return r.model_copy(deep=True)
            return None

    async def list_runs(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        async with self._lock:
            matched = [
                r.model_copy(deep=True)
                for r in self._runs.values()
                if r.workspace_id == workspace_id
                and (company_id is None or r.company_id == company_id)
            ]
            matched.sort(key=lambda x: x.started_at, reverse=True)
            return matched[:limit]

    async def checkpoint_stage(
        self,
        run_id: str,
        stage: WorkflowStageType,
        record: StageExecutionRecord,
        outputs: WorkflowProducedOutputs,
    ) -> None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run:
                run.stage_records[stage] = record.model_copy(deep=True)
                run.produced_outputs = outputs.model_copy(deep=True)
                run.last_checkpointed_at = datetime.now(timezone.utc)
                if record.error:
                    run.errors.append(record.error.model_copy(deep=True))
