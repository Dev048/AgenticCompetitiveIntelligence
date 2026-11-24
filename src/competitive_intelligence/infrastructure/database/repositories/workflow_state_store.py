"""SQL implementation of BaseWorkflowStateStore for durable workflow execution state."""

from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from competitive_intelligence.infrastructure.database.models import (
    StageExecutionRecordOrm,
    WorkflowRunOrm,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.orchestration.interfaces import BaseWorkflowStateStore
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


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlWorkflowStateStore(BaseWorkflowStateStore):
    """PostgreSQL-backed durable implementation of BaseWorkflowStateStore."""

    def __init__(
        self,
        session_manager: DatabaseSessionManager | async_sessionmaker[AsyncSession] | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        if isinstance(session_manager, DatabaseSessionManager):
            self._sessionmaker = session_manager.sessionmaker
        elif session_manager is not None:
            self._sessionmaker = session_manager
        elif session_factory is not None:
            self._sessionmaker = session_factory
        else:
            self._sessionmaker = get_db_manager().sessionmaker

    async def create_run(self, run: WorkflowRun) -> None:
        """Persist a new workflow run record and initial stage records."""
        async with self._sessionmaker() as session:
            errors_data = [e.model_dump(mode="json") for e in run.errors]

            orm_run = WorkflowRunOrm(
                run_id=run.run_id,
                workflow_id=run.workflow_id,
                workspace_id=run.workspace_id,
                company_id=run.company_id,
                trigger_type=run.trigger_type.value,
                status=run.status.value,
                current_stage=run.current_stage.value if run.current_stage else None,
                target_stop_stage=run.target_stop_stage.value,
                produced_outputs=run.produced_outputs.model_dump(mode="json"),
                errors=errors_data,
                started_at=run.started_at,
                completed_at=run.completed_at,
                last_checkpointed_at=run.last_checkpointed_at,
                correlation_id=run.correlation_id,
                idempotency_key=run.metadata.get("idempotency_key"),
                metadata_=run.metadata,
            )
            session.add(orm_run)
            await session.flush()

            for stage_type, record in run.stage_records.items():
                err_dict = record.error.model_dump(mode="json") if record.error else None
                orm_stage = StageExecutionRecordOrm(
                    run_id=run.run_id,
                    stage=stage_type.value,
                    status=record.status.value,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    duration_ms=record.duration_ms,
                    attempt_count=record.attempt_count,
                    idempotency_key=record.idempotency_key,
                    input_summary=record.input_summary,
                    output_ids=record.output_ids,
                    error=err_dict,
                )
                session.add(orm_stage)

            await session.commit()

    async def update_run(self, run: WorkflowRun) -> None:
        """Update existing workflow run state and stage records."""
        async with self._sessionmaker() as session:
            stmt = select(WorkflowRunOrm).where(WorkflowRunOrm.run_id == run.run_id)
            result = await session.execute(stmt)
            orm_run = result.scalar_one_or_none()
            if orm_run is None:
                return

            orm_run.status = run.status.value
            orm_run.current_stage = run.current_stage.value if run.current_stage else None
            orm_run.target_stop_stage = run.target_stop_stage.value
            orm_run.produced_outputs = run.produced_outputs.model_dump(mode="json")
            orm_run.errors = [e.model_dump(mode="json") for e in run.errors]
            orm_run.completed_at = run.completed_at
            orm_run.last_checkpointed_at = run.last_checkpointed_at
            orm_run.metadata_ = run.metadata

            for stage_type, record in run.stage_records.items():
                stage_stmt = (
                    select(StageExecutionRecordOrm)
                    .where(StageExecutionRecordOrm.run_id == run.run_id)
                    .where(StageExecutionRecordOrm.stage == stage_type.value)
                    .where(StageExecutionRecordOrm.attempt_count == record.attempt_count)
                )
                stage_res = await session.execute(stage_stmt)
                orm_stage = stage_res.scalar_one_or_none()
                err_dict = record.error.model_dump(mode="json") if record.error else None

                if orm_stage is None:
                    orm_stage = StageExecutionRecordOrm(
                        run_id=run.run_id,
                        stage=stage_type.value,
                        status=record.status.value,
                        started_at=record.started_at,
                        completed_at=record.completed_at,
                        duration_ms=record.duration_ms,
                        attempt_count=record.attempt_count,
                        idempotency_key=record.idempotency_key,
                        input_summary=record.input_summary,
                        output_ids=record.output_ids,
                        error=err_dict,
                    )
                    session.add(orm_stage)
                else:
                    orm_stage.status = record.status.value
                    orm_stage.started_at = record.started_at
                    orm_stage.completed_at = record.completed_at
                    orm_stage.duration_ms = record.duration_ms
                    orm_stage.input_summary = record.input_summary
                    orm_stage.output_ids = record.output_ids
                    orm_stage.error = err_dict

            await session.commit()

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        """Retrieve a workflow run by ID with all associated stage execution records."""
        async with self._sessionmaker() as session:
            stmt = select(WorkflowRunOrm).where(WorkflowRunOrm.run_id == run_id)
            result = await session.execute(stmt)
            orm_run = result.scalar_one_or_none()
            if orm_run is None:
                return None

            stage_stmt = (
                select(StageExecutionRecordOrm)
                .where(StageExecutionRecordOrm.run_id == run_id)
                .order_by(StageExecutionRecordOrm.created_at.asc())
            )
            stage_res = await session.execute(stage_stmt)
            stage_rows = stage_res.scalars().all()

            return self._orm_to_run(orm_run, stage_rows)

    async def get_run_by_idempotency_key(
        self,
        workspace_id: str,
        idempotency_key: str,
    ) -> WorkflowRun | None:
        """Retrieve a workflow run by its workspace-scoped idempotency key."""
        async with self._sessionmaker() as session:
            stmt = (
                select(WorkflowRunOrm)
                .where(
                    WorkflowRunOrm.workspace_id == workspace_id,
                    WorkflowRunOrm.idempotency_key == idempotency_key,
                )
            )
            result = await session.execute(stmt)
            orm_run = result.scalar_one_or_none()
            if orm_run is None:
                return None

            stage_stmt = (
                select(StageExecutionRecordOrm)
                .where(StageExecutionRecordOrm.run_id == orm_run.run_id)
                .order_by(StageExecutionRecordOrm.created_at.asc())
            )
            stage_res = await session.execute(stage_stmt)
            stage_rows = stage_res.scalars().all()

            return self._orm_to_run(orm_run, stage_rows)

    async def list_runs(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        """List workflow runs for a workspace ordered chronologically."""
        async with self._sessionmaker() as session:
            stmt = select(WorkflowRunOrm).where(WorkflowRunOrm.workspace_id == workspace_id)
            if company_id is not None:
                stmt = stmt.where(WorkflowRunOrm.company_id == company_id)
            stmt = stmt.order_by(WorkflowRunOrm.started_at.desc()).limit(limit)

            result = await session.execute(stmt)
            runs = result.scalars().all()

            runs_list: list[WorkflowRun] = []
            for r in runs:
                stage_stmt = (
                    select(StageExecutionRecordOrm)
                    .where(StageExecutionRecordOrm.run_id == r.run_id)
                    .order_by(StageExecutionRecordOrm.created_at.asc())
                )
                stage_res = await session.execute(stage_stmt)
                stage_rows = stage_res.scalars().all()
                runs_list.append(self._orm_to_run(r, stage_rows))

            return runs_list

    async def checkpoint_stage(
        self,
        run_id: str,
        stage: WorkflowStageType,
        record: StageExecutionRecord,
        outputs: WorkflowProducedOutputs,
    ) -> None:
        """Checkpoint stage execution record and updated domain output references."""
        async with self._sessionmaker() as session:
            stmt = select(WorkflowRunOrm).where(WorkflowRunOrm.run_id == run_id)
            result = await session.execute(stmt)
            orm_run = result.scalar_one_or_none()
            if orm_run is None:
                return

            now = datetime.now(timezone.utc)
            orm_run.current_stage = stage.value
            orm_run.last_checkpointed_at = now
            orm_run.produced_outputs = outputs.model_dump(mode="json")

            if record.error:
                err_dict = record.error.model_dump(mode="json")
                errs = list(orm_run.errors)
                errs.append(err_dict)
                orm_run.errors = errs

            # Upsert stage execution record
            stage_stmt = (
                select(StageExecutionRecordOrm)
                .where(StageExecutionRecordOrm.run_id == run_id)
                .where(StageExecutionRecordOrm.stage == stage.value)
                .where(StageExecutionRecordOrm.attempt_count == record.attempt_count)
            )
            stage_res = await session.execute(stage_stmt)
            orm_stage = stage_res.scalar_one_or_none()

            stage_err_dict: dict[str, Any] | None = (
                record.error.model_dump(mode="json") if record.error else None
            )

            if orm_stage is None:
                orm_stage = StageExecutionRecordOrm(
                    run_id=run_id,
                    stage=stage.value,
                    status=record.status.value,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    duration_ms=record.duration_ms,
                    attempt_count=record.attempt_count,
                    idempotency_key=record.idempotency_key,
                    input_summary=record.input_summary,
                    output_ids=record.output_ids,
                    error=stage_err_dict,
                )
                session.add(orm_stage)
            else:
                orm_stage.status = record.status.value
                orm_stage.started_at = record.started_at
                orm_stage.completed_at = record.completed_at
                orm_stage.duration_ms = record.duration_ms
                orm_stage.input_summary = record.input_summary
                orm_stage.output_ids = record.output_ids
                orm_stage.error = stage_err_dict

            await session.commit()

    @staticmethod
    def _orm_to_run(
        orm_run: WorkflowRunOrm, stage_rows: Sequence[StageExecutionRecordOrm]
    ) -> WorkflowRun:
        stage_records: dict[WorkflowStageType, StageExecutionRecord] = {}
        for s in stage_rows:
            stype = WorkflowStageType(s.stage)
            err_record: StageErrorRecord | None = None
            if s.error:
                occ_str = s.error.get("occurred_at")
                occ_at = (
                    datetime.fromisoformat(occ_str)
                    if isinstance(occ_str, str)
                    else (occ_str or datetime.now(timezone.utc))
                )
                err_record = StageErrorRecord(
                    stage=WorkflowStageType(s.error["stage"]),
                    error_category=ErrorCategory(s.error["error_category"]),
                    error_message=s.error["error_message"],
                    stack_trace=s.error.get("stack_trace"),
                    occurred_at=_ensure_utc(occ_at) or datetime.now(timezone.utc),
                    attempt_number=s.error.get("attempt_number", 1),
                    is_retryable=s.error.get("is_retryable", False),
                )

            stage_records[stype] = StageExecutionRecord(
                stage=stype,
                status=StageStatus(s.status),
                started_at=_ensure_utc(s.started_at),
                completed_at=_ensure_utc(s.completed_at),
                duration_ms=s.duration_ms,
                attempt_count=s.attempt_count,
                idempotency_key=s.idempotency_key,
                input_summary=s.input_summary,
                output_ids=s.output_ids,
                error=err_record,
            )

        errors_list: list[StageErrorRecord] = []
        for e in orm_run.errors:
            occ_str = e.get("occurred_at")
            occ_at = (
                datetime.fromisoformat(occ_str)
                if isinstance(occ_str, str)
                else (occ_str or datetime.now(timezone.utc))
            )
            errors_list.append(
                StageErrorRecord(
                    stage=WorkflowStageType(e["stage"]),
                    error_category=ErrorCategory(e["error_category"]),
                    error_message=e["error_message"],
                    stack_trace=e.get("stack_trace"),
                    occurred_at=_ensure_utc(occ_at) or datetime.now(timezone.utc),
                    attempt_number=e.get("attempt_number", 1),
                    is_retryable=e.get("is_retryable", False),
                )
            )

        prod_out = WorkflowProducedOutputs(**orm_run.produced_outputs)
        started_at = _ensure_utc(orm_run.started_at)
        assert started_at is not None

        return WorkflowRun(
            run_id=orm_run.run_id,
            workflow_id=orm_run.workflow_id,
            workspace_id=orm_run.workspace_id,
            company_id=orm_run.company_id,
            trigger_type=TriggerType(orm_run.trigger_type),
            status=WorkflowRunStatus(orm_run.status),
            current_stage=WorkflowStageType(orm_run.current_stage)
            if orm_run.current_stage
            else None,
            target_stop_stage=WorkflowStageType(orm_run.target_stop_stage),
            stage_records=stage_records,
            produced_outputs=prod_out,
            errors=errors_list,
            started_at=started_at,
            completed_at=_ensure_utc(orm_run.completed_at),
            last_checkpointed_at=_ensure_utc(orm_run.last_checkpointed_at),
            correlation_id=orm_run.correlation_id,
            metadata=orm_run.metadata_,
        )
