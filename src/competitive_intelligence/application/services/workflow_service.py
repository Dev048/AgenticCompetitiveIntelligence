"""Application use-case service for Operational Workflow orchestration and dispatch."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine
from uuid import uuid4

from competitive_intelligence.application.interfaces import (
    BaseCompetitorRepository,
    BaseWorkspaceRepository,
)
from competitive_intelligence.core.errors import (
    EntityNotFoundError,
    ValidationError,
    WorkspaceIsolationError,
)
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.orchestration.interfaces import (
    BaseWorkflowOrchestrator,
    BaseWorkflowStateStore,
)
from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowInputs,
    WorkflowResult,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowStageType,
)
from competitive_intelligence.orchestration.policies import ExecutionBudget

logger = get_logger(__name__)


class WorkflowExecutionService:
    """Application use-case service coordinating workflow submission, status, and background dispatch.

    Decouples HTTP/API endpoints from the execution engine, ensuring a future
    continuous-monitoring scheduler can trigger workflows using the identical interface.
    """

    def __init__(
        self,
        state_store: BaseWorkflowStateStore,
        orchestrator: BaseWorkflowOrchestrator | None = None,
        workspace_repo: BaseWorkspaceRepository | None = None,
        competitor_repo: BaseCompetitorRepository | None = None,
        task_dispatcher: Callable[[Coroutine[Any, Any, Any]], Any] | None = None,
    ) -> None:
        self._state_store = state_store
        if orchestrator is not None:
            self._orchestrator = orchestrator
        else:
            from competitive_intelligence.orchestration.orchestrator import (
                DeterministicWorkflowOrchestrator,
            )

            self._orchestrator = DeterministicWorkflowOrchestrator(state_store=state_store)
        self._workspace_repo = workspace_repo
        self._competitor_repo = competitor_repo
        self._task_dispatcher = task_dispatcher or asyncio.create_task
        self._submission_lock = asyncio.Lock()

    async def submit_workflow_run(
        self,
        workspace_id: str,
        inputs: WorkflowInputs,
        target_stop_stage: WorkflowStageType = WorkflowStageType.INSIGHT_GENERATION,
        trigger_type: TriggerType = TriggerType.MANUAL_USER_REQUEST,
        budget: ExecutionBudget | None = None,
    ) -> WorkflowRun:
        """Validate preconditions, create durable PENDING run, dispatch execution, and return immediately."""
        # 1. Precondition & workspace boundary checks
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        if inputs.workspace_id != workspace_id:
            inputs = inputs.model_copy(update={"workspace_id": workspace_id})

        if self._competitor_repo is not None:
            comp = await self._competitor_repo.get_by_id(inputs.company_id)
            if comp is None:
                raise EntityNotFoundError(f"Competitor '{inputs.company_id}' not found")
            if comp.workspace_id != workspace_id:
                raise WorkspaceIsolationError(
                    f"Competitor '{inputs.company_id}' does not belong to workspace '{workspace_id}'"
                )

        idempotency_key = inputs.metadata.get("idempotency_key")

        async with self._submission_lock:
            # 2. Check for existing workflow run by idempotency key
            if idempotency_key:
                existing_run = await self._state_store.get_run_by_idempotency_key(
                    workspace_id=workspace_id,
                    idempotency_key=str(idempotency_key),
                )
                if existing_run is not None:
                    logger.info(
                        "WORKFLOW_IDEMPOTENT_HIT",
                        workspace_id=workspace_id,
                        idempotency_key=idempotency_key,
                        run_id=existing_run.run_id,
                    )
                    return existing_run

            # 3. Create durable WorkflowRun in PENDING status
            run = WorkflowRun(
                run_id=str(uuid4()),
                workspace_id=workspace_id,
                company_id=inputs.company_id,
                trigger_type=trigger_type,
                status=WorkflowRunStatus.PENDING,
                target_stop_stage=target_stop_stage,
                started_at=datetime.now(timezone.utc),
                metadata=inputs.metadata,
            )

            # 4. Commit initial state to durable store (with concurrent collision handling)
            try:
                await self._state_store.create_run(run)
            except Exception as exc:
                if idempotency_key:
                    existing_run = await self._state_store.get_run_by_idempotency_key(
                        workspace_id=workspace_id,
                        idempotency_key=str(idempotency_key),
                    )
                    if existing_run is not None:
                        logger.info(
                            "WORKFLOW_IDEMPOTENT_CONCURRENT_HIT",
                            workspace_id=workspace_id,
                            idempotency_key=idempotency_key,
                            run_id=existing_run.run_id,
                        )
                        return existing_run
                raise exc

        # 5. Dispatch asynchronous execution ONLY for newly created runs
        async def _run_in_background() -> None:
            try:
                await self._orchestrator.run_workflow(
                    inputs=inputs,
                    target_stop_stage=target_stop_stage,
                    budget=budget,
                    run_id=run.run_id,
                )
            except Exception as err:
                logger.error("Background workflow execution failed", run_id=run.run_id, error=str(err))

        self._task_dispatcher(_run_in_background())

        return run

    async def get_workflow_run(self, workspace_id: str, run_id: str) -> WorkflowRun:
        """Fetch workflow run by ID and verify workspace isolation."""
        run = await self._state_store.get_run(run_id)
        if run is None:
            raise EntityNotFoundError(f"Workflow run '{run_id}' not found")
        if run.workspace_id != workspace_id:
            raise WorkspaceIsolationError(
                f"Workflow run '{run_id}' does not belong to workspace '{workspace_id}'"
            )
        return run

    async def list_workflow_runs(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowRun]:
        """List workflow runs in workspace."""
        if self._workspace_repo is not None:
            ws = await self._workspace_repo.get_by_id(workspace_id)
            if ws is None:
                raise EntityNotFoundError(f"Workspace '{workspace_id}' not found")

        return await self._state_store.list_runs(
            workspace_id=workspace_id, company_id=company_id, limit=limit
        )

    async def resume_workflow_run(
        self,
        workspace_id: str,
        run_id: str,
        budget: ExecutionBudget | None = None,
    ) -> WorkflowRun:
        """Resume a failed or paused workflow run from its latest valid checkpoint."""
        run = await self.get_workflow_run(workspace_id, run_id)
        if run.status not in (WorkflowRunStatus.FAILED, WorkflowRunStatus.PAUSED):
            raise ValidationError(
                f"Cannot resume workflow run with status '{run.status.value}'. Must be 'failed' or 'paused'."
            )

        async def _resume_in_background() -> None:
            try:
                await self._orchestrator.resume_workflow(run_id=run_id, budget=budget)
            except Exception as err:
                logger.error("Background workflow resume failed", run_id=run_id, error=str(err))

        self._task_dispatcher(_resume_in_background())
        return run

    async def cancel_workflow_run(self, workspace_id: str, run_id: str) -> bool:
        """Cancel an in-flight workflow run."""
        # Verify workspace ownership
        await self.get_workflow_run(workspace_id, run_id)
        return await self._orchestrator.cancel_workflow(run_id)
