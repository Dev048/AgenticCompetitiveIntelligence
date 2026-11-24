from typing import Any
from fastapi import APIRouter, Depends, Query, Response, status

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_workflow_service,
)
from competitive_intelligence.application.api.v1.schemas.workflow import (
    StageExecutionRecordResponse,
    WorkflowRunCreateRequest,
    WorkflowRunDetailResponse,
    WorkflowRunSummaryResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.workflow_service import (
    WorkflowExecutionService,
)
from competitive_intelligence.decision_support.models import DecisionContext
from competitive_intelligence.orchestration.models import (
    WorkflowInputs,
    WorkflowRun,
)

router = APIRouter(prefix="/workspaces/{workspace_id}/workflows", tags=["Workflows"])


def _to_run_detail(run: WorkflowRun) -> WorkflowRunDetailResponse:
    stage_recs = {
        stype.value: StageExecutionRecordResponse(
            stage=stype.value,
            status=rec.status.value,
            started_at=rec.started_at,
            completed_at=rec.completed_at,
            duration_ms=rec.duration_ms,
            attempt_count=rec.attempt_count,
            output_ids=rec.output_ids,
            error=rec.error.model_dump(mode="json") if rec.error else None,
        )
        for stype, rec in run.stage_records.items()
    }
    return WorkflowRunDetailResponse(
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        company_id=run.company_id,
        workflow_id=run.workflow_id,
        trigger_type=run.trigger_type.value,
        status=run.status.value,
        current_stage=run.current_stage.value if run.current_stage else None,
        target_stop_stage=run.target_stop_stage.value,
        stage_records=stage_recs,
        produced_outputs=run.produced_outputs.model_dump(mode="json"),
        errors=[e.model_dump(mode="json") for e in run.errors],
        started_at=run.started_at,
        completed_at=run.completed_at,
        last_checkpointed_at=run.last_checkpointed_at,
        correlation_id=run.correlation_id,
        metadata=run.metadata,
    )


@router.post("/runs", response_model=WorkflowRunSummaryResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_workflow_run(
    workspace_id: str,
    payload: WorkflowRunCreateRequest,
    response: Response,
    _ws: Workspace = Depends(get_current_workspace),
    service: WorkflowExecutionService = Depends(get_workflow_service),
) -> WorkflowRunSummaryResponse:
    """Submit an operational workflow execution run.

    Dispatches background orchestration and immediately returns HTTP 202 Accepted
    with Location header to poll execution progress.
    """
    dec_ctx: DecisionContext | None = None
    if payload.decision_context:
        dec_ctx = DecisionContext.model_validate(payload.decision_context)

    inputs = WorkflowInputs(
        workspace_id=workspace_id,
        company_id=payload.company_id,
        raw_document_content=payload.raw_document_content,
        source_url=payload.source_url,
        source_id=payload.source_id,
        document_id=payload.document_id,
        query_text=payload.query_text,
        decision_context=dec_ctx,
        enable_decision_support=payload.enable_decision_support,
        metadata=payload.metadata,
    )

    run = await service.submit_workflow_run(
        workspace_id=workspace_id,
        inputs=inputs,
        target_stop_stage=payload.target_stop_stage,
        trigger_type=payload.trigger_type,
    )

    location_url = f"/api/v1/workspaces/{workspace_id}/workflows/runs/{run.run_id}"
    response.headers["Location"] = location_url

    return WorkflowRunSummaryResponse(
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        company_id=run.company_id,
        status=run.status.value,
        target_stop_stage=run.target_stop_stage.value,
        started_at=run.started_at,
        location=location_url,
    )


@router.get("/runs", response_model=list[WorkflowRunDetailResponse])
async def list_workflow_runs(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    limit: int = Query(default=50, ge=1, le=100),
    _ws: Workspace = Depends(get_current_workspace),
    service: WorkflowExecutionService = Depends(get_workflow_service),
) -> list[WorkflowRunDetailResponse]:
    """List workflow execution runs in workspace."""
    runs = await service.list_workflow_runs(
        workspace_id=workspace_id, company_id=company_id, limit=limit
    )
    return [_to_run_detail(r) for r in runs]


@router.get("/runs/{run_id}", response_model=WorkflowRunDetailResponse)
async def get_workflow_run(
    workspace_id: str,
    run_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: WorkflowExecutionService = Depends(get_workflow_service),
) -> WorkflowRunDetailResponse:
    """Get full workflow execution state, stage audit telemetry, and output IDs."""
    run = await service.get_workflow_run(workspace_id=workspace_id, run_id=run_id)
    return _to_run_detail(run)


@router.post("/runs/{run_id}/resume", response_model=WorkflowRunSummaryResponse, status_code=status.HTTP_202_ACCEPTED)
async def resume_workflow_run(
    workspace_id: str,
    run_id: str,
    response: Response,
    _ws: Workspace = Depends(get_current_workspace),
    service: WorkflowExecutionService = Depends(get_workflow_service),
) -> WorkflowRunSummaryResponse:
    """Resume a failed or paused workflow run from its last valid checkpoint."""
    run = await service.resume_workflow_run(workspace_id=workspace_id, run_id=run_id)
    location_url = f"/api/v1/workspaces/{workspace_id}/workflows/runs/{run.run_id}"
    response.headers["Location"] = location_url

    return WorkflowRunSummaryResponse(
        run_id=run.run_id,
        workspace_id=run.workspace_id,
        company_id=run.company_id,
        status="resuming",
        target_stop_stage=run.target_stop_stage.value,
        started_at=run.started_at,
        location=location_url,
    )


@router.post("/runs/{run_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_workflow_run(
    workspace_id: str,
    run_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: WorkflowExecutionService = Depends(get_workflow_service),
) -> dict[str, Any]:
    """Cancel an in-flight workflow run."""
    success = await service.cancel_workflow_run(workspace_id=workspace_id, run_id=run_id)
    return {"run_id": run_id, "cancelled": success}
