from typing import Any
from fastapi import APIRouter, Depends, Query, Response, status

from competitive_intelligence.application.api.deps import (
    get_current_workspace,
    get_monitoring_lease_repo,
    get_source_service,
    get_workflow_service,
)
from competitive_intelligence.application.api.v1.schemas.source import (
    MonitoredSourceDiagnostic,
    MonitoringDiagnosticsResponse,
    SourceCreateRequest,
    SourceResponse,
    SourceTriggerRequest,
    SourceUpdateRequest,
)
from competitive_intelligence.application.api.v1.schemas.workflow import (
    WorkflowRunSummaryResponse,
)
from competitive_intelligence.application.models import Workspace
from competitive_intelligence.application.services.source_service import SourceService
from competitive_intelligence.application.services.workflow_service import (
    WorkflowExecutionService,
)
from competitive_intelligence.core.config import Settings, get_settings
from competitive_intelligence.core.errors import ValidationError
from competitive_intelligence.monitoring.interfaces import (
    BaseMonitoringLeaseRepository,
)
from competitive_intelligence.orchestration.models import (
    TriggerType,
    WorkflowInputs,
    WorkflowStageType,
)
from competitive_intelligence.source_acquisition.models import Source, SourceCategory

router = APIRouter(prefix="/workspaces/{workspace_id}/sources", tags=["Sources"])


def _build_source_response(s: Source) -> SourceResponse:
    return SourceResponse(
        source_id=s.identity.source_id,
        workspace_id=s.identity.workspace_id,
        company_id=s.identity.company_id,
        display_name=s.identity.display_name,
        category=s.classification.category.value,
        platform=s.classification.platform.value,
        content_type=s.classification.content_type.value,
        url_or_identifier=s.location.url_or_identifier,
        access_pattern=s.location.access_pattern.value,
        monitoring_enabled=s.monitoring.enabled,
        frequency_seconds=s.monitoring.frequency,
        priority=s.monitoring.priority,
        last_fetch_status=s.acquisition.last_fetch_status.value,
        consecutive_failure_count=s.acquisition.consecutive_failure_count,
        last_checked_at=s.monitoring.last_checked_at,
        created_at=s.lifecycle.created_at,
    )



@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def register_source(
    workspace_id: str,
    payload: SourceCreateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Register a new source under a competitor."""
    src = await service.register_source(
        workspace_id=workspace_id,
        company_id=payload.company_id,
        display_name=payload.display_name,
        url_or_identifier=payload.url_or_identifier,
        category=payload.category,
    )
    return _build_source_response(src)


@router.get("", response_model=list[SourceResponse])
async def list_sources(
    workspace_id: str,
    company_id: str | None = Query(default=None, description="Filter by competitor company ID"),
    category: SourceCategory | None = Query(default=None, description="Filter by category"),
    _ws: Workspace = Depends(get_current_workspace),
    service: SourceService = Depends(get_source_service),
) -> list[SourceResponse]:
    """List sources in workspace."""
    sources = await service.list_sources(
        workspace_id=workspace_id, company_id=company_id, category=category
    )
    return [_build_source_response(s) for s in sources]


@router.get("/monitoring/diagnostics", response_model=MonitoringDiagnosticsResponse)
async def get_monitoring_diagnostics(
    workspace_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    source_service: SourceService = Depends(get_source_service),
    lease_repo: BaseMonitoringLeaseRepository = Depends(get_monitoring_lease_repo),
    settings: Settings = Depends(get_settings),
) -> MonitoringDiagnosticsResponse:
    """Retrieve operational diagnostics for continuous source monitoring."""
    sources = await source_service.list_sources(workspace_id=workspace_id)

    healthy = 0
    failing = 0
    disabled = 0
    active_leases_count = 0
    diagnostic_sources: list[MonitoredSourceDiagnostic] = []

    for s in sources:
        if not s.monitoring.enabled:
            disabled += 1
        elif s.acquisition.consecutive_failure_count > 0 or s.acquisition.last_fetch_status.value == "failure":
            failing += 1
        else:
            healthy += 1

        active_lease_info: dict[str, Any] | None = None
        try:
            lease = await lease_repo.get_active_lease(s.identity.source_id)
            if lease and not lease.is_expired:
                active_leases_count += 1
                active_lease_info = {
                    "lease_owner": lease.lease_owner,
                    "lease_acquired_at": lease.lease_acquired_at.isoformat(),
                    "lease_expires_at": lease.lease_expires_at.isoformat(),
                    "attempt_count": lease.attempt_count,
                }
        except Exception:
            pass

        diagnostic_sources.append(
            MonitoredSourceDiagnostic(
                source_id=s.identity.source_id,
                display_name=s.identity.display_name,
                company_id=s.identity.company_id,
                category=s.classification.category.value,
                platform=s.classification.platform.value,
                url_or_identifier=s.location.url_or_identifier,
                monitoring_enabled=s.monitoring.enabled,
                frequency_seconds=s.monitoring.frequency,
                last_fetch_status=s.acquisition.last_fetch_status.value,
                consecutive_failure_count=s.acquisition.consecutive_failure_count,
                last_checked_at=s.monitoring.last_checked_at,
                active_lease=active_lease_info,
            )
        )


    return MonitoringDiagnosticsResponse(
        workspace_id=workspace_id,
        total_sources=len(sources),
        healthy_sources_count=healthy,
        failing_sources_count=failing,
        disabled_sources_count=disabled,
        active_leases_count=active_leases_count,
        scheduler_mode=settings.monitoring_mode,
        scheduler_running=settings.monitoring_enabled,
        sources=diagnostic_sources,
    )


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    workspace_id: str,
    source_id: str,
    _ws: Workspace = Depends(get_current_workspace),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Get source configuration & lifecycle details."""
    s = await service.get_source(workspace_id=workspace_id, source_id=source_id)
    return _build_source_response(s)


@router.patch("/{source_id}", response_model=SourceResponse)
async def update_source(
    workspace_id: str,
    source_id: str,
    payload: SourceUpdateRequest,
    _ws: Workspace = Depends(get_current_workspace),
    service: SourceService = Depends(get_source_service),
) -> SourceResponse:
    """Update source monitoring policy."""
    s = await service.update_source_monitoring(
        workspace_id=workspace_id,
        source_id=source_id,
        monitoring_enabled=payload.monitoring_enabled,
        frequency_seconds=payload.frequency_seconds,
        priority=payload.priority,
    )
    return _build_source_response(s)



@router.post(
    "/{source_id}/trigger",
    response_model=WorkflowRunSummaryResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_source_workflow(
    workspace_id: str,
    source_id: str,
    payload: SourceTriggerRequest,
    response: Response,
    _ws: Workspace = Depends(get_current_workspace),
    source_service: SourceService = Depends(get_source_service),
    workflow_service: WorkflowExecutionService = Depends(get_workflow_service),
) -> WorkflowRunSummaryResponse:
    """Trigger an on-demand acquisition and intelligence workflow run for a source."""
    source = await source_service.get_source(workspace_id=workspace_id, source_id=source_id)

    try:
        stop_stage = WorkflowStageType(payload.target_stop_stage)
    except ValueError as err:
        raise ValidationError(
            f"Invalid target_stop_stage '{payload.target_stop_stage}'. Allowed: {[s.value for s in WorkflowStageType]}"
        ) from err

    doc_content = (
        payload.raw_document_content
        or f"On-demand intelligence acquisition for source '{source.identity.display_name}' ({source.location.url_or_identifier}). Category: {source.classification.category.value}."
    )

    inputs = WorkflowInputs(
        workspace_id=workspace_id,
        company_id=source.identity.company_id,
        source_url=source.location.url_or_identifier,
        raw_document_content=doc_content,
        metadata={
            "source_id": source_id,
            "source_category": source.classification.category.value,
            "trigger": "manual_on_demand",
        },
    )

    run = await workflow_service.submit_workflow_run(
        workspace_id=workspace_id,
        inputs=inputs,
        target_stop_stage=stop_stage,
        trigger_type=TriggerType.MANUAL_USER_REQUEST,
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
