"""Prometheus-compatible operational metrics instrumentation.

Provides low-cardinality counters, histograms, and gauges for:
- HTTP requests and latencies.
- Workflow execution and stage transitions.
- Source acquisition and continuous monitoring jobs.
- LLM invocations, token consumption, and budget enforcement.
- Dense, sparse, and adaptive retrieval performance.
"""

import re
import time
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

# Dedicated registry for application metrics
APP_REGISTRY = CollectorRegistry(auto_describe=True)

# ------------------------------------------------------------------------------
# 1. HTTP Metrics
# ------------------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests processed by endpoint and status.",
    ["method", "endpoint", "status_code"],
    registry=APP_REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=APP_REGISTRY,
)

# ------------------------------------------------------------------------------
# 2. Workflow & Stage Execution Metrics
# ------------------------------------------------------------------------------
WORKFLOW_RUNS_TOTAL = Counter(
    "workflow_runs_total",
    "Total workflow executions partitioned by final status and trigger type.",
    ["status", "trigger_type"],
    registry=APP_REGISTRY,
)

WORKFLOW_RUN_DURATION_SECONDS = Histogram(
    "workflow_run_duration_seconds",
    "Total workflow execution duration in seconds.",
    ["status"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
    registry=APP_REGISTRY,
)

WORKFLOW_STAGE_DURATION_SECONDS = Histogram(
    "workflow_stage_duration_seconds",
    "Duration of individual pipeline stages in seconds.",
    ["stage", "status"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=APP_REGISTRY,
)

WORKFLOW_STAGE_RETRIES_TOTAL = Counter(
    "workflow_stage_retries_total",
    "Count of stage retry attempts caused by transient errors.",
    ["stage"],
    registry=APP_REGISTRY,
)

WORKFLOW_FAILURES_TOTAL = Counter(
    "workflow_failures_total",
    "Total stage failures categorized by failure mode.",
    ["stage", "error_category"],
    registry=APP_REGISTRY,
)

# ------------------------------------------------------------------------------
# 3. Source Monitoring & Acquisition Metrics
# ------------------------------------------------------------------------------
MONITORING_JOBS_TOTAL = Counter(
    "monitoring_jobs_total",
    "Count of monitoring polling cycles and single-source jobs.",
    ["status"],
    registry=APP_REGISTRY,
)

MONITORING_JOB_DURATION_SECONDS = Histogram(
    "monitoring_job_duration_seconds",
    "Execution duration for monitoring ingestion runs in seconds.",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=APP_REGISTRY,
)

SOURCE_ACQUISITION_FAILURES_TOTAL = Counter(
    "source_acquisition_failures_total",
    "Total source acquisition failures classified by platform, category, and reason.",
    ["platform", "category", "failure_reason"],
    registry=APP_REGISTRY,
)

MONITORING_ACTIVE_LEASES = Gauge(
    "monitoring_active_leases",
    "Current count of active monitoring leases held across sources.",
    registry=APP_REGISTRY,
)

# ------------------------------------------------------------------------------
# 4. LLM Telemetry Metrics
# ------------------------------------------------------------------------------
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM generation calls by provider, model, and outcome status.",
    ["provider", "model", "status"],
    registry=APP_REGISTRY,
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM call round-trip latency in seconds.",
    ["provider", "model"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=APP_REGISTRY,
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM invocations.",
    ["provider", "model", "token_type"],
    registry=APP_REGISTRY,
)

LLM_BUDGET_EXHAUSTION_TOTAL = Counter(
    "llm_budget_exhaustion_total",
    "Count of tasks hitting or exceeding token/call budgets.",
    ["provider"],
    registry=APP_REGISTRY,
)

LLM_RETRIES_TOTAL = Counter(
    "llm_retries_total",
    "Total retried LLM calls due to rate limits or transient network errors.",
    ["provider", "model"],
    registry=APP_REGISTRY,
)

# ------------------------------------------------------------------------------
# 5. Retrieval Observability Metrics
# ------------------------------------------------------------------------------
RETRIEVAL_REQUESTS_TOTAL = Counter(
    "retrieval_requests_total",
    "Total retrieval operations by route, mode, and execution status.",
    ["route", "mode", "status"],
    registry=APP_REGISTRY,
)

RETRIEVAL_DURATION_SECONDS = Histogram(
    "retrieval_duration_seconds",
    "Retrieval execution latency in seconds.",
    ["route", "mode"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=APP_REGISTRY,
)

RETRIEVAL_CHUNKS_RETURNED = Histogram(
    "retrieval_chunks_returned",
    "Distribution of retrieved chunk candidate counts.",
    ["route"],
    buckets=(0, 1, 3, 5, 10, 20, 50, 100),
    registry=APP_REGISTRY,
)

RETRIEVAL_RERANKED_TOTAL = Counter(
    "retrieval_reranked_total",
    "Total retrieval operations that executed neural cross-encoder reranking.",
    registry=APP_REGISTRY,
)

RETRIEVAL_ESCALATED_TOTAL = Counter(
    "retrieval_escalated_total",
    "Total retrieval queries escalated to selective agentic multi-hop execution.",
    registry=APP_REGISTRY,
)

# ------------------------------------------------------------------------------
# Helper Utilities (Safe, isolated, non-throwing)
# ------------------------------------------------------------------------------

# Regex patterns for normalizing dynamic UUIDs and slugs to bound metric cardinality
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
CUSTOM_ID_PATTERN = re.compile(
    r"/(workspaces|sources|competitors|documents|runs|contexts|recommendations|insights|events)/[a-zA-Z0-9_\-]+",
    re.IGNORECASE,
)


def normalize_endpoint(path: str) -> str:
    """Normalize dynamic entity IDs in URL paths to prevent unbounded metric cardinality."""
    if not path:
        return "/"
    normalized = UUID_PATTERN.sub("{id}", path)
    normalized = CUSTOM_ID_PATTERN.sub(r"/\1/{id}", normalized)
    return normalized


def track_http_request(method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Safely record HTTP request metrics."""
    try:
        endpoint = normalize_endpoint(path)
        HTTP_REQUESTS_TOTAL.labels(
            method=method.upper(),
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method.upper(),
            endpoint=endpoint,
        ).observe(max(0.0, duration_seconds))
    except Exception as exc:
        logger.debug("Metrics recording notice (HTTP)", error=str(exc))


def track_workflow_run(status: str, trigger_type: str, duration_seconds: float) -> None:
    """Safely record workflow execution conclusion metrics."""
    try:
        norm_status = status.lower()
        norm_trigger = trigger_type.lower()
        WORKFLOW_RUNS_TOTAL.labels(status=norm_status, trigger_type=norm_trigger).inc()
        WORKFLOW_RUN_DURATION_SECONDS.labels(status=norm_status).observe(max(0.0, duration_seconds))
    except Exception as exc:
        logger.debug("Metrics recording notice (Workflow)", error=str(exc))


def track_stage_execution(
    stage: str,
    status: str,
    duration_seconds: float,
    error_category: str | None = None,
    retries: int = 0,
) -> None:
    """Safely record stage execution telemetry."""
    try:
        norm_stage = stage.lower()
        norm_status = status.lower()
        WORKFLOW_STAGE_DURATION_SECONDS.labels(stage=norm_stage, status=norm_status).observe(
            max(0.0, duration_seconds)
        )
        if retries > 0:
            WORKFLOW_STAGE_RETRIES_TOTAL.labels(stage=norm_stage).inc(retries)
        if norm_status == "failed" and error_category:
            WORKFLOW_FAILURES_TOTAL.labels(
                stage=norm_stage, error_category=error_category.lower()
            ).inc()
    except Exception as exc:
        logger.debug("Metrics recording notice (Stage)", error=str(exc))


def track_monitoring_job(
    status: str,
    duration_seconds: float,
    platform: str | None = None,
    category: str | None = None,
    failure_reason: str | None = None,
) -> None:
    """Safely record continuous monitoring job metrics."""
    try:
        norm_status = status.lower()
        MONITORING_JOBS_TOTAL.labels(status=norm_status).inc()
        MONITORING_JOB_DURATION_SECONDS.observe(max(0.0, duration_seconds))
        if norm_status == "failed":
            SOURCE_ACQUISITION_FAILURES_TOTAL.labels(
                platform=(platform or "unknown").lower(),
                category=(category or "unknown").lower(),
                failure_reason=(failure_reason or "unknown").lower(),
            ).inc()
    except Exception as exc:
        logger.debug("Metrics recording notice (Monitoring)", error=str(exc))


def set_active_monitoring_leases(count: int) -> None:
    """Update active leases gauge."""
    try:
        MONITORING_ACTIVE_LEASES.set(max(0, count))
    except Exception as exc:
        logger.debug("Metrics recording notice (Leases)", error=str(exc))


def track_llm_call(
    provider: str,
    model: str,
    status: str,
    duration_seconds: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    retries: int = 0,
) -> None:
    """Safely record LLM invocation telemetry."""
    try:
        norm_prov = provider.lower()
        norm_model = model.lower()
        norm_status = status.lower()
        LLM_REQUESTS_TOTAL.labels(provider=norm_prov, model=norm_model, status=norm_status).inc()
        LLM_REQUEST_DURATION_SECONDS.labels(provider=norm_prov, model=norm_model).observe(
            max(0.0, duration_seconds)
        )
        if prompt_tokens > 0:
            LLM_TOKENS_TOTAL.labels(provider=norm_prov, model=norm_model, token_type="prompt").inc(
                prompt_tokens
            )
        if completion_tokens > 0:
            LLM_TOKENS_TOTAL.labels(
                provider=norm_prov, model=norm_model, token_type="completion"
            ).inc(completion_tokens)
        if retries > 0:
            LLM_RETRIES_TOTAL.labels(provider=norm_prov, model=norm_model).inc(retries)
    except Exception as exc:
        logger.debug("Metrics recording notice (LLM)", error=str(exc))


def track_llm_budget_exhaustion(provider: str) -> None:
    """Record task-level token or call budget exhaustion."""
    try:
        LLM_BUDGET_EXHAUSTION_TOTAL.labels(provider=provider.lower()).inc()
    except Exception as exc:
        logger.debug("Metrics recording notice (Budget)", error=str(exc))


def track_retrieval(
    route: str,
    mode: str,
    status: str,
    duration_seconds: float,
    chunk_count: int = 0,
    reranked: bool = False,
    escalated: bool = False,
) -> None:
    """Safely record vector / hybrid / adaptive retrieval telemetry."""
    try:
        norm_route = route.lower()
        norm_mode = mode.lower()
        norm_status = status.lower()
        RETRIEVAL_REQUESTS_TOTAL.labels(
            route=norm_route, mode=norm_mode, status=norm_status
        ).inc()
        RETRIEVAL_DURATION_SECONDS.labels(route=norm_route, mode=norm_mode).observe(
            max(0.0, duration_seconds)
        )
        RETRIEVAL_CHUNKS_RETURNED.labels(route=norm_route).observe(chunk_count)
        if reranked:
            RETRIEVAL_RERANKED_TOTAL.inc()
        if escalated:
            RETRIEVAL_ESCALATED_TOTAL.inc()
    except Exception as exc:
        logger.debug("Metrics recording notice (Retrieval)", error=str(exc))


def generate_prometheus_metrics() -> bytes:
    """Generate serialized OpenMetrics / Prometheus exposition text."""
    return generate_latest(APP_REGISTRY)


def get_metrics_content_type() -> str:
    """Return standard Prometheus media type string."""
    return CONTENT_TYPE_LATEST
