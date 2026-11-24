# Observability and Operational Reliability Guide

This document details the observability architecture, telemetry models, metrics, logging invariants, and operational troubleshooting procedures for the **Competitive Intelligence Monitoring Platform**.

---

## 1. Observability Architecture Overview

The platform implements a production-first, lightweight, and zero-external-broker observability stack designed to give operators complete visibility into system health, workflow execution, continuous source monitoring, LLM resource consumption, and retrieval accuracy without impacting pipeline latency or reliability.

```
                           [ Prometheus / VictoriaMetrics ]
                                          ▲
                                          │ Scrapes GET /metrics
                                          │
 ┌────────────────────────────────────────┴────────────────────────────────────────┐
 │                                ci-api / ci-worker                               │
 │                                                                                 │
 │   ┌───────────────────────┐ ┌───────────────────────┐ ┌─────────────────────┐   │
 │   │  Contextual Logging   │ │  Prometheus Registry  │ │  Telemetry Store    │   │
 │   │  - JSON / Console     │ │  - Bounded Counters   │ │  - WorkflowRun      │   │
 │   │  - structlog + vars   │ │  - Latency Histograms │ │  - StageExecution   │   │
 │   │  - Secret Redaction   │ │  - Lease Gauges       │ │  - Diagnostics API  │   │
 │   └──────────┬────────────┘ └──────────┬────────────┘ └──────────┬──────────┘   │
 └──────────────┼─────────────────────────┼─────────────────────────┼──────────────┘
                ▼                         ▼                         ▼
      [ FluentBit / Loki ]      [ Prometheus Scraper ]    [ PostgreSQL / UI SPA ]
      (JSON stdout logs)         (OpenMetrics text)       (Relational audit trail)
```

### Core Design Invariants:
1. **Fail-Safe Telemetry**: Observability instrumentation failures (e.g. metric recording errors or log formatting errors) are caught internally and **never bubble up or abort business transactions**.
2. **Cardinality Safety**: Metric labels are strictly bounded to small enums (HTTP methods, status categories, workflow stage names, providers, routes). Unbounded fields such as `workspace_id`, `document_id`, `source_url`, or `correlation_id` are **strictly forbidden** in Prometheus labels.
3. **Sensitive Data Protection**: Prompts, API keys, bearer tokens, passwords, and raw document contents are stripped, redacted with `[REDACTED]`, or truncated (`... [TRUNCATED]`) before logging.
4. **End-to-End Correlation**: Every inbound request or background task receives a `correlation_id` (via HTTP header `X-Correlation-ID` or generated UUID) propagated across all log lines, workflow runs, stage executions, and audit records.

---

## 2. Structured Logging

Logging is implemented using `structlog` configured in [`src/competitive_intelligence/core/logger.py`](file:///C:/programming/agentic_ai/competitive_intelligence/src/competitive_intelligence/core/logger.py).

### Log Formats
- **Production (`ENVIRONMENT=production` or `LOG_FORMAT=json`)**: Standardized single-line JSON output to `stdout` compatible with cloud log aggregators (Datadog, Loki, AWS CloudWatch, Google Cloud Logging).
- **Development (`ENVIRONMENT=development` and `LOG_FORMAT=console`)**: Colorized, human-readable terminal output.

### Standard Context Fields
Every structured log record contains the following standardized fields when available:

| Field | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | ISO-8601 string | UTC timestamp of the log event (`2026-11-02T13:45:00.123456Z`) |
| `level` | string | Severity level: `debug`, `info`, `warning`, `error`, `critical` |
| `logger` | string | Originating module or subsystem name |
| `event` | string | Primary log message or event name |
| `correlation_id` | string (UUID) | Distributed trace/request identifier across HTTP calls and stages |
| `workspace_id` | string (UUID) | Tenant workspace boundary |
| `workflow_run_id` | string (UUID) | Active workflow run instance (if inside a workflow execution) |
| `stage_id` | string | Currently executing workflow stage name |
| `source_id` | string (UUID) | Target source ID (for acquisition/monitoring events) |
| `exc_info` | string / dict | Formatted exception trace if an error occurred |

### Context Propagation
The logging subsystem uses Python `contextvars` via `bind_log_context()` and `clear_log_context()` to ensure context persists across `async`/`await` boundaries without explicit manual parameter passing:

```python
from competitive_intelligence.core.logger import bind_log_context, clear_log_context

# Automatically bound by HTTP middleware & workflow orchestrator
bind_log_context(
    correlation_id="corr-987123",
    workspace_id="ws-456",
    workflow_run_id="run-789",
    stage_id="synthesis_stage",
)
```

### Sensitive Data Redaction
All log processors pass through `redact_sensitive_processor`:
- Keys matching `*api_key*`, `*token*`, `*secret*`, `*password*`, `*authorization*` have their values masked as `[REDACTED]`.
- String representations matching regex patterns for JWT tokens, `sk-` / `ghp_` API keys, and Bearer authorization tokens are redacted.
- Document and prompt content fields longer than 500 characters are automatically truncated with `... [TRUNCATED]` to prevent log bloat and data leakage.

---

## 3. Metrics Endpoint and Catalog

The application exposes a lightweight Prometheus/OpenMetrics text exposition endpoint at `GET /metrics`.

### Endpoint Configuration
- **Path**: `/metrics`
- **Port**: `8000` (configurable via `APP_PORT`)
- **Content-Type**: `text/plain; version=0.0.4; charset=utf-8`
- **Protection**: If `METRICS_TOKEN` is set in the environment, scrapers must provide the header:
  ```http
  Authorization: Bearer <METRICS_TOKEN>
  ```
  If unset, `/metrics` is accessible without authentication inside the internal network.

### Metric Catalog

#### HTTP Server Metrics
| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `http_requests_total` | Counter | `method`, `endpoint`, `status` | Total HTTP requests processed (paths normalized to prevent cardinality leaks) |
| `http_request_duration_seconds` | Histogram | `method`, `endpoint` | End-to-end HTTP request processing latency |

#### Workflow & Stage Execution Metrics
| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `workflow_runs_total` | Counter | `workflow_name`, `status` | Total workflow runs initiated and completed |
| `workflow_run_duration_seconds` | Histogram | `workflow_name`, `status` | End-to-end workflow execution latency |
| `workflow_stage_duration_seconds` | Histogram | `workflow_name`, `stage_name`, `status` | Execution latency of individual workflow stages |
| `workflow_stage_retries_total` | Counter | `workflow_name`, `stage_name` | Stage retry attempts due to transient errors |
| `workflow_failures_total` | Counter | `workflow_name`, `stage_name`, `error_type` | Workflow failures classified by stage and error category |

#### Monitoring & Ingestion Metrics
| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `monitoring_jobs_total` | Counter | `status`, `job_type` | Continuous monitoring jobs executed |
| `monitoring_job_duration_seconds` | Histogram | `job_type` | Duration of source check and acquisition |
| `source_acquisition_failures_total` | Counter | `source_type`, `error_category` | Ingestion failures categorized by source type and failure reason |
| `monitoring_active_leases` | Gauge | _none_ | Current number of active source leases held by worker processes |

#### LLM Observability Metrics
| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `llm_requests_total` | Counter | `provider`, `model`, `status` | Total LLM invocations |
| `llm_request_duration_seconds` | Histogram | `provider`, `model` | Latency of LLM API requests |
| `llm_tokens_total` | Counter | `provider`, `model`, `token_type` | Token consumption partitioned by `input` and `output` |
| `llm_budget_exhaustion_total` | Counter | `budget_type` | Occurrences where token, cost, or hop budgets were exhausted |
| `llm_retries_total` | Counter | `provider`, `model` | Retry attempts due to rate limits or API errors |

#### Retrieval & Routing Metrics
| Metric Name | Type | Labels | Description |
| :--- | :--- | :--- | :--- |
| `retrieval_requests_total` | Counter | `route`, `mode`, `status` | Retrieval executions by route (`hybrid`, `semantic`, `lexical`, `agentic`) |
| `retrieval_duration_seconds` | Histogram | `route`, `mode` | Retrieval latency from query dispatch to chunk return |
| `retrieval_chunks_returned` | Histogram | `route` | Number of chunks retrieved per query |
| `retrieval_reranked_total` | Counter | `model` | Total times neural cross-encoder reranking was applied |
| `retrieval_escalated_total` | Counter | _none_ | Total queries escalated to multi-hop agentic retrieval |

---

## 4. Health, Liveness, and Readiness Endpoints

The API provides three standard container lifecycle endpoints:

### 1. `GET /health` (Consolidated Health)
Returns comprehensive system health including database connectivity, vector database status, and system uptime.
- **Healthy Response**: `200 OK`
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "environment": "production",
    "uptime_seconds": 12845.2,
    "database": {"connected": true, "status": "ok"},
    "vector_store": {"connected": true, "status": "ok"}
  }
  ```
- **Degraded Response**: `503 Service Unavailable` if core infrastructure is disconnected.

### 2. `GET /health/live` (Kubernetes / Docker Liveness Probe)
Verifies process responsiveness. Does not touch external databases to prevent cascade restarts during intermittent network blips.
- **Response**: `200 OK`
  ```json
  {"status": "ok", "live": true}
  ```

### 3. `GET /health/ready` (Kubernetes / Docker Readiness Probe)
Verifies database connectivity (`SELECT 1`) and Qdrant readiness before traffic routing.
- **Response**: `200 OK` if dependencies are online; `503 Service Unavailable` if initializing.

---

## 5. Monitoring Diagnostics & Source Troubleshooting

Continuous monitoring status is diagnosable via API and the web UI:

### Diagnostics API
`GET /api/v1/workspaces/{workspace_id}/sources/monitoring/diagnostics`

#### Example Response:
```json
{
  "workspace_id": "ws-123456",
  "active_leases_count": 2,
  "sources_monitored_total": 8,
  "sources_failing_count": 1,
  "scheduler_status": "operational",
  "diagnostics": [
    {
      "source_id": "src-789012",
      "name": "Acme Product Pricing",
      "source_type": "web_page",
      "url": "https://competitor.com/pricing",
      "monitoring_status": "failing",
      "consecutive_failures": 3,
      "last_checked_at": "2026-11-02T13:30:00Z",
      "last_fetch_status": "failed",
      "last_error": "HTTP 403 Forbidden: Cloudflare anti-bot challenged",
      "active_lease": null
    }
  ]
}
```

### Distinguishing Acquisition Failures vs. Downstream Pipeline Errors
The platform strictly separates acquisition failures from workflow execution errors:
- **Source Acquisition Failure**: Recorded in `sources.acquisition.consecutive_failure_count` and increments `source_acquisition_failures_total`. Common causes: HTTP 403/404, DNS failure, timeout.
- **Downstream Pipeline Failure**: If source acquisition succeeds, the crawler creates a `WorkflowRun`. If chunking, embedding, or synthesis fails, the workflow status transitions to `FAILED` and `workflow_failures_total` is incremented with `stage_name="ingestion_stage"`.

---

## 6. Workflow Troubleshooting Procedure

When a workflow fails or produces unexpected results:

### Step 1: Locate the Workflow Run
In the Web UI, navigate to **Workflows**. Filter by status `FAILED`.
Or query via API:
```bash
curl -H "X-Workspace-ID: $WS_ID" http://localhost:8000/api/v1/workflows/?status=failed
```

### Step 2: Extract Correlation ID
Click the **Copy Correlation ID** icon next to the workflow run ID in the UI, or inspect `workflow_run.correlation_id`.

### Step 3: Inspect Stage Execution Records
In the UI, click **Audit Trace** to view the granular breakdown of all stage executions:
- **`stage_name`**: The exact stage that faulted (`ingestion_stage`, `signal_stage`, `synthesis_stage`).
- **`status`**: `failed`
- **`retry_count`**: Number of automated retry attempts before failure.
- **`error_details`**: Sanitized error message and root cause.
- **`telemetry`**: Execution duration and stage-specific metadata.

### Step 4: Search Aggregated Logs
Query your central log store using the extracted `correlation_id`:
```bash
# If using jq on container logs:
docker compose logs ci-api | jq 'select(.correlation_id == "corr-987123")'
```

---

## 7. LLM & Retrieval Diagnostics

### LLM Troubleshooting
- **Token Budget Exhaustion**: If a workflow fails with `BudgetExceededError`, inspect the metric `llm_budget_exhaustion_total{budget_type="tokens"}`. The orchestrator enforces strict per-run token limits to avoid runaways.
- **Provider Fallbacks & Retries**: Check `llm_retries_total{provider="..."}` to diagnose provider-side throttling or rate limits (HTTP 429).
- **Latency Spikes**: Review histogram `llm_request_duration_seconds` bucket distribution to identify slow inference endpoints.

### Retrieval Troubleshooting
- **Route Selection**: Check `retrieval_requests_total{route="..."}`. If queries with specific keywords are misrouted to vector search, examine `AdaptiveRetriever` routing confidence thresholds.
- **Agentic Multi-Hop Escalation**: Monitor `retrieval_escalated_total`. A high escalation rate indicates ambiguous initial queries or low-density evidence chunks requiring iterative hops.
- **Reranker Impact**: Review `retrieval_reranked_total`. If reranking latency is high, ensure GPU acceleration or optimized BGE-M3 cross-encoder batching is active.

---

## 8. Production Operational Recommendations

### 1. Prometheus Scrape Configuration
Add the following job to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'competitive-intelligence'
    scrape_interval: 15s
    scrape_timeout: 5s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['ci-api:8000']
    # If METRICS_TOKEN is configured:
    # authorization:
    #   type: Bearer
    #   credentials: 'your-secure-metrics-token'
```

### 2. Recommended Alerting Rules
```yaml
groups:
  - name: competitive_intelligence_alerts
    rules:
      - alert: HighWorkflowFailureRate
        expr: rate(workflow_failures_total[5m]) / rate(workflow_runs_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Workflow failure rate exceeds 5%"

      - alert: LLMBudgetExhaustion
        expr: increase(llm_budget_exhaustion_total[15m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "Workflows failing due to LLM budget limits"

      - alert: SourceMonitoringLeaseStall
        expr: monitoring_active_leases > 10 and rate(monitoring_jobs_total[10m]) == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Monitoring leases held but no jobs completing"
```

### 3. Log Retention & Shipping
Configure Docker log rotation in `/etc/docker/daemon.json`:
```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5"
  }
}
```
Ship logs using Vector, FluentBit, or Promtail targeting Loki or Elasticsearch.
