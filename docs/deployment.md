# Deployment and Operations Guide

This guide describes how to configure, deploy, run, and maintain the **Competitive Intelligence Monitoring Platform** in production and local environments.

---

## 1. Architectural Topology

The deployment architecture consists of four primary services orchestrated via Docker Compose:

```
                                  [ Browser / Client ]
                                            │
                                            ▼
                      ┌──────────────────────────────────────────┐
                      │                 ci-api                   │
                      │          (FastAPI + Static SPA)          │
                      │               Port 8000                  │
                      └──────────────┬───────────┬───────────────┘
                                     │           │
                     Database Queries│           │ Vector Queries
                     (JSONB, Leases) │           │ (1024-dim BGE-M3)
                                     ▼           ▼
                      ┌──────────────────┐   ┌──────────────────┐
                      │   ci-postgres    │   │    ci-qdrant     │
                      │  PostgreSQL 16   │   │  Qdrant v1.12.1  │
                      │    Port 5432     │   │  Ports 6333/6334 │
                      │  [postgres_data] │   │  [qdrant_data]   │
                      └──────────────────┘   └──────────────────┘
                                     ▲           ▲
                      Database Leases│           │
                                     │           │
                      ┌──────────────┴───────────┴───────────────┐
                      │                ci-worker                 │
                      │     (Continuous Monitoring Scheduler)    │
                      └──────────────────────────────────────────┘
```

1. **`ci-postgres`**: PostgreSQL 16 database storing 25 relational tables, JSONB evidence payloads, user/workspace boundaries, and atomic monitoring leases. Data persists in volume `ci_postgres_data`.
2. **`ci-qdrant`**: Qdrant vector database managing 1024-dimensional dense chunk embeddings with cosine similarity. Data persists in volume `ci_qdrant_data`.
3. **`ci-api`**: FastAPI application serving both the REST APIs (`/api/v1`) and the Single Page Application UI (`/`). Automatically validates production configuration and waits for database readiness before startup.
4. **`ci-worker`**: Autonomous background service executing continuous monitoring scheduling, due source claiming, heartbeat lease renewal, and ingestion dispatch.

---

## 2. Environment Configuration

### Step 1: Initialize Environment File
Copy the documented template:
```bash
cp .env.example .env
```

### Step 2: Configure Production Variables
Edit `.env` with your deployment credentials and settings:

```dotenv
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json

# Persistence & Storage
DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/competitive_intelligence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_postgres_password
POSTGRES_DB=competitive_intelligence

# Vector Store
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=ci_knowledge_chunks

# Embeddings
EMBEDDING_PROVIDER=bge_m3
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

# LLM Reasoning
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key
DEFAULT_LLM_MODEL=gpt-4o-mini
REASONING_LLM_MODEL=gpt-4o

# Scheduler & Monitoring
MONITORING_ENABLED=true
MONITORING_MODE=in_process
```

> [!IMPORTANT]
> When `ENVIRONMENT=production`, the application strictly enforces:
> - `DEBUG=false`
> - `LLM_PROVIDER` cannot be `mock` (must be `openai` or `anthropic` with valid key)
> - `EMBEDDING_PROVIDER` cannot be `mock` (must be `bge_m3`)
> - `DATABASE_URL` cannot be SQLite or in-memory
> - `QDRANT_LOCATION` cannot be `:memory:`

---

## 3. Deployment Commands

### Build Images
Build the production application image:
```bash
docker compose build
```

### Start Services
Start all infrastructure and application services in detached mode:
```bash
docker compose up -d
```

### Verify Service Health
Check container status and healthchecks:
```bash
docker compose ps
```

Expected output:
```
NAME          IMAGE                           COMMAND                  SERVICE      STATUS
ci-postgres   postgres:16-alpine              "docker-entrypoint.s…"   postgres     Up (healthy)
ci-qdrant     qdrant/qdrant:v1.12.1           "./entrypoint.sh"        qdrant       Up (healthy)
ci-api        competitive-intelligence:latest "/app/scripts/docker…"   api          Up (healthy)
ci-worker     competitive-intelligence:latest "/app/scripts/docker…"   worker       Up (running)
```

### Inspect Container Logs
Follow live logs across all services:
```bash
docker compose logs -f api
```

### Stop Services
Stop containers while safely preserving all persisted PostgreSQL data and Qdrant vector indexes:
```bash
docker compose down
```

> [!CAUTION]
> Do NOT use `docker compose down -v` in production unless you intentionally want to delete all database rows and vector embeddings.

---

## 4. Database Migrations

Database migrations are executed automatically during the container startup sequence via `scripts/docker-entrypoint.sh`.

### Manual Migration Commands
To inspect or run migrations manually:

```bash
# Check current revision status
docker compose exec api python -m alembic current

# Upgrade to latest schema revision
docker compose exec api python -m alembic upgrade head

# Generate offline SQL without applying
docker compose exec api python -m alembic upgrade head --sql
```

---

## 5. Health Probes and Monitoring

The API exposes three distinct health probes:

| Probe | Endpoint | Status Code | Purpose |
| :--- | :--- | :--- | :--- |
| **Liveness** | `GET /health/live` | `200` | Confirms the HTTP server process is running. |
| **Readiness** | `GET /health/ready` | `200` (Ready) / `503` (Degraded) | Verifies PostgreSQL and Qdrant infrastructure connectivity before routing traffic. |
| **Status** | `GET /health` | `200` | Returns JSON status envelope with database and vector store connectivity. |

### Probing with cURL:
```bash
# Check liveness
curl -f http://localhost:8000/health/live

# Check readiness
curl -f http://localhost:8000/health/ready

# Check complete telemetry
curl http://localhost:8000/health
```

Sample Readiness Output:
```json
{
  "status": "ready",
  "database": "connected",
  "qdrant": "connected",
  "environment": "production",
  "ready": true
}
```

---

## 6. Backup and Persistence Management

### PostgreSQL Backup
```bash
docker compose exec -T postgres pg_dump -U postgres -d competitive_intelligence > backup_$(date +%Y%m%d_%H%M%S).sql
```

### PostgreSQL Restore
```bash
docker compose exec -T postgres psql -U postgres -d competitive_intelligence < backup_file.sql
```

### Qdrant Vector Snapshots
Qdrant supports snapshotting vector collections directly via API:
```bash
curl -X POST "http://localhost:6333/collections/ci_knowledge_chunks/snapshots"
```

---

## 7. Known Deployment Limitations

1. **In-Process Scheduler Concurrency**:
   - The monitoring scheduler uses PostgreSQL row leases (`source_monitoring_leases`) for distributed safety.
   - It runs as either an in-process asyncio task inside `api` or as a standalone process inside `worker`.
   - External distributed job schedulers (Celery, Kafka, Redis queues) are intentionally omitted in accordance with architectural invariants.
2. **Local Embedding Compute**:
   - `BGEM3EmbeddingProvider` runs on CPU by default inside the container.
   - For high-throughput production ingestion, set `EMBEDDING_DEVICE=cuda` and mount NVIDIA GPU drivers via Docker compose device reservations.
3. **LLM Network Access**:
   - The container must have outbound HTTPS network connectivity to reach OpenAI (`api.openai.com`) or Anthropic (`api.anthropic.com`).
