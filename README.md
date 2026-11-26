# Competitive Intelligence Monitoring Platform

A production-grade, evidence-grounded strategic competitive intelligence and decision-support platform. It continuously tracks competitor actions across websites, pricing structures, documentation, and platform sources, transforming raw market signals into auditable, prescriptive strategic recommendations subject to strict human operator boundaries.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Type-Check](https://img.shields.io/badge/mypy-strict-blue.svg)](pyproject.toml)

---

## Key Characteristics

- **Production-First Agentic RAG**: Bounded query decomposition, sufficiency validation, and cross-source contradiction detection via deterministic state machine orchestration.
- **Strict 10-Layer Provenance**: Unbroken backward auditability from the final strategic recommendation down to the exact source chunk, document version, and canonical origin URL.
- **Mandatory Human Decision Boundary**: Prescriptive recommendations are generated strictly in the `PROPOSED` status. Autonomous execution of business decisions is architecturally prohibited; changes require explicit operator review (`ACCEPTED` or `REJECTED`).
- **Deterministic Over LLM**: Cryptographic hashing (SHA-256), change detection, chunking, deduplication, and score fusion (RRF) are strictly deterministic. LLMs are reserved for structured semantic interpretation and option synthesis.
- **Multi-Tenant Isolation**: Complete workspace and tenant isolation enforced at database schema, vector payload, and retrieval filter levels.
- **Budget & Token Aware**: Strict per-task ceilings on LLM calls, token budgets, and retrieval hops.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Sources["External Monitored Sources"]
        S1[Competitor Websites]
        S2[Pricing Pages]
        S3[Release Notes]
        S4[Social & Platform Feeds]
    end

    subgraph Acquisition["1. Source Acquisition"]
        C4[Crawl4AI Web Engine]
        AR[Agent Reach Adapters]
        GOV[ToS & Governance Engine]
        Sources --> C4 & AR --> GOV
    end

    subgraph Pipeline["2. Knowledge Processing"]
        NORM[Content Normalization]
        HASH[SHA-256 Hashing]
        CHUNK[Evaluated Chunking]
        EMB[BGE-M3 Dense Embedding]
        GOV --> NORM --> HASH --> CHUNK --> EMB
    end

    subgraph Storage["Durable Storage"]
        PG[(PostgreSQL 16: Canonical Docs & Relational State)]
        QD[(Qdrant: Dense Vector Store)]
        HASH --> PG
        EMB --> QD
    end

    subgraph Retrieval["3. Retrieval & Verification"]
        ROUTER[Adaptive Router]
        HYBRID[Dense Qdrant + Sparse BM25 + RRF]
        RERANK[Cross-Encoder Reranker]
        VERIF[Sufficiency & Contradiction Verifier]
        QD & PG --> ROUTER --> HYBRID --> RERANK --> VERIF
    end

    subgraph Intelligence["4. Intelligence & Decisions"]
        EVT[Event Analysis Stage]
        SIG[Signal Detection Stage]
        INS[Insight Generation Stage]
        DEC[Decision Support Stage]
        VERIF --> EVT --> SIG --> INS --> DEC
    end

    subgraph Application["5. Application API & Review UI"]
        API[FastAPI Endpoints & Prometheus /metrics]
        UI[Interactive UI Dashboard]
        OPERATOR{"Human Reviewer"}
        DEC --> API --> UI --> OPERATOR
    end

    OPERATOR -->|ACCEPTED / REJECTED| API
```

---

## 10-Layer Provenance Chain

```
[Layer 1: Recommendation] (Title, status: PROPOSED, urgency, trade-offs)
   │
[Layer 2: Selected Option] (Action type, predicted effects, identified risks)
   │
[Layer 3: Decision Context] (Business objectives, constraints, time horizons)
   │
[Layer 4: Supporting Insights] (Titles, statements, factual/interpretive confidence)
   │
[Layer 5: Inferred Signals] (Detected trajectories, market direction)
   │
[Layer 6: Supporting Events] (Discrete factual occurrences, detected timestamp)
   │
[Layer 7: Raw Evidence] (Direct quotes, evidence role: primary direct / context)
   │
[Layer 8: Document Chunks] (Chunk ID, token count, character offsets)
   │
[Layer 9: Document Version] (Content SHA-256 hash, acquisition timestamp)
   │
[Layer 10: Monitored Source] (Source ID, canonical URL, platform type)
```

---

## Subsystem Inventory

| Subsystem | Source Path | Description |
| :--- | :--- | :--- |
| **Source Acquisition** | [`src/competitive_intelligence/source_acquisition/`](src/competitive_intelligence/source_acquisition/) | Multi-source crawlers (`Crawl4AI`), social adapters, rate limiters, ToS governance. |
| **Knowledge Pipeline** | [`src/competitive_intelligence/knowledge_pipeline/`](src/competitive_intelligence/knowledge_pipeline/) | Normalization, change detection, semantic chunking, `BGE-M3` embeddings, `Qdrant` vector storage. |
| **Retrieval Engine** | [`src/competitive_intelligence/retrieval_engine/`](src/competitive_intelligence/retrieval_engine/) | Sparse BM25, dense vector search, Reciprocal Rank Fusion (RRF), cross-encoder reranking. |
| **Agentic RAG** | [`src/competitive_intelligence/agentic_rag/`](src/competitive_intelligence/agentic_rag/) | Multi-hop query decomposition, sufficiency checking, contradiction detection, bounded reasoning. |
| **Intelligence** | [`src/competitive_intelligence/intelligence/`](src/competitive_intelligence/intelligence/) | Event analysis, signal inference, strategic insight generation with multi-dimensional confidence. |
| **Decision Support** | [`src/competitive_intelligence/decision_support/`](src/competitive_intelligence/decision_support/) | Option evaluation, hard constraint filtering, prescriptive proposal assembly (`PROPOSED`). |
| **Continuous Monitoring** | [`src/competitive_intelligence/monitoring/`](src/competitive_intelligence/monitoring/) | Background scheduler daemon, lease acquisition, distributed heartbeat, worker failover. |
| **Application & UI** | [`src/competitive_intelligence/application/`](src/competitive_intelligence/application/) | FastAPI REST endpoints, web dashboard, operator review workflows, Prometheus `/metrics`. |

---

## Getting Started

### Option A: Running with Docker Compose (Recommended)
```bash
# 1. Clone repository and set environment
git clone <repo-url>
cd competitive_intelligence
cp .env.example .env

# 2. Launch production stack
docker compose up -d

# 3. Verify health
curl http://localhost:8000/health
```

### Option B: Local Development
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies (runtime extras + dev tools)
pip install -e ".[all]"

# 3. Run the test suite (hermetic: never reads your .env or calls a real LLM)
python -m pytest tests/ -q

# 4. Run static type checker
python -m mypy src

# 5. Start API server
uvicorn competitive_intelligence.application.api.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints & Interfaces

- **API Documentation**: [`http://localhost:8000/api/v1/docs`](http://localhost:8000/api/v1/docs) (Swagger UI) & [`/api/v1/redoc`](http://localhost:8000/api/v1/redoc)
- **Operator Web Dashboard**: [`http://localhost:8000/`](http://localhost:8000/)
- **Prometheus Metrics**: [`http://localhost:8000/metrics`](http://localhost:8000/metrics)
- **Health Probes**: [`http://localhost:8000/health`](http://localhost:8000/health), [`/health/live`](http://localhost:8000/health/live), [`/health/ready`](http://localhost:8000/health/ready)

### Key REST Routes:
- `POST /api/v1/workspaces`: Create isolated multi-tenant workspace.
- `POST /api/v1/workspaces/{id}/competitors`: Register tracked competitor.
- `POST /api/v1/workspaces/{id}/sources`: Configure monitored web or platform source.
- `POST /api/v1/workspaces/{id}/workflows/runs`: Trigger multi-stage ingestion & intelligence pipeline.
- `POST /api/v1/workspaces/{id}/query`: Evidence query with an agentic trace (escalation decision, sub-queries, verification, contradictions).
- `GET  /api/v1/workspaces/{id}/recommendations`: List strategic recommendations (`PROPOSED`).
- `PATCH /api/v1/workspaces/{id}/recommendations/{rec_id}/status`: Operator review (`ACCEPTED` or `REJECTED`).
- `GET  /api/v1/workspaces/{id}/recommendations/{rec_id}/provenance`: Reconstruct the 10-layer provenance chain with per-layer details.

---

## Testing & Quality Assurance

- **Regression Suite**: unit and integration suites, hermetic by default (`tests/conftest.py` ignores `.env`; tests marked `integration` need a live Crawl4AI/Qdrant and can be skipped with `-m "not integration"`).
- **Automated End-to-End Validation**: [`tests/integration/test_final_end_to_end_validation.py`](tests/integration/test_final_end_to_end_validation.py) covers the complete 19-step lifecycle.
- **Type Checking**: strict `mypy` across `src/`.
- **Zero Host Contamination**: Automated tests run with in-memory SQLite (`aiosqlite`) and Qdrant (`:memory:`), providing complete isolation and zero reliance on external daemons.

---

## Documentation

- **Final System Validation**: [`docs/final_system_validation.md`](docs/final_system_validation.md)
- **Final Project Completion Report**: [`FINAL_PROJECT_COMPLETION_REPORT.md`](FINAL_PROJECT_COMPLETION_REPORT.md)
- **Architectural Rules & Invariants**: [`AGENTS.md`](AGENTS.md)
- **Detailed Subsystem Architecture**: [`docs/architecture.md`](docs/architecture.md)
- **Production Deployment Guide**: [`docs/deployment.md`](docs/deployment.md)
- **Observability Specification**: [`docs/observability.md`](docs/observability.md)
- **LLM Integration Specification**: [`docs/llm_integration.md`](docs/llm_integration.md)
