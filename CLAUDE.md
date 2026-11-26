# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `AGENTS.md` first.** It holds the binding architectural rules: the domain ontology (Source → DocumentVersion → DocumentChunk → RawEvidence → DetectedEvent → InferredSignal → GeneratedInsight), the subsystem boundaries, the provenance rules, and the coding standards (mypy `--strict`, Pydantic v2, async I/O, structlog). This file does not repeat them.

## Commands

The project needs Python 3.11+. Package source lives in `src/` (setuptools `src` layout; pytest sets `pythonpath = ["src"]`).

```powershell
pip install -e ".[all]"            # extras: acquisition, knowledge, agentic, api, dev
python -m pytest tests/ -q          # full suite (asyncio_mode=auto, so async tests need no decorator)
python -m pytest tests/unit/test_hybrid_retriever.py::test_name -q   # single test
python -m pytest -m "not integration" -q    # skip tests that need real Crawl4AI / Qdrant
python -m mypy src                  # strict mode is configured in pyproject.toml
ruff check src tests                # line-length 100, rules E,F,I,N,W,UP,B,C4
uvicorn competitive_intelligence.application.api.app:app --reload --port 8000
python -m competitive_intelligence.monitoring.worker_cli     # standalone monitoring worker
python -m alembic upgrade head      # migrations live in alembic/versions/
docker compose up -d                # postgres:16 + qdrant + api + worker
```

The ASGI app is `competitive_intelligence.application.api.app:app`, built by `create_app()`; the UI is served at `/` and the API docs at `/api/v1/docs`.

The evaluation and benchmark scripts are in `scripts/` (`evaluate_full_system.py`, `benchmark_agentic_rag*.py`). Their datasets live in `*/evaluation/` packages inside `retrieval_engine` and `agentic_rag`.

## Configuration gotchas

- `core/config.py` `Settings` reads `.env`. The local `.env` sets `ENVIRONMENT=production`. At startup, `validate_production_configuration()` then rejects the mock LLM, mock embeddings, SQLite and in-memory Qdrant, and the API fails fast. For local runs without real services, override with `ENVIRONMENT=development`, `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=mock` and a SQLite or `QDRANT_LOCATION=:memory:` setup.
- In `development` and `test`, the app lifespan calls `create_tables()` directly. In `production` and `staging`, the schema must come from Alembic (`scripts/docker-entrypoint.sh` runs `alembic upgrade head`).
- Budget settings (`MAX_LLM_CALLS_PER_TASK`, `MAX_RETRIEVAL_HOPS`, `TASK_TOKEN_BUDGET`) feed `core/budgets.py` `LLMBudgetTracker`. Every LLM-calling node must record its usage there and short-circuit when the budget runs out.
- `MONITORING_MODE` can be `in_process`, `worker` or `disabled`. In docker-compose, the `worker` service runs the scheduler and the API does not.
- The BGE-M3 provider, the embedded Qdrant client (`QDRANT_PATH`) and `get_db_manager()` are process-wide singletons. Don't construct them per request.
- Live LLM calls from the API go through `GuardedLLMClient`, which applies a concurrency cap, a bounded queue and a cool-down after a 429. The query endpoint uses a separate, smaller budget (`QUERY_*` settings) and degrades to deterministic retrieval instead of failing.

## Architecture (big picture)

`AGENTS.md` lists six subsystems. The code also contains packages it does not describe:

- **`orchestration/`**: `orchestrator.py` runs a fixed `STAGE_ORDER`: KNOWLEDGE_PROCESSING → RETRIEVAL → EVENT_ANALYSIS → SIGNAL_DETECTION → INSIGHT_GENERATION → DECISION_SUPPORT. Each stage is a `BaseWorkflowStage` in `orchestration/stages/`. A run stops at `target_stop_stage` (default INSIGHT_GENERATION) under an `ExecutionBudget` and `RetryPolicy` (`policies.py`). Run state is persisted through `BaseWorkflowStateStore`, with the SQL implementation in `infrastructure/database/repositories/workflow_state_store.py`. Runs are idempotent (see migration 0003).
- **`decision_support/`**: `DeterministicDecisionAnalyzer` plus `LLMOptionGenerator` and a validator. It produces recommendations that are always created in `PROPOSED` status. Only an operator review through the API moves them to `ACCEPTED` or `REJECTED`. Never add autonomous acceptance.
- **`monitoring/`**: a scheduler and worker that use DB leases with heartbeats (`monitoring_lease_repo.py`, migration 0002), so several workers can run safely.
- **`infrastructure/database/`**: SQLAlchemy async ORM models, `DatabaseSessionManager` (`session.py`), and the `Sql*Repository` implementations of the repository ABCs declared in `application/interfaces.py`.

Layering in the application subsystem goes endpoint (`application/api/v1/endpoints/*`) → service (`application/services/*`) → repository interface → `Sql*Repository`. All wiring lives in `application/api/deps.py`: LLM client via `agentic_rag/llm/factory.py` `create_llm_client`, vector store, embedding service, retrievers, analyzers, and repositories. When adding a dependency, add a provider there. Tests swap implementations with `app.dependency_overrides` (see `tests/integration/api/test_api_v1.py`).

Retrieval stack (`retrieval_engine/`):
- dense Qdrant search (`dense/`) plus BM25 (`lexical/`), fused with RRF in `hybrid/`
- optional BGE-M3 reranking in `reranking/`
- a deterministic feature-based `routing/` layer that picks the strategy per query

`agentic_rag/` wraps this:
- `graph/workflow.py` is a LangGraph loop: analyze_query → execute_subqueries → verify_evidence → (corrective_retrieval → execute_subqueries)*. `should_continue` ends the loop on sufficiency, fallback, a termination reason, or `max_hops`.
- `selective/` holds `SelectiveAgenticRetriever` and `DeterministicEscalationPolicy`. They escalate to the agentic graph only when deterministic coverage checks fail on plain retrieval.

LLM providers sit behind `BaseLLMClient` (`agentic_rag/llm/`): OpenAI, Anthropic and a mock. Embeddings sit behind a provider interface: `bge_m3_provider` and `mock_provider`. Use the mocks in tests. Never call a real provider from a unit test.

The intelligence subsystem (`intelligence/`) has three parts:
- `analyzers/`: events from document diffs or retrieved evidence
- `signals/`: detector, comparators and tracker
- `insights/`: a deterministic analyzer, an LLM interpreter and a validator

The repo-wide pattern is "deterministic analyzer + LLM interpreter + validator". Keep the deterministic part authoritative and have the validator reject ungrounded LLM output.

Workspace (tenant) isolation is enforced in three places: in repository queries, in Qdrant payload filters, and in retrieval filters. Every new query path must carry `workspace_id`.

## Frontend

`application/static/` is a framework-free ES-module SPA, mounted by `create_app()` as static files. `js/router.js` is a hash router, `js/api.js` wraps the `/api/v1` REST calls, `js/state.js` holds shared state, and `js/views/*` contains one module per screen. There is no build step: edit the files and reload.

## Tests

`tests/conftest.py` disables `.env` loading, so the suite is hermetic. `tests/unit/` covers deterministic transforms, nodes and services with mocks. `tests/integration/` covers API flows on SQLite and Qdrant `:memory:`, plus `test_final_end_to_end_validation.py`, which runs the full lifecycle. Tests that run a background workflow while polling must use a file-backed SQLite (`tmp_path`). In-memory SQLite shares one `StaticPool` connection, and concurrent session rollbacks drop writes. `tests/integration/api/conftest.py` provides `seeded_workspace`, a small test corpus (`tests/support/sample_corpus.py`) seeded once through the real pipeline with the mock LLM. Tests marked `@pytest.mark.integration` (`test_crawl4ai_integration.py`, `test_qdrant_integration.py`) need real services.

The agentic graph applies the repo's deterministic-first pattern in two places. LLM-proposed sub-query filters are resolved to stored IDs in `agentic_rag/filters.py`. Numeric contradictions across documents are detected by rules in `agentic_rag/contradictions.py` and only interpreted by the verifier LLM.
