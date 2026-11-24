# AGENTS.md: Architectural Rules & Engineering Guidelines

This document establishes the binding architectural invariants, design principles, and coding standards for all AI agents and engineers contributing to the **Competitive Intelligence Monitoring Agent** codebase.

---

## 1. Core Architecture & Philosophy

1. **Production-First Agentic RAG**: This is an evidence-backed intelligence monitoring platform, not a simple RAG chatbot. Agentic behavior is used for bounded, structured reasoning (routing, query planning, verification, synthesis), not uncontrolled open-ended execution.
2. **Deterministic Over LLM**: Keep deterministic logic (change detection, hashing, deduplication, rule-based filtering, score fusion, parsing) strictly deterministic. Never substitute simple algorithms or hash checks with LLM calls.
3. **Budget and Latency Aware**: LLM calls and token budgets are strictly tracked and capped. Every pipeline and agent node must be capable of short-circuiting when budget thresholds are reached.
4. **Tool Abstraction & Preferred Tools**: External tools must always be wrapped behind internal domain interfaces.
   - **Preferred Tools**:
     - `Crawl4AI` (Web crawling and structured extraction)
     - `Agent Reach` (Social & platform sources: X, Reddit, GitHub, YouTube)
     - `Qdrant` (Vector storage & payload filtering)
     - `LangGraph` (Deterministic state machine orchestration)
   - **No Premature Infrastructure**: Do NOT introduce or adopt additional external services (such as external message brokers, distributed queues, or third-party monitoring platforms) simply because they appear in standard architecture diagrams. Additional tools must only be introduced when a concrete requirement justifies them.
5. **Incremental Subsystem Design**: `models.py` and `interfaces.py` files in subsystems that have not yet been designed are **provisional placeholders**, not finalized domain contracts. Each subsystem's contract is designed iteratively before implementation.

---

## 2. Ontology & Domain Disambiguation

Do **NOT** treat data types interchangeably. The system distinguishes the following core entities:

```
[Source]
   │ (where & how to acquire)
   ▼
[SourceDocument] ──▶ [DocumentVersion]
   │ (content snapshot & hash)
   ▼
[DocumentChunk]
   │ (searchable unit with vector/lexical index)
   ▼
[RawEvidence]
   │ (grounded excerpt with provenance: chunk_id, source_id, timestamp)
   ▼
[DetectedEvent]
   │ (verified factual occurrence: e.g., price dropped from $99 to $79)
   ▼
[InferredSignal]
   │ (cross-evidence trend/implication: e.g., aggressive down-market expansion)
   ▼
[GeneratedInsight] ──▶ [Report] / [Alert]
   │ (actionable business intelligence backed by chained evidence)
```

### Definitions:
- **`Source`**: Metadata describing WHERE and HOW information is acquired (URL, monitoring frequency, reliability, tier, auth). Contains *no document text*.
- **`SourceDocument`**: Identity container for documents gathered from a `Source`.
- **`DocumentVersion`**: An immutable snapshot of acquired text/content at a specific timestamp with content hashing for diffing.
- **`DocumentChunk`**: A chunked unit of a `DocumentVersion` enriched with metadata, embeddings, and provenance pointers.
- **`RawEvidence`**: A factual snippet retrieved from one or more chunks with explicit source citations.
- **`DetectedEvent`**: A discrete, verifiable occurrence extracted from document versions or diffs (e.g., "Feature X added to pricing tier Y").
- **`InferredSignal`**: A higher-level pattern or strategic indicator synthesized from multiple events or evidence.
- **`GeneratedInsight`**: Synthesized analytical takeaway with explicit evidence grounding.
- **`Report` / `Alert`**: Final delivery artifacts distributed to users/workspaces.

---

## 3. Subsystem Boundaries & Invariants

```
src/competitive_intelligence/
├── core/                  # Shared settings, logging, metrics, budgets, error types
├── source_acquisition/    # Subsystem 1: Source management, crawlers, schedulers, adapters
├── knowledge_pipeline/    # Subsystem 2: Normalization, versioning, deduplication, chunking, embeddings, Qdrant
├── retrieval_engine/      # Subsystem 3: Vector search, BM25, hybrid retrieval, fusion (RRF), cross-encoder reranking
├── agentic_rag/           # Subsystem 4: LangGraph workflows, query router, decomposition, verification, self-correction
├── intelligence/          # Subsystem 5: Event detection, signal extraction, cross-source synthesis, reports, alerts
└── application/           # Subsystem 6: FastAPI routers, domain schemas, workspace/tenant boundaries
```

### Invariant Rules:
1. **Source Acquisition** only interacts with external web/API resources and emits `SourceDocument` / `DocumentVersion`. It does not perform chunking or embedding.
2. **Knowledge Pipeline** processes `DocumentVersion` into normalized `DocumentChunk` records and indexes them into Vector/Keyword stores. It does not perform agentic reasoning.
3. **Retrieval Engine** takes structured `RetrievalQuery` inputs and returns scored `ScoredDocumentChunk` / `RawEvidence` items. It does not synthesize end-user intelligence.
4. **Agentic RAG** orchestrates query planning, verification, and multi-hop retrieval using LangGraph. Agents operate within deterministic state machines with bounded max hops and budget checks.
5. **Intelligence Subsystem** performs change detection, event/signal extraction, and report/alert generation while preserving end-to-end provenance.
6. **Application Subsystem** exposes clean REST API endpoints. It does not directly manage low-level crawling or raw database connections.

---

## 4. Provenance & Verification Invariants

- Every piece of retrieved data must maintain full provenance:
  `source_id`, `document_id`, `version_id`, `chunk_id`, `retrieved_at`, and `source_url`.
- Any synthesized claim in `GeneratedInsight` or `Report` **MUST** include direct references to its supporting `RawEvidence`.
- Unsubstantiated claims must be flagged by verification nodes (`evidence_sufficiency` and `contradiction_detection`).

---

## 5. Open Architectural Decisions (To Be Decided During Subsystem Design)

1. **Raw Content Storage Strategy**: Local filesystem, database blob, or dedicated object store.
2. **Scheduling Mechanism**: In-process scheduler, event-driven triggers, or background worker queue.
3. **Crawl4AI Execution Mode**: In-process async library vs standalone containerized service.

---

## 6. Engineering & Code Quality Standards

1. **Python 3.11+ Type Annotations**: All function signatures, data models, and interfaces must have strict type annotations (`mypy --strict` compliant).
2. **Pydantic V2**: Use Pydantic models for data validation, serialization, and settings.
3. **Async First**: All I/O operations (crawling, database queries, vector searches, API requests, LLM calls) must be `async`/`await` enabled.
4. **Testing**:
   - Unit tests for all deterministic transforms (chunking, deduplication, score fusion, parsing).
   - Mocked integration tests for vector DB, external APIs, and LLM calls.
   - Retrieval evaluation suites for benchmarking Recall@K, MRR, and NDCG.
5. **Structured Logging**: Use `structlog` to log structured JSON with correlation IDs, `workspace_id`, and `task_id`.
