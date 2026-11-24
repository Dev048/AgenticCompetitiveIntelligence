"""Deterministic Synthetic Corpus and Frozen Ground Truth for Final System Evaluation.

Provides:
- 12 synthetic competitor profiles.
- 120+ structured synthetic documents, versions, and chunks spanning 7 categories and 4 temporal epochs.
- Frozen ground-truth definitions for 12 end-to-end evaluation scenarios.
- 4-tier outcome taxonomy: CORRECT, CAPABILITY_LIMITATION, DEFECT, UNSUPPORTED.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any

from competitive_intelligence.decision_support.models import (
    ConstraintType,
    DecisionConstraint,
    DecisionContext,
    DecisionDomain,
    DecisionObjective,
    ObjectiveDimension,
    RiskTolerance,
)
from competitive_intelligence.intelligence.models import (
    InsightType,
    IntelligenceEventType,
    SignalType,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult
from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    PlatformType,
    Source,
    SourceCategory,
    SourceClassification,
    SourceIdentity,
    SourceLocation,
    ToolChain,
)


class OutcomeClassification(str, Enum):
    """4-tier outcome taxonomy for system evaluation."""

    CORRECT = "CORRECT"
    CAPABILITY_LIMITATION = "CAPABILITY_LIMITATION"
    DEFECT = "DEFECT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class CorpusDocument:
    """Container representing a synthetic document with full lineage."""

    document_id: str
    company_id: str
    source_id: str
    workspace_id: str
    source_category: str
    canonical_url: str
    acquired_at: datetime
    content: str
    version_id: str
    chunks: list[DocumentChunk]
    source: Source
    document: Document
    version: DocumentVersion


@dataclass
class GroundTruthScenario:
    """Frozen ground truth definition for an end-to-end evaluation scenario."""

    scenario_id: str
    name: str
    description: str
    workspace_id: str
    company_ids: list[str]
    query_text: str
    required_chunk_ids: list[str]
    distractor_chunk_ids: list[str] = field(default_factory=list)
    expected_event_types: list[IntelligenceEventType] = field(default_factory=list)
    expected_signal_type: SignalType | None = None
    expected_insight_type: InsightType | None = None
    is_uncertain: bool = False
    is_negative_or_gap: bool = False
    decision_context: DecisionContext | None = None
    expected_recommendation_justified: bool = False
    expected_eliminated_option_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Detailed evaluation result for a single scenario."""

    scenario_id: str
    name: str
    outcome: OutcomeClassification
    retrieval_correct: bool = False
    intelligence_correct: bool = False
    decision_support_correct: bool = False
    provenance_correct: bool = False
    invariants_passed: bool = False
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    missed_chunk_ids: list[str] = field(default_factory=list)
    escalated_to_agentic: bool = False
    extracted_event_ids: list[str] = field(default_factory=list)
    detected_signal_ids: list[str] = field(default_factory=list)
    generated_insight_ids: list[str] = field(default_factory=list)
    recommendation_id: str | None = None
    recommendation_status: str | None = None
    duration_ms: float = 0.0
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemCorpus:
    """Collection of all synthetic documents, chunks, and ground truth scenarios."""

    workspace_id: str
    documents: list[CorpusDocument]
    chunks: list[DocumentChunk]
    retrieval_index: dict[str, RetrievalResult]
    scenarios: list[GroundTruthScenario]

    def get_chunks_for_company(self, company_id: str) -> list[DocumentChunk]:
        return [c for c in self.chunks if c.metadata.company_id == company_id]

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)


def _make_source_category(cat_str: str) -> SourceCategory:
    mapping = {
        "pricing": SourceCategory.PRICING,
        "product": SourceCategory.WEBSITE,
        "changelog": SourceCategory.DOCUMENTATION,
        "news": SourceCategory.NEWS,
        "hiring": SourceCategory.CAREERS,
        "infrastructure": SourceCategory.DOCUMENTATION,
        "compliance": SourceCategory.DOCUMENTATION,
    }
    return mapping.get(cat_str, SourceCategory.WEBSITE)


def _build_doc(
    workspace_id: str,
    company_id: str,
    doc_num: int,
    category: str,
    content: str,
    date_str: str,
    url: str,
) -> CorpusDocument:
    doc_id = f"doc_{company_id}_{doc_num:03d}"
    source_id = f"src_{company_id}_{category}"
    version_id = f"ver_{company_id}_{doc_num:03d}_1"
    chunk_id = f"chunk_{company_id}_{doc_num:03d}_1"

    dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    c_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    source = Source(
        identity=SourceIdentity(
            source_id=source_id,
            company_id=company_id,
            workspace_id=workspace_id,
            display_name=f"{company_id.replace('_', ' ').title()} {category.title()}",
        ),
        classification=SourceClassification(
            category=_make_source_category(category),
            platform=PlatformType.WEB,
            content_type=ContentType.HTML,
        ),
        location=SourceLocation(
            url_or_identifier=url,
            access_pattern=AccessPattern.PUBLIC_WEB,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(primary=AcquisitionMethod.CRAWL4AI),
        ),
    )

    doc = Document(
        document_id=doc_id,
        source_id=source_id,
        workspace_id=workspace_id,
        company_id=company_id,
        canonical_url=url,
    )

    version = DocumentVersion(
        version_id=version_id,
        document_id=doc_id,
        content=content,
        content_hash=c_hash,
        acquired_at=dt,
    )

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        version_id=version_id,
        content=content,
        metadata=ChunkMetadata(
            source_id=source_id,
            document_id=doc_id,
            version_id=version_id,
            workspace_id=workspace_id,
            company_id=company_id,
            source_url=url,
            source_category=category,
            collected_at=dt,
        ),
    )

    return CorpusDocument(
        document_id=doc_id,
        company_id=company_id,
        source_id=source_id,
        workspace_id=workspace_id,
        source_category=category,
        canonical_url=url,
        acquired_at=dt,
        content=content,
        version_id=version_id,
        chunks=[chunk],
        source=source,
        document=doc,
        version=version,
    )


def create_system_evaluation_corpus(workspace_id: str = "ws_sys_eval") -> SystemCorpus:
    """Construct the comprehensive 12-competitor, 120+ document synthetic evaluation corpus."""
    docs: list[CorpusDocument] = []
    doc_count = 0

    # --------------------------------------------------------------------------
    # 1. ACME CLOUD (Pricing reduction, down-market shift) - 12 docs
    # --------------------------------------------------------------------------
    acme_items = [
        ("pricing", "Acme Cloud updated Enterprise Plan pricing from $50 to $30/month effective Q1 2026.", "2026-01-15T00:00:00", "https://acme.com/pricing"),
        ("pricing", "Acme Cloud Standard Plan remains at $20/month with 5 seats included.", "2026-01-20T00:00:00", "https://acme.com/pricing/standard"),
        ("product", "Acme Cloud released Enterprise Dashboard v2 with real-time compute telemetry.", "2026-02-05T00:00:00", "https://acme.com/blog/dashboard-v2"),
        ("changelog", "Acme Cloud Changelog: Added SSO integration and SAML support for Enterprise.", "2026-02-18T00:00:00", "https://acme.com/changelog/2026-02"),
        ("news", "Acme Cloud announced expansion of their European datacenter region in Frankfurt.", "2026-03-01T00:00:00", "https://technews.com/acme-frankfurt"),
        ("hiring", "Acme Cloud is hiring Senior Distributed Systems Engineers for cluster orchestration.", "2026-03-10T00:00:00", "https://acme.com/careers/distributed-eng"),
        ("infrastructure", "Acme Cloud architectural documentation: Cluster provisioning latency reduced to 45ms.", "2026-03-25T00:00:00", "https://docs.acme.com/arch"),
        ("compliance", "Acme Cloud completed annual SOC 2 Type II audit across all compute zones.", "2026-04-02T00:00:00", "https://acme.com/trust/soc2"),
        ("news", "Industry analysis: Acme Cloud price reduction puts pressure on legacy hosters.", "2026-04-15T00:00:00", "https://cloudreport.io/acme-pricing-impact"),
        ("product", "Acme Cloud announced developer preview of Serverless Containers.", "2026-05-01T00:00:00", "https://acme.com/products/serverless"),
        ("changelog", "Acme Cloud Changelog: Fixed webhook delivery timeouts and payload serialization.", "2026-05-20T00:00:00", "https://acme.com/changelog/2026-05"),
        ("news", "Acme Cloud customer spotlight: Retail enterprise migrates 500TB analytics workload.", "2026-06-10T00:00:00", "https://acme.com/customers/retail-giant"),
    ]
    for cat, text, dt, url in acme_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "acme_cloud", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 2. NOVADB (Repeated sequential price reductions -> Directional Trend) - 12 docs
    # --------------------------------------------------------------------------
    nova_items = [
        ("pricing", "NovaDB updated Cluster Plan pricing from $100 to $80/month on January 10, 2026.", "2026-01-10T00:00:00", "https://novadb.io/pricing-q1"),
        ("pricing", "NovaDB updated Cluster Plan pricing from $80 to $60/month on April 15, 2026.", "2026-04-15T00:00:00", "https://novadb.io/pricing-q2"),
        ("pricing", "NovaDB updated Cluster Plan pricing from $60 to $40/month on July 20, 2026.", "2026-07-20T00:00:00", "https://novadb.io/pricing-q3"),
        ("product", "NovaDB launched vector indexing extension with HNSW support.", "2026-02-12T00:00:00", "https://novadb.io/blog/vector-support"),
        ("changelog", "NovaDB v3.1: Distributed raft consensus optimizations for secondary replicas.", "2026-03-05T00:00:00", "https://novadb.io/changelog/3-1"),
        ("news", "NovaDB featured in Cloud Database Benchmark Report 2026 for high write throughput.", "2026-03-22T00:00:00", "https://dbbench.org/2026/novadb"),
        ("hiring", "NovaDB opened 5 new roles for Storage Engine Kernel Developers in Seattle.", "2026-04-01T00:00:00", "https://novadb.io/careers/kernel"),
        ("infrastructure", "NovaDB documentation on multi-region active-active replication topologies.", "2026-05-15T00:00:00", "https://docs.novadb.io/multi-region"),
        ("compliance", "NovaDB achieved HIPAA compliance certification for healthcare data tenants.", "2026-06-01T00:00:00", "https://novadb.io/trust/hipaa"),
        ("product", "NovaDB announced Managed Serverless tier with instant autoscaling.", "2026-08-01T00:00:00", "https://novadb.io/products/serverless"),
        ("changelog", "NovaDB v3.2: Automated snapshot restore time reduced by 60%.", "2026-08-25T00:00:00", "https://novadb.io/changelog/3-2"),
        ("news", "NovaDB ecosystem reaches 20,000 active GitHub stars across client SDKs.", "2026-09-01T00:00:00", "https://github.blog/novadb-milestone"),
    ]
    for cat, text, dt, url in nova_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "novadb", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 3. APEX SEARCH (Price reduction then price restoration -> Reversal) - 10 docs
    # --------------------------------------------------------------------------
    apex_items = [
        ("pricing", "Apex Search updated Enterprise Plan pricing from $50 to $20/month on February 1, 2026.", "2026-02-01T00:00:00", "https://apexsearch.com/pricing-feb"),
        ("pricing", "Apex Search updated Enterprise Plan pricing from $20 to $50/month on June 1, 2026.", "2026-06-01T00:00:00", "https://apexsearch.com/pricing-jun"),
        ("product", "Apex Search launched hybrid sparse-dense lexical reranking engine.", "2026-02-20T00:00:00", "https://apexsearch.com/products/hybrid"),
        ("changelog", "Apex Search v2.4: Sub-millisecond filter execution on payload metadata.", "2026-03-15T00:00:00", "https://apexsearch.com/changelog/2-4"),
        ("news", "Apex Search raises $40M Series B to build semantic enterprise discovery.", "2026-04-10T00:00:00", "https://techcrunch.com/apex-search-b"),
        ("hiring", "Apex Search hiring VP of Sales to lead enterprise account expansion.", "2026-05-05T00:00:00", "https://apexsearch.com/careers/vp-sales"),
        ("infrastructure", "Apex Search published whitepaper on memory-mapped vector compression.", "2026-06-15T00:00:00", "https://apexsearch.com/research/vector-compression"),
        ("compliance", "Apex Search ISO 27001 certification renewed for 2026.", "2026-07-01T00:00:00", "https://apexsearch.com/trust/iso27001"),
        ("product", "Apex Search released LangChain and LlamaIndex official integrations.", "2026-07-20T00:00:00", "https://apexsearch.com/blog/framework-integrations"),
        ("news", "Apex Search CEO interviewed on pivot towards enterprise gross margins.", "2026-08-05T00:00:00", "https://podcast.ai/apex-pivot"),
    ]
    for cat, text, dt, url in apex_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "apex_search", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 4. VECTORFLOW (Multi-entity benchmarking vs TensorScale) - 10 docs
    # --------------------------------------------------------------------------
    vf_items = [
        ("pricing", "VectorFlow updated Streaming Tier pricing from $40 to $25/month with 10M vectors included.", "2026-02-15T00:00:00", "https://vectorflow.ai/pricing"),
        ("product", "VectorFlow released Real-Time Ingestion Pipeline with 50,000 vectors/sec throughput.", "2026-03-01T00:00:00", "https://vectorflow.ai/products/pipeline"),
        ("changelog", "VectorFlow v1.8: Native Kafka connector and streaming embeddings support.", "2026-03-18T00:00:00", "https://vectorflow.ai/changelog/1-8"),
        ("news", "VectorFlow partners with major CDN provider for edge vector caching.", "2026-04-20T00:00:00", "https://vectorflow.ai/news/edge-caching"),
        ("hiring", "VectorFlow is hiring Rust Systems Engineers for high-speed indexing.", "2026-05-10T00:00:00", "https://vectorflow.ai/careers/rust-eng"),
        ("infrastructure", "VectorFlow technical brief: Zero-copy deserialization in SIMD embeddings.", "2026-06-01T00:00:00", "https://docs.vectorflow.ai/simd"),
        ("compliance", "VectorFlow announced GDPR compliance and EU data residency options.", "2026-06-25T00:00:00", "https://vectorflow.ai/trust/gdpr"),
        ("pricing", "VectorFlow Enterprise Dedicated cluster pricing set to $500/month.", "2026-07-10T00:00:00", "https://vectorflow.ai/pricing/dedicated"),
        ("product", "VectorFlow launched Python SDK v2.0 with async client pooling.", "2026-08-01T00:00:00", "https://vectorflow.ai/sdk/python"),
        ("news", "VectorFlow achieves 99.99% uptime SLA across Q2 2026 operations.", "2026-08-20T00:00:00", "https://status.vectorflow.ai/q2-report"),
    ]
    for cat, text, dt, url in vf_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "vectorflow", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 5. TENSORSCALE (Multi-entity benchmarking vs VectorFlow) - 10 docs
    # --------------------------------------------------------------------------
    ts_items = [
        ("pricing", "TensorScale updated GPU Inference pricing from $80 to $50/hour for H100 dedicated clusters.", "2026-02-16T00:00:00", "https://tensorscale.com/pricing-gpu"),
        ("product", "TensorScale launched Distributed Inference Engine with 100,000 tokens/sec capacity.", "2026-03-02T00:00:00", "https://tensorscale.com/products/inference"),
        ("changelog", "TensorScale v2.1: FlashAttention-3 integration for long-context windows.", "2026-03-25T00:00:00", "https://tensorscale.com/changelog/2-1"),
        ("news", "TensorScale secures 5,000 NVIDIA Blackwell GPUs for Q3 datacenter expansion.", "2026-04-25T00:00:00", "https://reuters.com/tensorscale-nvidia"),
        ("hiring", "TensorScale recruiting Principal CUDA Optimization Specialists in Austin.", "2026-05-15T00:00:00", "https://tensorscale.com/careers/cuda"),
        ("infrastructure", "TensorScale published datacenter architecture with liquid cooling pods.", "2026-06-05T00:00:00", "https://tensorscale.com/infra/liquid-cooling"),
        ("compliance", "TensorScale completes FedRAMP High readiness assessment.", "2026-07-01T00:00:00", "https://tensorscale.com/trust/fedramp"),
        ("pricing", "TensorScale introduces spot instance pricing for batch embeddings at $10/hour.", "2026-07-15T00:00:00", "https://tensorscale.com/pricing/spot"),
        ("product", "TensorScale released Open-Source Model Gateway with speculative decoding.", "2026-08-05T00:00:00", "https://tensorscale.com/blog/model-gateway"),
        ("news", "TensorScale ranks #1 in MLPerf Inference benchmark throughput.", "2026-08-25T00:00:00", "https://mlperf.org/results/tensorscale-2026"),
    ]
    for cat, text, dt, url in ts_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "tensorscale", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 6. BYTEBASE AI (Temporal version evolution v1.0 -> v2.0 -> v3.0) - 10 docs
    # --------------------------------------------------------------------------
    bb_items = [
        ("changelog", "ByteBase AI v1.0 released with basic schema migration and SQL review for MySQL.", "2026-01-15T00:00:00", "https://bytebase.com/changelog/1-0"),
        ("changelog", "ByteBase AI v2.0 released with PostgreSQL support and automated rollback pipelines.", "2026-04-15T00:00:00", "https://bytebase.com/changelog/2-0"),
        ("changelog", "ByteBase AI v3.0 released with multi-cloud Snowflake and BigQuery data governance.", "2026-07-15T00:00:00", "https://bytebase.com/changelog/3-0"),
        ("pricing", "ByteBase AI updated Team Tier pricing from $30 to $15/user/month.", "2026-02-10T00:00:00", "https://bytebase.com/pricing"),
        ("product", "ByteBase AI launched AI-assisted SQL query optimizer and index advisor.", "2026-03-01T00:00:00", "https://bytebase.com/products/ai-advisor"),
        ("news", "ByteBase AI named Gartner Cool Vendor in Database DevOps 2026.", "2026-05-01T00:00:00", "https://gartner.com/cool-vendors/bytebase"),
        ("hiring", "ByteBase AI hiring Developer Advocates for European developer communities.", "2026-05-20T00:00:00", "https://bytebase.com/careers/advocate"),
        ("infrastructure", "ByteBase AI architecture: Zero-downtime schema migrations via shadow tables.", "2026-06-15T00:00:00", "https://docs.bytebase.com/shadow-tables"),
        ("compliance", "ByteBase AI achieved SOC 2 Type II and ISO 27001 dual compliance.", "2026-07-01T00:00:00", "https://bytebase.com/trust"),
        ("news", "ByteBase AI community edition crosses 1M Docker pulls.", "2026-08-10T00:00:00", "https://bytebase.com/blog/1m-pulls"),
    ]
    for cat, text, dt, url in bb_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "bytebase_ai", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 7. ZENITH AI (Contradictory evidence: Blog vs Press leak) - 10 docs
    # --------------------------------------------------------------------------
    zen_items = [
        ("product", "Zenith AI Official Blog: Multimodal Voice Mode release delayed to Q4 2026 for safety evaluation.", "2026-03-10T00:00:00", "https://zenith.ai/blog/voice-status"),
        ("news", "TechNews Insider Report: Zenith AI launched Multimodal Voice Mode in select European enterprise accounts in Q2 2026.", "2026-03-12T00:00:00", "https://technews.com/zenith-voice-leak"),
        ("pricing", "Zenith AI Enterprise Tier pricing announced at $60/user/month with unlimited tokens.", "2026-01-20T00:00:00", "https://zenith.ai/pricing"),
        ("product", "Zenith AI released Vision Pro SDK for visual reasoning and document parsing.", "2026-02-15T00:00:00", "https://zenith.ai/products/vision"),
        ("changelog", "Zenith AI API v2: Streaming audio chunks and latency telemetry headers.", "2026-04-01T00:00:00", "https://zenith.ai/changelog/api-v2"),
        ("hiring", "Zenith AI is recruiting Senior Speech Synthesis Scientists in London.", "2026-04-20T00:00:00", "https://zenith.ai/careers/speech"),
        ("infrastructure", "Zenith AI infrastructure notes: 8-bit quantized weights deployed on edge clusters.", "2026-05-15T00:00:00", "https://zenith.ai/infra/quantization"),
        ("compliance", "Zenith AI complies with EU AI Act High-Risk system documentation guidelines.", "2026-06-01T00:00:00", "https://zenith.ai/trust/ai-act"),
        ("news", "Zenith AI customer case study: Global bank automates loan underwriting with Vision Pro.", "2026-07-10T00:00:00", "https://zenith.ai/customers/global-bank"),
        ("product", "Zenith AI previewed Reasoning Agents with structured tool calling.", "2026-08-01T00:00:00", "https://zenith.ai/products/agents"),
    ]
    for cat, text, dt, url in zen_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "zenith_ai", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 8. CLOUDQUERY IO (Retrieval distractors: High keyword overlap) - 10 docs
    # --------------------------------------------------------------------------
    cq_items = [
        ("pricing", "CloudQuery IO updated Enterprise Analytics pricing from $45 to $25/month for unlimited syncs.", "2026-03-01T00:00:00", "https://cloudquery.io/pricing"),
        ("news", "Distractor: The pricing model for cloud infrastructure in IO operations requires careful budgeting.", "2026-01-10T00:00:00", "https://techblog.com/cloud-pricing-discussion"),
        ("news", "Distractor: How query engines optimize pricing and IO bandwidth in big data workloads.", "2026-02-05T00:00:00", "https://dbweekly.com/query-pricing-io"),
        ("news", "Distractor: Comparative analysis of IO cloud cost allocation across multi-cloud enterprises.", "2026-02-28T00:00:00", "https://finops.org/cloud-io-pricing"),
        ("news", "Distractor: Cloud query performance benchmarks and IO cost tradeoffs.", "2026-04-10T00:00:00", "https://perfreport.io/query-io-costs"),
        ("product", "CloudQuery IO released AWS and GCP sync plugins with incremental change capture.", "2026-03-20T00:00:00", "https://cloudquery.io/products/plugins"),
        ("changelog", "CloudQuery IO v3.5: Support for destination tables in ClickHouse and DuckDB.", "2026-04-25T00:00:00", "https://cloudquery.io/changelog/3-5"),
        ("hiring", "CloudQuery IO hiring Integration Engineers for SaaS connector ecosystem.", "2026-05-15T00:00:00", "https://cloudquery.io/careers/connectors"),
        ("infrastructure", "CloudQuery IO documentation on high-concurrency schema reflection.", "2026-06-10T00:00:00", "https://docs.cloudquery.io/reflection"),
        ("compliance", "CloudQuery IO achieves SOC 2 Type II compliance.", "2026-07-05T00:00:00", "https://cloudquery.io/trust"),
    ]
    for cat, text, dt, url in cq_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "cloudquery_io", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 9. DATAPULSE ANALYTICS (Hiring expansion) - 10 docs
    # --------------------------------------------------------------------------
    dp_items = [
        ("hiring", "DataPulse Analytics launched massive hiring campaign: recruiting 50 AI and Machine Learning Engineers across North America.", "2026-02-01T00:00:00", "https://datapulse.ai/careers/expansion"),
        ("hiring", "DataPulse Analytics appoints former Google VP as Chief Technology Officer.", "2026-03-15T00:00:00", "https://datapulse.ai/news/cto-appointment"),
        ("pricing", "DataPulse Analytics updated Starter Plan pricing from $29 to $19/month.", "2026-01-10T00:00:00", "https://datapulse.ai/pricing"),
        ("product", "DataPulse Analytics launched Semantic BI Layer with natural language queries.", "2026-03-01T00:00:00", "https://datapulse.ai/products/semantic-bi"),
        ("changelog", "DataPulse Analytics v4.0: Real-time dashboard caching with sub-second reload.", "2026-04-10T00:00:00", "https://datapulse.ai/changelog/4-0"),
        ("news", "DataPulse Analytics named leader in Modern BI Platforms 2026.", "2026-05-01T00:00:00", "https://bireport.com/datapulse"),
        ("infrastructure", "DataPulse Analytics whitepaper: In-memory columnar storage for dashboard aggregates.", "2026-06-01T00:00:00", "https://datapulse.ai/research/columnar"),
        ("compliance", "DataPulse Analytics achieves HIPAA and SOC 2 Type II certification.", "2026-06-20T00:00:00", "https://datapulse.ai/trust"),
        ("product", "DataPulse Analytics announced Slack and Teams bot for metric alerts.", "2026-07-15T00:00:00", "https://datapulse.ai/products/bot"),
        ("news", "DataPulse Analytics customer base reaches 1,500 mid-market companies.", "2026-08-01T00:00:00", "https://datapulse.ai/milestones"),
    ]
    for cat, text, dt, url in dp_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "datapulse_analytics", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 10. OMNISCALE INFRA (Infrastructure investment) - 10 docs
    # --------------------------------------------------------------------------
    omni_items = [
        ("infrastructure", "OmniScale Infra committed $100M capital expenditure to build 3 green energy datacenters in Scandinavia.", "2026-01-25T00:00:00", "https://omniscale.net/news/scandinavia-datacenters"),
        ("infrastructure", "OmniScale Infra connected 100Gbps dedicated dark fiber backbone between Frankfurt and Stockholm.", "2026-04-10T00:00:00", "https://omniscale.net/infra/fiber"),
        ("pricing", "OmniScale Infra updated Compute Node pricing from $70 to $45/month.", "2026-02-15T00:00:00", "https://omniscale.net/pricing"),
        ("product", "OmniScale Infra released Bare-Metal Kubernetes Orchestration v2.", "2026-03-10T00:00:00", "https://omniscale.net/products/k8s"),
        ("changelog", "OmniScale Infra Changelog: Sub-5ms network transit times across European nodes.", "2026-05-01T00:00:00", "https://omniscale.net/changelog/2026-05"),
        ("news", "OmniScale Infra partners with European clean energy grid operators.", "2026-05-20T00:00:00", "https://reuters.com/omniscale-green-energy"),
        ("hiring", "OmniScale Infra hiring Datacenter Operations Directors in Stockholm.", "2026-06-01T00:00:00", "https://omniscale.net/careers/dc-ops"),
        ("compliance", "OmniScale Infra achieves ISO 14001 environmental sustainability certification.", "2026-07-01T00:00:00", "https://omniscale.net/trust/iso14001"),
        ("product", "OmniScale Infra announced confidential computing pods with AMD SEV-SNP.", "2026-07-25T00:00:00", "https://omniscale.net/products/confidential"),
        ("news", "OmniScale Infra hosts European Sovereign Cloud Summit 2026.", "2026-08-15T00:00:00", "https://omniscale.net/summit"),
    ]
    for cat, text, dt, url in omni_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "omniscale_infra", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 11. SECUREVAULT SEC (Compliance & security certifications) - 10 docs
    # --------------------------------------------------------------------------
    sec_items = [
        ("compliance", "SecureVault Sec achieved FedRAMP High Authorization and Department of Defense Impact Level 5 clearance.", "2026-02-10T00:00:00", "https://securevault.io/trust/fedramp-high"),
        ("compliance", "SecureVault Sec announced full compliance with EU Digital Operational Resilience Act (DORA).", "2026-05-15T00:00:00", "https://securevault.io/trust/dora"),
        ("pricing", "SecureVault Sec updated Enterprise Vault pricing from $150 to $99/month.", "2026-01-15T00:00:00", "https://securevault.io/pricing"),
        ("product", "SecureVault Sec launched Quantum-Resistant Key Management Module.", "2026-03-01T00:00:00", "https://securevault.io/products/post-quantum"),
        ("changelog", "SecureVault Sec v5.2: Automated hardware security module (HSM) key rotation.", "2026-04-01T00:00:00", "https://securevault.io/changelog/5-2"),
        ("news", "SecureVault Sec protects cryptographic secrets for 40% of Fortune 100 banks.", "2026-04-20T00:00:00", "https://fintechtimes.com/securevault"),
        ("hiring", "SecureVault Sec is recruiting Cryptographic Verification Engineers in Zurich.", "2026-05-01T00:00:00", "https://securevault.io/careers/crypto"),
        ("infrastructure", "SecureVault Sec published formal verification proofs for key derivation protocol.", "2026-06-10T00:00:00", "https://securevault.io/research/formal-proofs"),
        ("product", "SecureVault Sec announced Kubernetes Secrets Operator with automated injection.", "2026-07-01T00:00:00", "https://securevault.io/products/k8s-operator"),
        ("news", "SecureVault Sec CEO keynotes RSA Conference 2026 on zero-trust cryptography.", "2026-08-01T00:00:00", "https://rsaconference.com/securevault-keynote"),
    ]
    for cat, text, dt, url in sec_items:
        doc_count += 1
        docs.append(_build_doc(workspace_id, "securevault_sec", doc_count, cat, text, dt, url))

    # --------------------------------------------------------------------------
    # 12. GHOSTCORP ADVERSARIAL (Zero factual data in corpus) - 0 docs (intentional)
    # --------------------------------------------------------------------------

    # Flatten all chunks and build retrieval index
    all_chunks: list[DocumentChunk] = []
    retrieval_index: dict[str, RetrievalResult] = {}

    for doc in docs:
        for chunk in doc.chunks:
            all_chunks.append(chunk)
            retrieval_index[chunk.chunk_id] = RetrievalResult(
                chunk_id=chunk.chunk_id,
                score=0.95,
                text=chunk.content,
                document_id=doc.document_id,
                version_id=doc.version_id,
                source_id=doc.source_id,
                company_id=doc.company_id,
                workspace_id=workspace_id,
                source_category=doc.source_category,
                canonical_url=doc.canonical_url,
            )

    # --------------------------------------------------------------------------
    # FROZEN GROUND TRUTH DEFINITIONS FOR THE 12 SCENARIOS
    # --------------------------------------------------------------------------
    scenarios: list[GroundTruthScenario] = [
        # SC-01: Simple Competitive Event
        GroundTruthScenario(
            scenario_id="SC-01",
            name="Simple Competitive Event",
            description="Acme Cloud single price drop from $50 to $30/month.",
            workspace_id=workspace_id,
            company_ids=["acme_cloud"],
            query_text="Acme Cloud Enterprise Plan pricing updates",
            required_chunk_ids=["chunk_acme_cloud_001_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-02: Repeated Change (Directional Trend)
        GroundTruthScenario(
            scenario_id="SC-02",
            name="Repeated Pricing Reductions (Directional Trend)",
            description="NovaDB executes 3 sequential price drops across Q1, Q2, and Q3 ($100 -> $80 -> $60 -> $40).",
            workspace_id=workspace_id,
            company_ids=["novadb"],
            query_text="NovaDB sequential Cluster Plan pricing history over 2026",
            required_chunk_ids=["chunk_novadb_013_1", "chunk_novadb_014_1", "chunk_novadb_015_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=SignalType.DIRECTIONAL_TREND,
            expected_insight_type=InsightType.PRICING_STRATEGY,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-03: Strategic Reversal
        GroundTruthScenario(
            scenario_id="SC-03",
            name="Strategic Pricing Reversal",
            description="Apex Search drops Enterprise Plan price to $20 and later restores it to $50.",
            workspace_id=workspace_id,
            company_ids=["apex_search"],
            query_text="Apex Search Enterprise Plan pricing adjustments in 2026",
            required_chunk_ids=["chunk_apex_search_025_1", "chunk_apex_search_026_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=SignalType.REVERSAL,
            expected_insight_type=InsightType.PRICING_STRATEGY,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-04: Multi-Entity Comparison
        GroundTruthScenario(
            scenario_id="SC-04",
            name="Multi-Entity Competitive Comparison",
            description="Compare VectorFlow streaming embeddings pricing/throughput with TensorScale GPU cluster pricing.",
            workspace_id=workspace_id,
            company_ids=["vectorflow", "tensorscale"],
            query_text="Compare VectorFlow and TensorScale throughput and pricing",
            required_chunk_ids=["chunk_vectorflow_035_1", "chunk_tensorscale_045_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-05: Temporal Version Evolution
        GroundTruthScenario(
            scenario_id="SC-05",
            name="Temporal Version Evolution",
            description="ByteBase AI changelog evolution across v1.0, v2.0, and v3.0.",
            workspace_id=workspace_id,
            company_ids=["bytebase_ai"],
            query_text="ByteBase AI feature release timeline from v1.0 to v3.0",
            required_chunk_ids=["chunk_bytebase_ai_055_1", "chunk_bytebase_ai_056_1", "chunk_bytebase_ai_057_1"],
            expected_event_types=[IntelligenceEventType.RELEASE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-06: Contradictory Evidence
        GroundTruthScenario(
            scenario_id="SC-06",
            name="Contradictory Evidence Preservation",
            description="Zenith AI Multimodal Voice Mode: Official blog states delayed to Q4 vs TechNews leak reports launched in Q2.",
            workspace_id=workspace_id,
            company_ids=["zenith_ai"],
            query_text="Zenith AI Voice Mode release status and availability",
            required_chunk_ids=["chunk_zenith_ai_065_1", "chunk_zenith_ai_066_1"],
            expected_event_types=[IntelligenceEventType.PRODUCT_CHANGE],
            expected_signal_type=None,
            expected_insight_type=InsightType.UNCERTAINTY,
            is_uncertain=True,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-07: Evidence Gap (Insufficiency Exposure)
        GroundTruthScenario(
            scenario_id="SC-07",
            name="Evidence Gap (Unannounced Proprietary Algorithm)",
            description="Query for Acme Cloud unannounced custom silicon TPU architecture (not present in corpus).",
            workspace_id=workspace_id,
            company_ids=["acme_cloud"],
            query_text="Acme Cloud proprietary custom TPU v5 silicon accelerator specs",
            required_chunk_ids=[],  # Zero chunks exist
            expected_event_types=[],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=True,
            expected_recommendation_justified=False,
        ),

        # SC-08: Retrieval Distractors
        GroundTruthScenario(
            scenario_id="SC-08",
            name="Retrieval Distractors Robustness",
            description="CloudQuery IO pricing query surrounded by high lexical-overlap distractor chunks.",
            workspace_id=workspace_id,
            company_ids=["cloudquery_io"],
            query_text="CloudQuery IO Enterprise Analytics pricing updates",
            required_chunk_ids=["chunk_cloudquery_io_075_1"],
            distractor_chunk_ids=[
                "chunk_cloudquery_io_076_1",
                "chunk_cloudquery_io_077_1",
                "chunk_cloudquery_io_078_1",
                "chunk_cloudquery_io_079_1",
            ],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
        ),

        # SC-09: Negative / Adversarial Query
        GroundTruthScenario(
            scenario_id="SC-09",
            name="Negative / Adversarial Query",
            description="Query asks about GhostCorp acquiring Acme Cloud for $10B (fabricated claim).",
            workspace_id=workspace_id,
            company_ids=["ghostcorp_adversarial", "acme_cloud"],
            query_text="GhostCorp acquisition of Acme Cloud for $10 billion in cash",
            required_chunk_ids=[],  # Zero chunks exist
            expected_event_types=[],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=True,
            expected_recommendation_justified=False,
        ),

        # SC-10: Decision Support Full Pipeline
        GroundTruthScenario(
            scenario_id="SC-10",
            name="Decision Support Full Pipeline",
            description="Grounded pricing insight evaluated under explicit objectives, priority weights, and budget constraint.",
            workspace_id=workspace_id,
            company_ids=["acme_cloud"],
            query_text="Acme Cloud Enterprise Plan price drop to $30",
            required_chunk_ids=["chunk_acme_cloud_001_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=True,
            expected_eliminated_option_count=1,
            decision_context=DecisionContext(
                decision_id="dec_sc10",
                workspace_id=workspace_id,
                title="Acme Cloud Down-Market Response Strategy",
                decision_domain=DecisionDomain.PRICING,
                business_objectives=[
                    DecisionObjective(
                        name="Market Share Defense",
                        dimension=ObjectiveDimension.GROWTH,
                        target_description="Neutralize competitor acquisition volume in entry tier.",
                        priority_weight=0.85,
                    ),
                    DecisionObjective(
                        name="Cost Control",
                        dimension=ObjectiveDimension.COST_EFFICIENCY,
                        target_description="Keep response implementation costs within strict limit.",
                        priority_weight=0.15,
                    ),
                ],
                constraints=[
                    DecisionConstraint(
                        constraint_type=ConstraintType.BUDGET,
                        description="Maximum allowable budget $15,000",
                        is_hard_constraint=True,
                        threshold_value=15000.0,
                    )
                ],
                risk_tolerance=RiskTolerance.MODERATE,
            ),
        ),

        # SC-11: Missing Decision Context Safety
        GroundTruthScenario(
            scenario_id="SC-11",
            name="Missing Decision Context Safety",
            description="Valid pricing insight provided with empty DecisionContext objectives.",
            workspace_id=workspace_id,
            company_ids=["acme_cloud"],
            query_text="Acme Cloud pricing update",
            required_chunk_ids=["chunk_acme_cloud_001_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,
            decision_context=DecisionContext(
                decision_id="dec_sc11_empty",
                workspace_id=workspace_id,
                title="Unspecified Objectives Context",
                decision_domain=DecisionDomain.PRICING,
                business_objectives=[],  # Empty objectives -> must produce 0 options & rec=None
            ),
        ),

        # SC-12: Conflicting Objectives (No Hidden Optimizer)
        GroundTruthScenario(
            scenario_id="SC-12",
            name="Conflicting Objectives Without Priority Weights",
            description="Multiple feasible options evaluated under competing objectives without priority weights.",
            workspace_id=workspace_id,
            company_ids=["acme_cloud"],
            query_text="Acme Cloud Enterprise pricing update",
            required_chunk_ids=["chunk_acme_cloud_001_1"],
            expected_event_types=[IntelligenceEventType.PRICING_CHANGE],
            expected_signal_type=None,
            expected_insight_type=None,
            is_uncertain=False,
            is_negative_or_gap=False,
            expected_recommendation_justified=False,  # Unweighted multi-objective -> candidate options returned, rec=None
            decision_context=DecisionContext(
                decision_id="dec_sc12_unweighted",
                workspace_id=workspace_id,
                title="Unweighted Multi-Objective Dilemma",
                decision_domain=DecisionDomain.PRICING,
                business_objectives=[
                    DecisionObjective(
                        name="Aggressive Growth",
                        dimension=ObjectiveDimension.GROWTH,
                        target_description="Maximize prospect volume.",
                        priority_weight=None,  # Missing explicit weight
                    ),
                    DecisionObjective(
                        name="Margin Preservation",
                        dimension=ObjectiveDimension.COST_EFFICIENCY,
                        target_description="Preserve gross margins at all costs.",
                        priority_weight=None,  # Missing explicit weight
                    ),
                ],
                constraints=[],
            ),
        ),
    ]

    return SystemCorpus(
        workspace_id=workspace_id,
        documents=docs,
        chunks=all_chunks,
        retrieval_index=retrieval_index,
        scenarios=scenarios,
    )
