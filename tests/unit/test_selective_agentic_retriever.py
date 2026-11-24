"""Unit tests for SelectiveAgenticRetriever, DeterministicEscalationPolicy, and ResultSelection."""

import asyncio
from typing import Any
import pytest

from competitive_intelligence.agentic_rag.selective import (
    DeterministicEscalationPolicy,
    EscalationAssessment,
    EscalationPolicyConfig,
    EscalationSignalType,
    ResultSelectionStrategy,
    SelectiveAgenticRetriever,
    SelectiveRetrieverConfig,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


class MockRetriever(BaseRetriever):
    """Mock retriever that returns configured results and tracks query calls."""

    def __init__(self, canned_results: dict[str, list[RetrievalResult]] | None = None) -> None:
        self.canned_results = canned_results or {}
        self.recorded_queries: list[RetrievalQuery] = []
        self.call_count = 0
        self.should_raise: Exception | None = None

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievalResult]:
        self.call_count += 1
        self.recorded_queries.append(query)
        if self.should_raise:
            raise self.should_raise

        if query.query_text in self.canned_results:
            return [r.model_copy(deep=True) for r in self.canned_results[query.query_text]]

        return [
            RetrievalResult(
                chunk_id=f"chk_default_{query.query_text[:10]}",
                score=0.85,
                text=f"Default text for: {query.query_text}",
                document_id="doc_def",
                version_id="v1",
                workspace_id=query.metadata_filter.workspace_id,
            )
        ]


def create_sample_result(
    chunk_id: str,
    text: str,
    score: float = 0.85,
    workspace_id: str = "ws-test",
    company_id: str | None = None,
) -> RetrievalResult:
    """Helper to build a valid RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        document_id="doc_1",
        version_id="v1",
        workspace_id=workspace_id,
        company_id=company_id,
        metadata={"company_id": company_id} if company_id else {},
    )


@pytest.mark.anyio
async def test_direct_easy_query_does_not_escalate() -> None:
    """Direct single-entity query with good initial results should not trigger escalation."""
    adaptive = MockRetriever(
        canned_results={
            "Snowflake enterprise architecture": [
                create_sample_result("chk_1", "Snowflake features multi-cluster shared data architecture with virtual warehouses.", score=0.92, company_id="snowflake")
            ]
        }
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="Snowflake enterprise architecture",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert len(results) == 1
    assert results[0].chunk_id == "chk_1"
    assert results[0].metadata["selective_escalated"] is False
    assert results[0].metadata["selective_final_source"] == "adaptive"
    assert results[0].metadata["selective_llm_calls"] == 0
    assert agentic.call_count == 0  # Agentic retriever was NOT invoked
    assert controller.last_trace is not None
    assert controller.last_trace.escalated is False


@pytest.mark.anyio
async def test_low_score_triggers_escalation() -> None:
    """Initial retrieval with top score below threshold triggers LOW_TOP_SCORE signal."""
    adaptive = MockRetriever(
        canned_results={
            "obscure database pricing model": [
                create_sample_result("chk_weak", "Tangential mention of database systems", score=0.04)
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "obscure database pricing model": [
                create_sample_result("chk_strong", "Accurate database pricing tier breakdown", score=0.89)
            ]
        }
    )
    policy_config = EscalationPolicyConfig(min_score_threshold=0.10)
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        config=SelectiveRetrieverConfig(policy_config=policy_config),
    )

    query = RetrievalQuery(
        query_text="obscure database pricing model",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.LOW_TOP_SCORE in controller.last_assessment.signals_detected
    assert results[0].metadata["selective_escalated"] is True


@pytest.mark.anyio
async def test_low_evidence_count_triggers_escalation() -> None:
    """Initial retrieval returning 0 results triggers LOW_EVIDENCE_COUNT signal."""
    adaptive = MockRetriever(canned_results={"sparse query": []})
    agentic = MockRetriever(
        canned_results={
            "sparse query": [create_sample_result("chk_rec", "Recovered evidence via decomposition", score=0.75)]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="sparse query",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.LOW_EVIDENCE_COUNT in controller.last_assessment.signals_detected
    assert len(results) == 1
    assert results[0].chunk_id == "chk_rec"


@pytest.mark.anyio
async def test_multi_entity_unbalanced_escalation() -> None:
    """Query comparing two entities where initial results only cover one triggers MULTI_ENTITY_UNBALANCED."""
    adaptive = MockRetriever(
        canned_results={
            "compare Snowflake vs Databricks table format support": [
                create_sample_result("chk_sf", "Snowflake supports Apache Iceberg table formats.", score=0.88, company_id="snowflake")
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "compare Snowflake vs Databricks table format support": [
                create_sample_result("chk_db", "Databricks supports Delta Lake format with UniForm.", score=0.86, company_id="databricks"),
                create_sample_result("chk_sf", "Snowflake supports Apache Iceberg table formats.", score=0.88, company_id="snowflake"),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="compare Snowflake vs Databricks table format support",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.MULTI_ENTITY_UNBALANCED in controller.last_assessment.signals_detected
    assert len(results) == 2


@pytest.mark.anyio
async def test_multi_entity_balanced_does_not_escalate() -> None:
    """Query comparing two entities where initial results cover both entities should not trigger entity imbalance."""
    adaptive = MockRetriever(
        canned_results={
            "compare Snowflake vs Databricks table format support": [
                create_sample_result("chk_sf", "Snowflake supports Apache Iceberg table formats.", score=0.88, company_id="snowflake"),
                create_sample_result("chk_db", "Databricks UniForm enables Delta and Iceberg reading.", score=0.87, company_id="databricks"),
            ]
        }
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="compare Snowflake vs Databricks table format support",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 0  # No escalation needed
    assert results[0].metadata["selective_escalated"] is False


@pytest.mark.anyio
async def test_evidence_gap_critical_terms_escalation() -> None:
    """Query with critical terms (e.g. egress tariff penalty) missing in top-k triggers EVIDENCE_GAP_DETECTED."""
    adaptive = MockRetriever(
        canned_results={
            "OpenAI enterprise contract egress tariff penalty provisions": [
                create_sample_result("chk_gen", "OpenAI enterprise terms and data privacy overview.", score=0.72)
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "OpenAI enterprise contract egress tariff penalty provisions": [
                create_sample_result("chk_gap", "OpenAI contract clause 14 details egress penalty and remedies.", score=0.91)
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="OpenAI enterprise contract egress tariff penalty provisions",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert EscalationSignalType.EVIDENCE_GAP_DETECTED in controller.last_assessment.signals_detected
    assert any(r.chunk_id == "chk_gap" for r in results)


@pytest.mark.anyio
async def test_multi_facet_disjoint_escalation() -> None:
    """Multi-faceted compound query with incomplete facet coverage triggers MULTI_FACET_DISJOINT."""
    adaptive = MockRetriever(
        canned_results={
            "Cloudflare capabilities across DDoS protection, serverless workers, and zero trust": [
                create_sample_result("chk_cf_ddos", "Cloudflare offers high-capacity automated DDoS protection.", score=0.82)
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "Cloudflare capabilities across DDoS protection, serverless workers, and zero trust": [
                create_sample_result("chk_cf_workers", "Cloudflare Workers execute JavaScript at edge.", score=0.85),
                create_sample_result("chk_cf_zt", "Cloudflare Access provides Zero Trust security.", score=0.84),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="Cloudflare capabilities across DDoS protection, serverless workers, and zero trust",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert EscalationSignalType.MULTI_FACET_DISJOINT in controller.last_assessment.signals_detected


@pytest.mark.anyio
async def test_preserve_better_retains_adaptive_when_agentic_inferior() -> None:
    """PRESERVE_BETTER selection retains Adaptive results if Agentic offers no novel chunks and lower score."""
    adaptive = MockRetriever(
        canned_results={
            "OpenAI enterprise security": [
                create_sample_result("chk_ad1", "High quality SOC2 report details for OpenAI", score=0.95),
            ]
        }
    )
    # Agentic returned identical chunk with worse score and 0 novel chunks
    agentic = MockRetriever(
        canned_results={
            "OpenAI enterprise security": [
                create_sample_result("chk_ad1", "High quality SOC2 report details for OpenAI", score=0.70),
            ]
        }
    )
    # Policy forced to escalate via low score threshold setup
    policy = DeterministicEscalationPolicy(EscalationPolicyConfig(min_score_threshold=0.99))
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        policy=policy,
        config=SelectiveRetrieverConfig(selection_strategy=ResultSelectionStrategy.PRESERVE_BETTER),
    )

    query = RetrievalQuery(
        query_text="OpenAI enterprise security",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert results[0].score == 0.95  # Retained the higher Adaptive score
    assert results[0].metadata["selective_final_source"] == "adaptive"


@pytest.mark.anyio
async def test_preserve_better_retains_novel_agentic_chunks() -> None:
    """PRESERVE_BETTER merges and retains novel chunks recovered by Agentic retrieval."""
    adaptive = MockRetriever(
        canned_results={
            "OpenAI egress remedies": [
                create_sample_result("chk_ad1", "OpenAI general billing terms", score=0.80),
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "OpenAI egress remedies": [
                create_sample_result("chk_novel_ag", "OpenAI specific SLA credit remedy for egress", score=0.88),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="OpenAI egress remedies",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    chunk_ids = [r.chunk_id for r in results]
    assert "chk_novel_ag" in chunk_ids
    assert "chk_ad1" in chunk_ids
    assert results[0].metadata["selective_final_source"] == "merged"


@pytest.mark.anyio
async def test_atomic_fallback_on_agentic_failure() -> None:
    """When Agentic retriever raises an unexpected error, controller safely returns initial Adaptive results."""
    adaptive = MockRetriever(
        canned_results={
            "test query": [create_sample_result("chk_ad", "Solid adaptive fallback evidence", score=0.80)]
        }
    )
    agentic = MockRetriever()
    agentic.should_raise = RuntimeError("Simulated Graph LLM Timeout")

    policy = DeterministicEscalationPolicy(EscalationPolicyConfig(min_score_threshold=0.99))
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        policy=policy,
    )

    query = RetrievalQuery(
        query_text="test query",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert len(results) == 1
    assert results[0].chunk_id == "chk_ad"
    assert results[0].metadata["selective_escalated"] is True
    assert results[0].metadata["selective_fallback"] is True
    assert "RuntimeError" in results[0].metadata["selective_fallback_reason"]


@pytest.mark.anyio
async def test_provenance_metadata_complete() -> None:
    """Full selective provenance fields are attached to retrieved results."""
    adaptive = MockRetriever(
        canned_results={"test query": [create_sample_result("chk_1", "Sample text", score=0.85)]}
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="test query",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    for r in results:
        assert "selective_initial_strategy" in r.metadata
        assert "selective_escalated" in r.metadata
        assert "selective_reasons" in r.metadata
        assert "selective_signals" in r.metadata
        assert "selective_final_source" in r.metadata
        assert "selective_llm_calls" in r.metadata
        assert "selective_retrieval_hops" in r.metadata


@pytest.mark.anyio
async def test_tenant_isolation_maintained() -> None:
    """Workspace and metadata filters are strictly preserved when delegating across retrievers."""
    adaptive = MockRetriever()
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="Snowflake security",
        metadata_filter=MetadataFilter(
            workspace_id="ws-secure-tenant-42",
            company_ids=["snowflake"],
        ),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert adaptive.recorded_queries[0].metadata_filter.workspace_id == "ws-secure-tenant-42"
    assert adaptive.recorded_queries[0].metadata_filter.company_ids == ["snowflake"]
    assert results[0].workspace_id == "ws-secure-tenant-42"


@pytest.mark.anyio
async def test_concurrency_safe_trace_recording() -> None:
    """Concurrent retrieve calls with different queries execute safely without race conditions."""
    adaptive = MockRetriever()
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    queries = [
        RetrievalQuery(query_text=f"Query {i}", metadata_filter=MetadataFilter(workspace_id=f"ws-{i}"))
        for i in range(10)
    ]

    results_list = await asyncio.gather(*(controller.retrieve(q) for q in queries))

    assert len(results_list) == 10
    for i, res in enumerate(results_list):
        assert len(res) >= 1
        assert res[0].workspace_id == f"ws-{i}"


def test_selective_retriever_is_base_retriever() -> None:
    """Verify SelectiveAgenticRetriever strictly conforms to BaseRetriever interface."""
    adaptive = MockRetriever()
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    assert isinstance(controller, BaseRetriever)
    assert hasattr(controller, "retrieve")
    assert hasattr(controller, "aretrieve")


@pytest.mark.anyio
async def test_disabled_selective_escalation_config() -> None:
    """When enable_selective_escalation is False, queries bypass assessment and return adaptive."""
    adaptive = MockRetriever(
        canned_results={
            "obscure database": [create_sample_result("chk_ad", "Ad text", score=0.01)]
        }
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        config=SelectiveRetrieverConfig(enable_selective_escalation=False),
    )

    query = RetrievalQuery(
        query_text="obscure database",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 0
    assert results[0].metadata["selective_escalated"] is False
    assert results[0].metadata["selective_final_source"] == "adaptive_disabled"


@pytest.mark.anyio
async def test_selection_strategy_hybrid_merge() -> None:
    """HYBRID_MERGE strategy merges and ranks all unique results by score descending."""
    adaptive = MockRetriever(
        canned_results={
            "hybrid query": [
                create_sample_result("chk_1", "Text 1", score=0.80),
                create_sample_result("chk_2", "Text 2", score=0.60),
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "hybrid query": [
                create_sample_result("chk_2", "Text 2 upgraded", score=0.85),
                create_sample_result("chk_3", "Text 3 novel", score=0.90),
            ]
        }
    )
    policy = DeterministicEscalationPolicy(EscalationPolicyConfig(min_score_threshold=0.99))
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        policy=policy,
        config=SelectiveRetrieverConfig(selection_strategy=ResultSelectionStrategy.HYBRID_MERGE),
    )

    query = RetrievalQuery(
        query_text="hybrid query",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert len(results) == 3
    assert results[0].chunk_id == "chk_3"  # Score 0.90
    assert results[1].chunk_id == "chk_2"  # Score 0.85 (upgraded from 0.60)
    assert results[2].chunk_id == "chk_1"  # Score 0.80
    assert results[0].metadata["selective_final_source"] == "merged"


@pytest.mark.anyio
async def test_selection_strategy_agentic_preferred() -> None:
    """AGENTIC_PREFERRED strategy returns agentic results directly when available."""
    adaptive = MockRetriever(
        canned_results={
            "pref query": [create_sample_result("chk_ad", "Ad text", score=0.80)]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "pref query": [create_sample_result("chk_ag", "Ag text", score=0.85)]
        }
    )
    policy = DeterministicEscalationPolicy(EscalationPolicyConfig(min_score_threshold=0.99))
    controller = SelectiveAgenticRetriever(
        adaptive_retriever=adaptive,
        agentic_retriever=agentic,
        policy=policy,
        config=SelectiveRetrieverConfig(selection_strategy=ResultSelectionStrategy.AGENTIC_PREFERRED),
    )

    query = RetrievalQuery(
        query_text="pref query",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert len(results) == 1
    assert results[0].chunk_id == "chk_ag"
    assert results[0].metadata["selective_final_source"] == "agentic"


# ==============================================================================
# Hardened Entity/Facet & Temporal Coverage Tests
# ==============================================================================


@pytest.mark.anyio
async def test_entity_present_but_required_facet_missing_escalates() -> None:
    """Scenario 1: Entities present in top-5, but one requested facet is missing -> escalate."""
    # Simulating entity_q18 failure mode: Pinecone & Qdrant present, but quantization RAM missing in top-5
    adaptive = MockRetriever(
        canned_results={
            "Compare Pinecone Serverless and Qdrant Cloud pricing, quantization RAM benefits, and storage costs": [
                create_sample_result("chk_pc_price", "Pinecone Serverless pricing structure details.", score=0.90, company_id="pinecone"),
                create_sample_result("chk_qd_price", "Qdrant Cloud standard pricing and storage rates.", score=0.88, company_id="qdrant"),
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "Compare Pinecone Serverless and Qdrant Cloud pricing, quantization RAM benefits, and storage costs": [
                create_sample_result("chk_qd_quant", "Qdrant scalar and product quantization RAM compression savings.", score=0.92, company_id="qdrant"),
                create_sample_result("chk_pc_price", "Pinecone Serverless pricing structure details.", score=0.90, company_id="pinecone"),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="Compare Pinecone Serverless and Qdrant Cloud pricing, quantization RAM benefits, and storage costs",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.FACET_EVIDENCE_GAP in controller.last_assessment.signals_detected
    assert "quantization ram benefits" in controller.last_assessment.assessment_metadata["missing_facets"]
    # Verify novel agentic chunk was retained
    assert any(r.chunk_id == "chk_qd_quant" for r in results)


@pytest.mark.anyio
async def test_all_entity_facet_combinations_present_no_escalation() -> None:
    """Scenario 3: Both entities and all requested facets are sufficiently present -> fast path, no escalation."""
    adaptive = MockRetriever(
        canned_results={
            "Compare Pinecone Serverless and Qdrant Cloud pricing, quantization RAM benefits, and storage costs": [
                create_sample_result("chk_pc_price", "Pinecone Serverless pricing structure and storage costs.", score=0.92, company_id="pinecone"),
                create_sample_result("chk_qd_quant", "Qdrant Cloud quantization RAM benefits and memory compression.", score=0.91, company_id="qdrant"),
            ]
        }
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="Compare Pinecone Serverless and Qdrant Cloud pricing, quantization RAM benefits, and storage costs",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 0  # Fast path preserved
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is False
    assert results[0].metadata["selective_escalated"] is False
    assert results[0].metadata["selective_final_source"] == "adaptive"


@pytest.mark.anyio
async def test_multi_facet_one_missing_facet_escalates() -> None:
    """Scenario 4: Multi-facet query spanning 3 facets where 1 facet is missing in top-5 -> escalate."""
    adaptive = MockRetriever(
        canned_results={
            "What are OpenAI developments across enterprise pricing, SOC2 compliance, and supercomputing systems hiring?": [
                create_sample_result("chk_oa_price", "OpenAI enterprise pricing tiers and commitment rates.", score=0.90),
                create_sample_result("chk_oa_soc2", "OpenAI SOC2 Type II compliance and data privacy report.", score=0.88),
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "What are OpenAI developments across enterprise pricing, SOC2 compliance, and supercomputing systems hiring?": [
                create_sample_result("chk_oa_hiring", "OpenAI supercomputing systems team hiring lead engineer.", score=0.89),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="What are OpenAI developments across enterprise pricing, SOC2 compliance, and supercomputing systems hiring?",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.FACET_EVIDENCE_GAP in controller.last_assessment.signals_detected
    assert "supercomputing systems hiring" in controller.last_assessment.assessment_metadata["missing_facets"]


@pytest.mark.anyio
async def test_temporal_query_only_current_version_escalates() -> None:
    """Scenario 5: Temporal query requesting 2023 to 2024 evolution where only 2024 is in top-5 -> escalate."""
    adaptive = MockRetriever(
        canned_results={
            "How did Snowflake pricing evolve from 2023 standard credit rates to 2024 Capacity Unit Commitment tiers?": [
                create_sample_result("chk_sf_2024", "Snowflake 2024 Capacity Unit Commitment tiers documentation.", score=0.89),
            ]
        }
    )
    agentic = MockRetriever(
        canned_results={
            "How did Snowflake pricing evolve from 2023 standard credit rates to 2024 Capacity Unit Commitment tiers?": [
                create_sample_result("chk_sf_2023", "Snowflake 2023 standard credit rate archive.", score=0.91),
            ]
        }
    )
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="How did Snowflake pricing evolve from 2023 standard credit rates to 2024 Capacity Unit Commitment tiers?",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 1
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is True
    assert EscalationSignalType.TEMPORAL_COMPARISON in controller.last_assessment.signals_detected
    assert "2023" in controller.last_assessment.assessment_metadata["temporal_epochs_missing"]


@pytest.mark.anyio
async def test_temporal_query_both_versions_no_escalation() -> None:
    """Scenario 6: Temporal query where both requested epochs (2023 & 2024) are present in top-5 -> no escalation."""
    adaptive = MockRetriever(
        canned_results={
            "How did Snowflake pricing evolve from 2023 standard credit rates to 2024 Capacity Unit Commitment tiers?": [
                create_sample_result("chk_sf_2024", "Snowflake 2024 Capacity Unit Commitment tiers documentation.", score=0.92),
                create_sample_result("chk_sf_2023", "Snowflake 2023 standard credit rate archive comparison.", score=0.90),
            ]
        }
    )
    agentic = MockRetriever()
    controller = SelectiveAgenticRetriever(adaptive_retriever=adaptive, agentic_retriever=agentic)

    query = RetrievalQuery(
        query_text="How did Snowflake pricing evolve from 2023 standard credit rates to 2024 Capacity Unit Commitment tiers?",
        metadata_filter=MetadataFilter(workspace_id="ws-test"),
        top_k=5,
    )

    results = await controller.retrieve(query)

    assert agentic.call_count == 0  # No escalation needed
    assert controller.last_assessment is not None
    assert controller.last_assessment.should_escalate is False
    assert results[0].metadata["selective_escalated"] is False


def test_no_benchmark_specific_knowledge_in_policy() -> None:
    """Scenario 15: Policy is strictly generic and contains zero hardcoded query IDs or benchmark chunk constants."""
    import inspect
    policy_source = inspect.getsource(DeterministicEscalationPolicy)

    # Ensure no query_id or benchmark chunk constant appears in production policy code
    assert "entity_q18" not in policy_source
    assert "trap_q" not in policy_source
    assert "facet_q" not in policy_source
    assert "temp_q" not in policy_source
    assert "trap_chunk_" not in policy_source
    assert "multihop_chunk_" not in policy_source


