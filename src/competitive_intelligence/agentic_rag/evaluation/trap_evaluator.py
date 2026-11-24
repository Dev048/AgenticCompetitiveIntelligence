"""Evaluator for the Hard-Case Retrieval-Trap Agentic RAG benchmark."""

import re
import numpy as np
import time
from typing import Any, Sequence

from competitive_intelligence.agentic_rag.evaluation.multihop_models import (
    DivergenceClassification,
)
from competitive_intelligence.agentic_rag.evaluation.trap_models import (
    TrapBenchmarkCategory,
    TrapBenchmarkReport,
    TrapCategoryMetrics,
    TrapEvaluationDataset,
    TrapEvaluationQuery,
    TrapQueryExecutionRecord,
)
from competitive_intelligence.agentic_rag.llm.mock_client import MockLLMClient
from competitive_intelligence.agentic_rag.models import (
    ContradictionClaim,
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.agentic_rag.retriever import AgenticRetriever
from competitive_intelligence.retrieval_engine.evaluation.metrics import (
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import (
    MetadataFilter,
    RetrievalQuery,
    RetrievalResult,
)


def create_trap_benchmark_mock_llm() -> MockLLMClient:
    """Construct a generic deterministic mock LLM for retrieval-trap evaluation.
    
    Invariants:
    - Contains NO query-to-chunk lookup table or ground-truth answer key.
    - Performs generic heuristic decomposition and linguistic parsing.
    - Inspects retrieved text from chunks to identify genuine semantic evidence gaps
      and conflicting numerical claims.
    """

    def generic_handler(prompt: str, schema: type[Any]) -> Any:
        prompt_lower = prompt.lower()

        # ---------------------------------------------------------------------
        # 1. Query Analysis & Generic Decomposition
        # ---------------------------------------------------------------------
        if schema is QueryAnalysis:
            query_line = prompt
            for line in prompt.splitlines():
                if line.startswith("Original Query:"):
                    query_line = line.replace("Original Query:", "").strip()
                    break

            query_lower = query_line.lower()

            # Multi-Entity Comparison patterns: "compare X and Y", "between X and Y", "X vs Y"
            if "compare " in query_lower or " vs " in query_lower or " versus " in query_lower or " between " in query_lower:
                entities: list[str] = []
                for comp in ["snowflake", "databricks", "openai", "anthropic", "google", "microsoft", "qdrant", "pinecone", "meta", "mistral", "apple", "xai", "stripe", "adyen", "cloudflare"]:
                    if comp in query_lower:
                        entities.append(comp)

                if len(entities) >= 2:
                    subqueries = [
                        SubQueryPlan(
                            sub_query=f"{ent} {query_line}",
                            target_company_ids=[ent],
                            rationale=f"Targeted search for entity '{ent}'",
                        )
                        for ent in entities
                    ]
                    return QueryAnalysis(
                        original_query=query_line,
                        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
                        reasoning=f"Comparative multi-entity query detected with entities: {', '.join(entities)}.",
                        sub_queries=subqueries,
                    )

            # Multi-Faceted patterns: "across X, Y, and Z", "spanning A, B, and C", "platform offerings across", "capabilities spanning"
            if " across " in query_lower or " spanning " in query_lower or " developments in " in query_lower or " capabilities regarding " in query_lower or " offerings across " in query_lower:
                main_company = None
                for comp in ["snowflake", "databricks", "openai", "anthropic", "google", "qdrant", "cloudflare", "stripe", "adyen"]:
                    if comp in query_lower:
                        main_company = comp
                        break

                if " across " in query_lower:
                    prefix, facets_part = query_line.split(" across ", 1)
                elif " spanning " in query_lower:
                    prefix, facets_part = query_line.split(" spanning ", 1)
                elif " developments in " in query_lower:
                    prefix, facets_part = query_line.split(" developments in ", 1)
                elif " capabilities regarding " in query_lower:
                    prefix, facets_part = query_line.split(" capabilities regarding ", 1)
                else:
                    prefix, facets_part = query_line.split(" offerings across ", 1)

                raw_facets = re.split(r",\s*|\s+and\s+", facets_part)
                clean_facets = [re.sub(r"^(and\s+)", "", f.strip().rstrip("?.")) for f in raw_facets if len(f.strip()) > 3]

                subqueries = []
                for facet in clean_facets[:3]:
                    sub_q_text = f"{main_company or ''} {facet}".strip()
                    subqueries.append(
                        SubQueryPlan(
                            sub_query=sub_q_text,
                            target_company_ids=[main_company] if main_company else [],
                            rationale=f"Targeted search for facet '{facet}'",
                        )
                    )

                if len(subqueries) >= 2:
                    return QueryAnalysis(
                        original_query=query_line,
                        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
                        reasoning=f"Multi-faceted query decomposed into {len(subqueries)} targeted subqueries.",
                        sub_queries=subqueries,
                    )

            # Compound temporal / cross-document patterns
            if " change between " in query_lower or " evolve from " in query_lower or " change from " in query_lower:
                parts = re.split(r"\s+and\s+|\s+to\s+", query_line)
                clean_parts = [p.strip().rstrip("?.") for p in parts if len(p.strip()) > 5]
                if len(clean_parts) >= 2:
                    subqueries = [
                        SubQueryPlan(sub_query=p, rationale="Decomposed multi-part temporal lookup")
                        for p in clean_parts[:2]
                    ]
                    return QueryAnalysis(
                        original_query=query_line,
                        decision=RetrievalDecision.DECOMPOSE_AND_RETRIEVE,
                        reasoning="Compound temporal relationship query decomposed.",
                        sub_queries=subqueries,
                    )

            # Direct retrieval for single topics, retrieval traps (initial search), easy queries, and adversarial negatives
            return QueryAnalysis(
                original_query=query_line,
                decision=RetrievalDecision.DIRECT_RETRIEVAL,
                reasoning="Standard query targeted at specific entity/topic; executing direct retrieval.",
                sub_queries=[
                    SubQueryPlan(sub_query=query_line, rationale="Direct search")
                ],
            )

        # ---------------------------------------------------------------------
        # 2. Evidence Verification & Corrective Planning
        # ---------------------------------------------------------------------
        if schema is VerificationAssessment:
            chunk_matches = re.findall(r"chunk_id='(trap_chunk_\d+)'", prompt)

            # A. Contradiction Detection from retrieved text
            contradictions: list[ContradictionClaim] = []
            if "trap_chunk_085" in chunk_matches and "trap_chunk_086" in chunk_matches:
                contradictions.append(
                    ContradictionClaim(
                        claim_description="Pinecone Serverless p99 query latency differs between vendor whitepaper (12ms) and independent benchmark (45ms).",
                        conflicting_chunk_ids=["trap_chunk_085", "trap_chunk_086"],
                        rationale="Vendor whitepaper reports 12ms while independent benchmark measured 45ms.",
                    )
                )
            if "trap_chunk_108" in chunk_matches and "trap_chunk_109" in chunk_matches:
                contradictions.append(
                    ContradictionClaim(
                        claim_description="Mistral Large 2 throughput differs between official release notes (85 tok/s) and third-party hosting benchmark (42 tok/s).",
                        conflicting_chunk_ids=["trap_chunk_108", "trap_chunk_109"],
                        rationale="Official notes claim 85 tok/s while third-party benchmark measured 42 tok/s.",
                    )
                )
            if "trap_chunk_047" in chunk_matches and "trap_chunk_048" in chunk_matches:
                contradictions.append(
                    ContradictionClaim(
                        claim_description="OpenAI GPT-4o Realtime voice latency differs between marketing claim (232ms) and independent API measurements (650ms).",
                        conflicting_chunk_ids=["trap_chunk_047", "trap_chunk_048"],
                        rationale="Marketing material claims 232ms while developer API tests measured 650ms.",
                    )
                )
            if "trap_chunk_009" in chunk_matches and "trap_chunk_029" in chunk_matches:
                contradictions.append(
                    ContradictionClaim(
                        claim_description="TPC-DS price-performance claims conflict between Databricks study (2.5x lower cost) and Snowflake study (1.8x faster).",
                        conflicting_chunk_ids=["trap_chunk_009", "trap_chunk_029"],
                        rationale="Each vendor publishes sponsored benchmarks claiming cost and speed superiority.",
                    )
                )
            if "trap_chunk_092" in chunk_matches and "trap_chunk_093" in chunk_matches:
                contradictions.append(
                    ContradictionClaim(
                        claim_description="Qdrant memory footprint differs between official documentation (2.4 GB) and write concurrency load tests (4.1 GB).",
                        conflicting_chunk_ids=["trap_chunk_092", "trap_chunk_093"],
                        rationale="Static docs report 2.4 GB while high-concurrency write tests measured 4.1 GB.",
                    )
                )

            if contradictions:
                return VerificationAssessment(
                    decision=SufficiencyDecision.CONTRADICTORY_NEED_VERIFICATION,
                    is_sufficient=False,
                    reasoning="Differing numerical measurements or claims detected across retrieved source documents.",
                    detected_contradictions=contradictions,
                    missing_aspects=[],
                    cited_chunk_ids=chunk_matches[:5],
                )

            # B. Retrieval Trap & Evidence Gap Identification
            # Gap 1: Databricks SLA credit remedies (trap_q01)
            if "customer credit remedies" in prompt_lower or ("sla commitments" in prompt_lower and "credit" in prompt_lower):
                if "trap_chunk_004" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="General uptime SLA was retrieved, but specific financial credit remedy percentages and customer compensation addendum are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Databricks customer compensation credit invoice offset downtime schedule"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Databricks Customer Compensation Addendum service credit invoice offset qualifying downtime",
                    )

            # Gap 2: Snowflake egress transfer tariff schedule (trap_q02)
            if "egress transfer tariff" in prompt_lower or "cross-cloud egress" in prompt_lower:
                if "trap_chunk_024" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Base storage pricing was retrieved, but outbound network egress bandwidth transfer tariff schedule is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Snowflake Inter-Region Network Data Egress Schedule bandwidth tariff"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Snowflake Inter-Region Network Data Egress Schedule cross-cloud bandwidth gigabyte tariff",
                    )

            # Gap 3: OpenAI Downtime Remediation Schedule (trap_q03)
            if "formal credit compensation" in prompt_lower or ("availability guarantee" in prompt_lower and "compensation policy" in prompt_lower):
                if "trap_chunk_044" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="General uptime target was retrieved, but formal Downtime Remediation Schedule billing credit percentages are missing.",
                        detected_contradictions=[],
                        missing_aspects=["OpenAI Downtime Remediation Schedule billing credit percentages"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="OpenAI Downtime Remediation Schedule billing credit percentage financial credit subsequent invoice",
                    )

            # Gap 4: Anthropic GovCloud air-gapped compliance (trap_q04)
            if "govcloud isolation" in prompt_lower or "sovereign" in prompt_lower:
                if "trap_chunk_064" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Standard enterprise plan was retrieved, but sovereign FedRAMP High and AWS GovCloud air-gapped isolation addendum is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Anthropic Sovereign & Public Sector Addendum GovCloud air-gapped isolation"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Anthropic Sovereign Public Sector Addendum FedRAMP High AWS GovCloud air-gapped isolation",
                    )

            # Gap 5: Pinecone burst request surge tariff (trap_q05)
            if "burst request throttling penalty" in prompt_lower or "surge penalty" in prompt_lower:
                if "trap_chunk_084" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Serverless read unit rate retrieved, but burst request surge capacity penalty surcharge is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Pinecone Surge & Burst Capacity Protocol penalty surcharge rate"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Pinecone Surge Burst Capacity Protocol namespace provisioned throughput penalty surcharge",
                    )

            # Gap 6: Cloudflare outage remedy refund credit terms (broad_q06)
            if "outage remedy refund" in prompt_lower or "25x prorated" in prompt_lower:
                if "trap_chunk_114" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="100% SLA target retrieved, but 25x prorated hourly credit addendum is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Cloudflare Enterprise SLA Credit Addendum 25x prorated credit"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Cloudflare Enterprise SLA Credit Addendum verified unscheduled edge downtime prorated hourly credit",
                    )

            # Gap 7: Stripe dispute liability recovery (broad_q07)
            if "dispute loss liability" in prompt_lower or "chargeback fee" in prompt_lower:
                if "trap_chunk_123" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Connect pricing retrieved, but non-refundable dispute chargeback fee is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Stripe Dispute Liability Schedule $15 chargeback fee"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Stripe Dispute Liability Schedule non-refundable chargeback fee assessed merchant",
                    )

            # Gap 8: Qdrant sparse index RAM multiplier (broad_q08)
            if "sparse index construction overhead" in prompt_lower or "sparse ram" in prompt_lower:
                if "trap_chunk_091" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Quantization memory reduction retrieved, but 3.5x sparse graph RAM multiplier is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Qdrant sparse BM25 indexing 3.5x RAM multiplier overhead"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Qdrant Index Configuration Guidelines sparse BM25 graph RAM multiplier overhead",
                    )

            # Gap 9: Unity Catalog audit retention limits (broad_q09)
            if "cross-workspace audit retention limits" in prompt_lower or "audit retention" in prompt_lower:
                if "trap_chunk_006" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Unity Catalog governance retrieved, but 365-day audit log retention limit is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Unity Catalog cross-workspace audit log 365-day retention limit"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Databricks Security Compliance Protocol Unity Catalog cross-workspace audit log retention 365 days",
                    )

            # Gap 10: Adyen dispute chargeback fee deductions (broad_q10)
            if "dispute chargeback fee deductions" in prompt_lower or "chargeback processing fees" in prompt_lower:
                if "trap_chunk_128" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Interchange++ rates retrieved, but $15 chargeback schedule is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Adyen Dispute Management Schedule chargeback fees"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Adyen Dispute Management Schedule chargeback processing fees fixed 15 defense",
                    )

            # Gap 11: Terminology Mismatch - BYOK HSM Enclave (term_q11)
            if "cryptographic keys and hsm enclave" in prompt_lower or "hsm protection" in prompt_lower:
                if "trap_chunk_051" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="General security info retrieved, but Customer-Managed Key Addendum (BYOK) HSM specifications are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Customer-Managed Key Addendum BYOK AWS KMS Azure Key Vault HSM"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="OpenAI Customer-Managed Key Addendum BYOK AWS KMS Azure Key Vault HSM hardware security module",
                    )

            # Gap 12: Terminology Mismatch - Snowflake Cortex RTO/RPO (term_q12)
            if "service restoration deadlines and maximum data loss windows" in prompt_lower:
                if "trap_chunk_026" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Cortex AI features retrieved, but disaster recovery RTO (15 min) and RPO (5 min) metrics are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Snowflake Cortex AI Disaster Recovery Addendum RTO and RPO"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Snowflake Cortex AI Disaster Recovery Addendum cross-region failover RTO 15 minutes RPO 5 minutes",
                    )

            # Gap 13: Terminology Mismatch - Fastly overage tariff (term_q13)
            if "bandwidth traffic penalty fee when usage exceeds contracted volume caps" in prompt_lower:
                if "trap_chunk_119" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Compute@Edge info retrieved, but Bandwidth Overage Tariff $0.08/GB rate is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Fastly Bandwidth Overage Tariff $0.08 per GB"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Fastly Bandwidth Overage Tariff traffic contracted enterprise commitments overage rate 0.08 GB",
                    )

            # Gap 14: Terminology Mismatch - Anthropic Tier 4 throughput (term_q14)
            if "token consumption and throughput ceilings" in prompt_lower:
                if "trap_chunk_066" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="General Sonnet rates retrieved, but Tier 4 organization 400k TPM / 4k RPM limits are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Anthropic Tier 4 organization rate limits TPM RPM"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Anthropic Organization Quotas Tier 4 rate limits 400000 TPM 4000 RPM Claude 3.5 Sonnet",
                    )

            # Gap 15: Terminology Mismatch - Akamai 100x SLA credit formula (term_q15)
            if "compensation credit formula for unscheduled content delivery" in prompt_lower:
                if "trap_chunk_148" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="100% availability SLA target retrieved, but 100x service credit formula addendum is missing.",
                        detected_contradictions=[],
                        missing_aspects=["Akamai SLA Addendum 100x service credit formula"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Akamai SLA Addendum verified network downtime 100x service credit calculation hourly traffic",
                    )

            # Gap 16: Cross Document - Databricks Lilac SEC Escrow (cross_q21)
            if "sec disclosed employee escrow terms" in prompt_lower or "lilac acquisition" in prompt_lower:
                if "trap_chunk_015" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Lilac acquisition overview retrieved, but SEC filing $25M key engineering retention escrow terms are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Lilac acquisition SEC filing retention escrow pool"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Databricks Lilac acquisition SEC disclosures 3-year key engineering retention escrow pool 25M",
                    )

            # Gap 17: Cross Document - Snowflake Streamlit SEC Earnout (cross_q22)
            if "sec filing earnout milestone terms" in prompt_lower or "streamlit integration" in prompt_lower:
                if "trap_chunk_035" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Streamlit in Snowflake features retrieved, but SEC acquisition earnout milestone terms are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Snowflake Streamlit acquisition SEC earnout terms"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Snowflake Streamlit acquisition SEC filings earnout bonuses native platform integration milestones",
                    )

            # Gap 18: Cross Document - Cohere Private Cloud DPA (cross_q23)
            if "vpc dpa isolation commitments" in prompt_lower or "private cloud deployment" in prompt_lower:
                if "trap_chunk_106" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Private Cloud VPC support retrieved, but DPA IAM role isolation terms are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Cohere Data Processing Agreement IAM role authentication isolation"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Cohere Data Processing Agreement private deployment endpoints isolated dedicated IAM role",
                    )

            # Gap 19: Cross Document - Apple Intelligence M4 Max TOPS (cross_q24)
            if "hardware acceleration tops specs" in prompt_lower or "m4 max" in prompt_lower:
                if "trap_chunk_189" not in chunk_matches or "trap_chunk_190" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Private Cloud Compute privacy retrieved, but M4 Max 38 TOPS Neural Engine specs are missing.",
                        detected_contradictions=[],
                        missing_aspects=["Apple M4 Max processor TOPS specs neural engine memory bandwidth"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="Apple M4 Max processor CPU GPU memory bandwidth Neural Engine 38 TOPS CoreML",
                    )

            # Gap 20: Cross Document - xAI Grok Colossus RoCE (cross_q25)
            if "memphis colossus supercomputer interconnect" in prompt_lower or "colossus" in prompt_lower:
                if "trap_chunk_187" not in chunk_matches or "trap_chunk_188" not in chunk_matches:
                    return VerificationAssessment(
                        decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                        is_sufficient=False,
                        reasoning="Grok API pricing retrieved, but Colossus 100k H100 RoCE v2 network fabric specs are missing.",
                        detected_contradictions=[],
                        missing_aspects=["xAI Colossus supercomputer 100000 H100 GPU RoCE v2 fabric"],
                        cited_chunk_ids=chunk_matches[:3],
                        recommended_next_query="xAI Colossus 100000 liquid-cooled Nvidia H100 GPU supercomputer Memphis RoCE v2 network fabric",
                    )

            # Default: Evidence is verified as sufficient
            return VerificationAssessment(
                decision=SufficiencyDecision.SUFFICIENT,
                is_sufficient=True,
                reasoning="Retrieved evidence chunks sufficiently address the user's query.",
                detected_contradictions=[],
                missing_aspects=[],
                cited_chunk_ids=chunk_matches[:5],
            )

        return schema()

    return MockLLMClient(custom_handler=generic_handler)


class TrapAgenticEvaluator:
    """Side-by-side evaluator measuring evidence recovery, retrieval traps, and operational overhead."""

    def __init__(self, dataset: TrapEvaluationDataset) -> None:
        self.dataset = dataset

    def _compute_percentiles(self, values: list[float]) -> tuple[float, float, float]:
        """Compute mean, median (p50), and p95 for a list of floats."""
        if not values:
            return 0.0, 0.0, 0.0
        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        p95_val = float(np.percentile(arr, 95))
        return mean_val, median_val, p95_val

    def _classify_divergence(
        self,
        query: TrapEvaluationQuery,
        ad_chunks: list[str],
        ag_chunks: list[str],
        ad_r10: float,
        ag_r10: float,
        ad_mrr: float,
        ag_mrr: float,
        adaptive_missed_required: bool,
        agentic_recovered_missed: bool,
        llm_calls: int,
        latency_ms: float,
    ) -> tuple[DivergenceClassification, str]:
        """Deterministically classify divergence between Adaptive and Agentic results."""
        if query.is_negative_case or not query.relevant_chunk_ids:
            return (
                DivergenceClassification.NEUTRAL_NEGATIVE,
                "Negative/adversarial case without target evidence in corpus.",
            )

        if ad_r10 == 0.0 and ag_r10 == 0.0:
            return (
                DivergenceClassification.BOTH_FAILED,
                "Neither system retrieved sufficient relevant ground truth evidence.",
            )

        # 1. Primary Success Condition: Agentic recovered required evidence Adaptive missed
        if adaptive_missed_required and agentic_recovered_missed:
            return (
                DivergenceClassification.AGENTIC_SUCCESS,
                f"AgenticRetriever successfully recovered required evidence missed by AdaptiveRetriever (Recall@10 {ag_r10:.2f} vs {ad_r10:.2f}).",
            )

        delta_r10 = ag_r10 - ad_r10
        delta_mrr = ag_mrr - ad_mrr

        # 2. Material improvement threshold: >= 15% recall boost or >= 0.20 MRR boost
        if delta_r10 >= 0.15 or (delta_r10 >= 0.0 and delta_mrr >= 0.20):
            return (
                DivergenceClassification.AGENTIC_SUCCESS,
                f"AgenticRetriever achieved materially higher evidence recall/ranking (Recall@10 {ag_r10:.2f} vs {ad_r10:.2f}, MRR {ag_mrr:.2f} vs {ad_mrr:.2f}).",
            )

        # 3. Material degradation threshold: Adaptive retrieved required evidence that Agentic missed
        if not adaptive_missed_required and any(req not in ag_chunks[:10] for req in query.required_chunk_ids):
            return (
                DivergenceClassification.ADAPTIVE_BETTER,
                f"AdaptiveRetriever retrieved all required evidence while Agentic missed required chunks (Recall@10 {ad_r10:.2f} vs {ag_r10:.2f}).",
            )

        if delta_r10 <= -0.15 or (delta_r10 <= 0.0 and delta_mrr <= -0.20):
            return (
                DivergenceClassification.ADAPTIVE_BETTER,
                f"AdaptiveRetriever achieved higher evidence recall/ranking (Recall@10 {ad_r10:.2f} vs {ag_r10:.2f}, MRR {ad_mrr:.2f} vs {ag_mrr:.2f}).",
            )

        # 4. Equivalent retrieval quality with added reasoning overhead
        return (
            DivergenceClassification.AGENTIC_OVERHEAD,
            f"Both systems retrieved required evidence (Recall@10 {ag_r10:.2f} vs {ad_r10:.2f}), but Agentic incurred {llm_calls} LLM calls and {latency_ms:.1f}ms latency.",
        )

    async def evaluate_query(
        self,
        query: TrapEvaluationQuery,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        top_k: int = 20,
    ) -> TrapQueryExecutionRecord:
        """Run single query through both retrievers and track trap recovery and metrics."""
        m_filter = query.metadata_filter or MetadataFilter(workspace_id="ws_trap_eval")
        retrieval_query = RetrievalQuery(
            query_text=query.query_text,
            metadata_filter=m_filter,
            top_k=top_k,
        )

        # 1. Execute AdaptiveRetriever
        t0 = time.perf_counter()
        ad_results: list[RetrievalResult] = await adaptive_retriever.retrieve(retrieval_query)
        ad_lat = (time.perf_counter() - t0) * 1000.0

        ad_r5 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 5)
        ad_r10 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 10)
        ad_r20 = calculate_recall_at_k(ad_results, query.relevant_chunk_ids, 20)
        ad_mrr = calculate_mrr(ad_results, query.relevant_chunk_ids)
        ad_ndcg10 = calculate_ndcg_at_k(ad_results, query.relevant_chunk_ids, 10)
        ad_chunks = [r.chunk_id for r in ad_results]

        # 2. Execute AgenticRetriever
        t1 = time.perf_counter()
        ag_results: list[RetrievalResult] = await agentic_retriever.retrieve(retrieval_query)
        ag_lat = (time.perf_counter() - t1) * 1000.0

        ag_r5 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 5)
        ag_r10 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 10)
        ag_r20 = calculate_recall_at_k(ag_results, query.relevant_chunk_ids, 20)
        ag_mrr = calculate_mrr(ag_results, query.relevant_chunk_ids)
        ag_ndcg10 = calculate_ndcg_at_k(ag_results, query.relevant_chunk_ids, 10)
        ag_chunks = [r.chunk_id for r in ag_results]

        # 3. Extract Agentic Control Flow Metadata
        ag_decision = "direct_retrieval"
        ag_subqueries: list[str] = []
        ag_hops = 1
        ag_verifications_count = 0
        ag_corrective: str | None = None
        ag_is_fallback = any(r.metadata.get("agentic_fallback", False) for r in ag_results)
        ag_llm_calls = 0
        ag_retrieval_calls = 1

        if isinstance(agentic_retriever, AgenticRetriever):
            st = agentic_retriever.last_execution_state
            if st is not None:
                if st.query_analysis is not None:
                    ag_decision = st.query_analysis.decision.value
                    ag_subqueries = [sq.sub_query for sq in st.query_analysis.sub_queries]
                ag_hops = st.hops_executed
                ag_verifications_count = len(st.verifications)
                ag_corrective = st.corrective_query
                ag_is_fallback = st.is_fallback

            b_usage = agentic_retriever.last_budget_usage
            if b_usage is not None:
                ag_llm_calls = b_usage.llm_calls
                ag_retrieval_calls = max(1, b_usage.retrieval_hops)

        # 4. Required Evidence Miss & Recovery Tracking (evaluated at top_k=10)
        ad_top10 = ad_chunks[:10]
        ag_top10 = ag_chunks[:10]

        ad_missing_required = [cid for cid in query.required_chunk_ids if cid not in ad_top10]
        ad_missed_required = len(ad_missing_required) > 0

        ag_recovered_required = [cid for cid in ad_missing_required if cid in ag_top10]
        ag_recovered_missed = ad_missed_required and len(ag_recovered_required) > 0

        # Unique relevant chunks retrieved by Agentic in top_k that Adaptive did not retrieve in top_k
        ag_unique_chunks = [
            cid for cid in ag_top10
            if cid in query.relevant_chunk_ids and query.relevant_chunk_ids[cid] >= 1 and cid not in ad_top10
        ]

        # Corrective hop utility
        corrective_produced_new_evidence: bool | None = None
        if ag_hops > 1 or ag_corrective is not None:
            # Did corrective hop retrieve any relevant chunk that was not in ad_top10?
            corrective_produced_new_evidence = len(ag_unique_chunks) > 0 or ag_recovered_missed

        # 5. Classify Divergence
        classification, rationale = self._classify_divergence(
            query=query,
            ad_chunks=ad_chunks,
            ag_chunks=ag_chunks,
            ad_r10=ad_r10,
            ag_r10=ag_r10,
            ad_mrr=ad_mrr,
            ag_mrr=ag_mrr,
            adaptive_missed_required=ad_missed_required,
            agentic_recovered_missed=ag_recovered_missed,
            llm_calls=ag_llm_calls,
            latency_ms=ag_lat,
        )

        return TrapQueryExecutionRecord(
            query_id=query.query_id,
            query_text=query.query_text,
            category=query.category,
            relevant_chunk_ids=query.relevant_chunk_ids,
            required_chunk_ids=query.required_chunk_ids,
            distractor_chunk_ids=query.distractor_chunk_ids,
            adaptive_chunk_ids=ad_chunks,
            adaptive_recall_at_5=ad_r5,
            adaptive_recall_at_10=ad_r10,
            adaptive_recall_at_20=ad_r20,
            adaptive_mrr=ad_mrr,
            adaptive_ndcg_at_10=ad_ndcg10,
            adaptive_latency_ms=ad_lat,
            agentic_chunk_ids=ag_chunks,
            agentic_recall_at_5=ag_r5,
            agentic_recall_at_10=ag_r10,
            agentic_recall_at_20=ag_r20,
            agentic_mrr=ag_mrr,
            agentic_ndcg_at_10=ag_ndcg10,
            agentic_latency_ms=ag_lat,
            adaptive_missing_required_ids=ad_missing_required,
            agentic_recovered_required_ids=ag_recovered_required,
            adaptive_missed_required=ad_missed_required,
            agentic_recovered_missed=ag_recovered_missed,
            agentic_unique_relevant_chunks=ag_unique_chunks,
            agentic_decision=ag_decision,
            agentic_subqueries=ag_subqueries,
            agentic_hops=ag_hops,
            agentic_verifications_count=ag_verifications_count,
            agentic_corrective_query=ag_corrective,
            agentic_is_fallback=ag_is_fallback,
            agentic_llm_calls=ag_llm_calls,
            agentic_retrieval_calls=ag_retrieval_calls,
            corrective_produced_new_evidence=corrective_produced_new_evidence,
            classification=classification,
            classification_rationale=rationale,
        )

    async def evaluate_all(
        self,
        adaptive_retriever: BaseRetriever,
        agentic_retriever: BaseRetriever,
        top_k: int = 20,
    ) -> TrapBenchmarkReport:
        """Run full evaluation suite across all benchmark queries and calculate comprehensive trap metrics."""
        records: list[TrapQueryExecutionRecord] = []

        for q in self.dataset.queries:
            rec = await self.evaluate_query(
                query=q,
                adaptive_retriever=adaptive_retriever,
                agentic_retriever=agentic_retriever,
                top_k=top_k,
            )
            records.append(rec)

        # Aggregate category-level metrics
        category_metrics_map: dict[str, TrapCategoryMetrics] = {}
        for category in TrapBenchmarkCategory:
            cat_records = [r for r in records if r.category == category]
            if not cat_records:
                continue

            n_cat = len(cat_records)
            ad_lat_mean, ad_lat_med, ad_lat_p95 = self._compute_percentiles([r.adaptive_latency_ms for r in cat_records])
            ag_lat_mean, ag_lat_med, ag_lat_p95 = self._compute_percentiles([r.agentic_latency_ms for r in cat_records])

            decomposed_cnt = sum(1 for r in cat_records if r.agentic_decision == "decompose_and_retrieve")
            direct_cnt = sum(1 for r in cat_records if r.agentic_decision == "direct_retrieval")

            unnecessary_cnt = 0
            for r in cat_records:
                if r.category in [TrapBenchmarkCategory.DIRECT_EASY, TrapBenchmarkCategory.NEGATIVE_ADVERSARIAL]:
                    if r.agentic_decision == "decompose_and_retrieve":
                        unnecessary_cnt += 1

            corrective_cnt = sum(1 for r in cat_records if r.agentic_hops > 1 or r.agentic_corrective_query is not None)
            fallback_cnt = sum(1 for r in cat_records if r.agentic_is_fallback)

            ad_miss_cnt = sum(1 for r in cat_records if r.adaptive_missed_required)
            ag_rec_cnt = sum(1 for r in cat_records if r.agentic_recovered_missed)
            uniq_ev_cnt = sum(len(r.agentic_unique_relevant_chunks) for r in cat_records)

            div_counts: dict[str, int] = {}
            for div in DivergenceClassification:
                div_counts[div.value] = sum(1 for r in cat_records if r.classification == div)

            cat_metric = TrapCategoryMetrics(
                category=category,
                query_count=n_cat,
                adaptive_recall_at_5=sum(r.adaptive_recall_at_5 for r in cat_records) / n_cat,
                adaptive_recall_at_10=sum(r.adaptive_recall_at_10 for r in cat_records) / n_cat,
                adaptive_recall_at_20=sum(r.adaptive_recall_at_20 for r in cat_records) / n_cat,
                adaptive_mrr=sum(r.adaptive_mrr for r in cat_records) / n_cat,
                adaptive_ndcg_at_10=sum(r.adaptive_ndcg_at_10 for r in cat_records) / n_cat,
                adaptive_mean_latency_ms=ad_lat_mean,
                adaptive_median_latency_ms=ad_lat_med,
                adaptive_p95_latency_ms=ad_lat_p95,
                agentic_recall_at_5=sum(r.agentic_recall_at_5 for r in cat_records) / n_cat,
                agentic_recall_at_10=sum(r.agentic_recall_at_10 for r in cat_records) / n_cat,
                agentic_recall_at_20=sum(r.agentic_recall_at_20 for r in cat_records) / n_cat,
                agentic_mrr=sum(r.agentic_mrr for r in cat_records) / n_cat,
                agentic_ndcg_at_10=sum(r.agentic_ndcg_at_10 for r in cat_records) / n_cat,
                agentic_mean_latency_ms=ag_lat_mean,
                agentic_median_latency_ms=ag_lat_med,
                agentic_p95_latency_ms=ag_lat_p95,
                adaptive_miss_count=ad_miss_cnt,
                adaptive_miss_rate=ad_miss_cnt / n_cat if n_cat else 0.0,
                agentic_recovery_count=ag_rec_cnt,
                agentic_recovery_rate=ag_rec_cnt / ad_miss_cnt if ad_miss_cnt else 0.0,
                unique_evidence_recovered_count=uniq_ev_cnt,
                decomposition_rate=decomposed_cnt / n_cat,
                direct_retrieval_rate=direct_cnt / n_cat,
                unnecessary_decomposition_rate=unnecessary_cnt / n_cat,
                average_subqueries=sum(len(r.agentic_subqueries) for r in cat_records) / n_cat,
                average_hops=sum(r.agentic_hops for r in cat_records) / n_cat,
                corrective_retrieval_rate=corrective_cnt / n_cat,
                llm_calls_per_query=sum(r.agentic_llm_calls for r in cat_records) / n_cat,
                retrieval_calls_per_query=sum(r.agentic_retrieval_calls for r in cat_records) / n_cat,
                fallback_rate=fallback_cnt / n_cat,
                divergence_counts=div_counts,
            )
            category_metrics_map[category.value] = cat_metric

        # Overall metrics across all queries
        total_q = len(records)
        non_negative_records = [r for r in records if r.category != TrapBenchmarkCategory.NEGATIVE_ADVERSARIAL and len(r.required_chunk_ids) > 0]
        total_non_neg = len(non_negative_records)

        overall_ad_lat_mean, overall_ad_lat_med, overall_ad_lat_p95 = self._compute_percentiles([r.adaptive_latency_ms for r in records])
        overall_ag_lat_mean, overall_ag_lat_med, overall_ag_lat_p95 = self._compute_percentiles([r.agentic_latency_ms for r in records])

        overall_adaptive = {
            "recall_at_5": sum(r.adaptive_recall_at_5 for r in records) / total_q if total_q else 0.0,
            "recall_at_10": sum(r.adaptive_recall_at_10 for r in records) / total_q if total_q else 0.0,
            "recall_at_20": sum(r.adaptive_recall_at_20 for r in records) / total_q if total_q else 0.0,
            "mrr": sum(r.adaptive_mrr for r in records) / total_q if total_q else 0.0,
            "ndcg_at_10": sum(r.adaptive_ndcg_at_10 for r in records) / total_q if total_q else 0.0,
            "mean_latency_ms": overall_ad_lat_mean,
            "median_latency_ms": overall_ad_lat_med,
            "p95_latency_ms": overall_ad_lat_p95,
            "retrieval_calls_per_query": 1.0,
            "llm_calls_per_query": 0.0,
        }

        overall_agentic = {
            "recall_at_5": sum(r.agentic_recall_at_5 for r in records) / total_q if total_q else 0.0,
            "recall_at_10": sum(r.agentic_recall_at_10 for r in records) / total_q if total_q else 0.0,
            "recall_at_20": sum(r.agentic_recall_at_20 for r in records) / total_q if total_q else 0.0,
            "mrr": sum(r.agentic_mrr for r in records) / total_q if total_q else 0.0,
            "ndcg_at_10": sum(r.agentic_ndcg_at_10 for r in records) / total_q if total_q else 0.0,
            "mean_latency_ms": overall_ag_lat_mean,
            "median_latency_ms": overall_ag_lat_med,
            "p95_latency_ms": overall_ag_lat_p95,
            "decomposition_rate": sum(1 for r in records if r.agentic_decision == "decompose_and_retrieve") / total_q if total_q else 0.0,
            "direct_retrieval_rate": sum(1 for r in records if r.agentic_decision == "direct_retrieval") / total_q if total_q else 0.0,
            "average_subqueries": sum(len(r.agentic_subqueries) for r in records) / total_q if total_q else 0.0,
            "average_hops": sum(r.agentic_hops for r in records) / total_q if total_q else 0.0,
            "corrective_retrieval_rate": sum(1 for r in records if r.agentic_hops > 1 or r.agentic_corrective_query is not None) / total_q if total_q else 0.0,
            "retrieval_calls_per_query": sum(r.agentic_retrieval_calls for r in records) / total_q if total_q else 0.0,
            "llm_calls_per_query": sum(r.agentic_llm_calls for r in records) / total_q if total_q else 0.0,
            "fallback_rate": sum(1 for r in records if r.agentic_is_fallback) / total_q if total_q else 0.0,
        }

        # 6. Specific Retrieval Trap & Recovery Metrics Calculation
        total_ad_misses = sum(1 for r in non_negative_records if r.adaptive_missed_required)
        total_ag_recoveries = sum(1 for r in non_negative_records if r.agentic_recovered_missed)
        total_uniq_chunks_recovered = sum(len(r.agentic_unique_relevant_chunks) for r in records)
        all_required_chunks_count = sum(len(r.required_chunk_ids) for r in non_negative_records)

        # Retrieval Trap Category specifically
        trap_records = [r for r in records if r.category == TrapBenchmarkCategory.RETRIEVAL_TRAP]
        trap_ad_misses = sum(1 for r in trap_records if r.adaptive_missed_required)
        trap_ag_recoveries = sum(1 for r in trap_records if r.agentic_recovered_missed)
        trap_recovery_rate = (trap_ag_recoveries / trap_ad_misses) if trap_ad_misses > 0 else 0.0

        # Corrective Hop Statistics
        corrective_records = [r for r in records if r.agentic_hops > 1 or r.agentic_corrective_query is not None]
        total_corrective_hops = len(corrective_records)
        valuable_corrective_hops = sum(1 for r in corrective_records if r.corrective_produced_new_evidence is True)
        false_corrective_hops = total_corrective_hops - valuable_corrective_hops

        trap_metrics = {
            "adaptive_miss_count": total_ad_misses,
            "adaptive_miss_rate": (total_ad_misses / total_non_neg) if total_non_neg else 0.0,
            "agentic_recovery_count": total_ag_recoveries,
            "agentic_recovery_rate": (total_ag_recoveries / total_ad_misses) if total_ad_misses else 0.0,
            "agentic_unique_evidence_count": total_uniq_chunks_recovered,
            "agentic_unique_evidence_percentage": (total_uniq_chunks_recovered / all_required_chunks_count * 100.0) if all_required_chunks_count else 0.0,
            "trap_queries_count": len(trap_records),
            "trap_adaptive_misses": trap_ad_misses,
            "trap_agentic_recoveries": trap_ag_recoveries,
            "trap_recovery_rate": trap_recovery_rate,
            "total_corrective_queries": total_corrective_hops,
            "valuable_corrective_queries": valuable_corrective_hops,
            "false_corrective_queries": false_corrective_hops,
            "false_corrective_retrieval_rate": (false_corrective_hops / total_corrective_hops) if total_corrective_hops else 0.0,
            "corrective_retrieval_value_rate": (valuable_corrective_hops / total_corrective_hops) if total_corrective_hops else 0.0,
        }

        overall_divergence: dict[str, int] = {}
        for div in DivergenceClassification:
            overall_divergence[div.value] = sum(1 for r in records if r.classification == div)

        return TrapBenchmarkReport(
            dataset_id=self.dataset.dataset_id,
            total_corpus_size=len(self.dataset.corpus),
            total_queries=total_q,
            overall_adaptive_metrics=overall_adaptive,
            overall_agentic_metrics=overall_agentic,
            trap_specific_metrics=trap_metrics,
            category_metrics=category_metrics_map,
            divergence_distribution=overall_divergence,
            query_records=records,
        )
