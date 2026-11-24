"""Evidence verification node for evaluating sufficiency, gaps, and grounded contradictions."""

from typing import Any

from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.models import (
    ContradictionClaim,
    SufficiencyDecision,
    VerificationAssessment,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.retrieval_engine.models import RetrievalResult

logger = get_logger(__name__)

VERIFICATION_SYSTEM_PROMPT = """You are an evidence verification engine for competitive intelligence retrieval.
Evaluate the accumulated retrieved evidence chunks against the user's original information need.

Guidelines:
1. Determine if the evidence is SUFFICIENT to address the question factually.
2. If key aspects are missing, set is_sufficient=false, decision='insufficient_need_hop', list missing aspects, and formulate a targeted 'recommended_next_query'.
3. If you identify conflicting observations across different sources/chunks:
   - Record the conflicting observation in 'detected_contradictions'.
   - You MUST cite ONLY the exact chunk_ids where the conflicting data appears.
   - Do NOT declare one source factually false merely because another source differs; represent them as unresolved differing observations.
   - Set decision='contradictory_need_verification' if the conflict requires additional context.
4. Never hallucinate or invent chunk IDs or factual claims not present in the provided evidence.
"""


class VerifyEvidenceNode:
    """Node that verifies sufficiency and cross-source consistency of accumulated evidence."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker

    async def __call__(self, state: AgenticGraphState) -> dict[str, Any]:
        """Execute evidence verification node."""
        # Immediately short-circuit if terminal condition or budget exhaustion is already active
        if state.get("termination_reason") or state.get("is_fallback"):
            logger.info(
                "Terminal execution condition active; short-circuiting evidence verification",
                termination_reason=state.get("termination_reason"),
                is_fallback=state.get("is_fallback"),
            )
            return {}

        original_query = state.get("original_query", "")
        accumulated_results: list[RetrievalResult] = state.get("accumulated_results", [])
        hop_count = state.get("hop_count", 1)

        valid_chunk_ids = {r.chunk_id for r in accumulated_results}

        # If zero evidence was retrieved, automatically mark insufficient without wasting LLM call
        if not accumulated_results:
            logger.info("Zero evidence chunks retrieved; marking insufficient", hop=hop_count)
            fallback_assessment = VerificationAssessment(
                decision=SufficiencyDecision.INSUFFICIENT_NEED_HOP,
                is_sufficient=False,
                reasoning="No evidence chunks were retrieved during this hop.",
                detected_contradictions=[],
                missing_aspects=["All aspects missing: no documents found."],
                cited_chunk_ids=[],
                recommended_next_query=original_query,
            )
            history = list(state.get("verification_assessments", []))
            history.append(fallback_assessment)
            return {
                "latest_verification": fallback_assessment,
                "verification_assessments": history,
                "missing_aspects": fallback_assessment.missing_aspects,
            }

        # Format evidence snippets for LLM review (capped to top 10 chunks to preserve token budget)
        evidence_lines: list[str] = []
        for i, res in enumerate(accumulated_results[:10], 1):
            snippet = res.text[:250].replace("\n", " ")
            evidence_lines.append(
                f"[{i}] chunk_id='{res.chunk_id}' company='{res.company_id or 'unknown'}' "
                f"source='{res.source_category or 'general'}' score={res.score:.2f}:\n    \"{snippet}\""
            )
        evidence_block = "\n".join(evidence_lines)

        user_prompt = (
            f"Original Query: {original_query}\n"
            f"Current Hop: {hop_count}\n"
            f"Accumulated Evidence Chunks ({len(accumulated_results)} total):\n"
            f"{evidence_block}"
        )

        assessment = await self.llm_client.generate_structured(
            prompt=user_prompt,
            response_schema=VerificationAssessment,
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            budget_tracker=self.budget_tracker,
        )

        # Sanitize cited and contradictory chunk IDs against actual retrieved chunks
        # Contradictions require at least TWO distinct valid retrieved chunk IDs
        sanitized_contradictions: list[ContradictionClaim] = []
        for claim in assessment.detected_contradictions:
            valid_ids = list(dict.fromkeys(cid for cid in claim.conflicting_chunk_ids if cid in valid_chunk_ids))
            if len(valid_ids) >= 2:
                sanitized_contradictions.append(
                    ContradictionClaim(
                        claim_description=claim.claim_description,
                        conflicting_chunk_ids=valid_ids,
                        rationale=claim.rationale,
                    )
                )

        sanitized_cited_ids = list(dict.fromkeys(cid for cid in assessment.cited_chunk_ids if cid in valid_chunk_ids))

        assessment.detected_contradictions = sanitized_contradictions
        assessment.cited_chunk_ids = sanitized_cited_ids

        logger.info(
            "Evidence verification completed",
            decision=assessment.decision.value,
            is_sufficient=assessment.is_sufficient,
            contradictions_count=len(assessment.detected_contradictions),
            hop=hop_count,
        )

        history = list(state.get("verification_assessments", []))
        history.append(assessment)

        return {
            "latest_verification": assessment,
            "verification_assessments": history,
            "missing_aspects": assessment.missing_aspects,
        }
