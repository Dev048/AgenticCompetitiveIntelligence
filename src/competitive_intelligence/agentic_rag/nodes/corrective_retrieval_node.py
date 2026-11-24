"""Corrective retrieval node for planning subsequent targeted retrieval hops."""

from typing import Any

from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.models import SubQueryPlan
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)


class CorrectiveRetrievalNode:
    """Node that formulates targeted corrective queries when verification indicates missing evidence."""

    async def __call__(self, state: AgenticGraphState) -> dict[str, Any]:
        """Formulate next retrieval hop query."""
        original_query = state.get("original_query", "")
        hop_count = state.get("hop_count", 1)
        max_hops = state.get("max_hops", 3)
        latest_verification = state.get("latest_verification")
        company_id = state.get("company_id")

        next_hop = hop_count + 1

        if next_hop > max_hops:
            logger.info("Max hops reached; terminating corrective retrieval", hop=hop_count, max_hops=max_hops)
            return {
                "termination_reason": "max_hops_reached",
            }

        # Resolve corrective query string
        next_query_text = ""
        if latest_verification and latest_verification.recommended_next_query:
            next_query_text = latest_verification.recommended_next_query.strip()

        if not next_query_text:
            missing = state.get("missing_aspects", [])
            if missing:
                aspects_str = " ".join(missing[:2])
                next_query_text = f"{original_query} {aspects_str}".strip()
            else:
                next_query_text = original_query

        corrective_plan = SubQueryPlan(
            sub_query=next_query_text,
            target_company_ids=[company_id] if company_id else [],
            rationale=f"Corrective hop {next_hop} targeting missing evidence.",
        )

        logger.info(
            "Planned corrective retrieval hop",
            hop=next_hop,
            max_hops=max_hops,
            corrective_query=next_query_text,
        )

        return {
            "hop_count": next_hop,
            "current_sub_queries": [corrective_plan],
            "corrective_query": next_query_text,
        }
