"""Query analysis and decomposition node for Agentic RAG."""

from typing import Any

from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.models import (
    QueryAnalysis,
    RetrievalDecision,
    SubQueryPlan,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger

logger = get_logger(__name__)

QUERY_ANALYSIS_SYSTEM_PROMPT = """You are a retrieval query analyzer for a competitive intelligence system.
Analyze the user's information need and determine if it can be satisfied with a single direct retrieval or requires decomposition into targeted sub-queries.

Guidelines:
1. For focused, single-topic, or specific lookup questions, select 'direct_retrieval' and generate exactly one sub-query.
2. For comparative, multi-part, or cross-entity questions (e.g. comparing two competitors, checking pricing and features together), select 'decompose_and_retrieve' and generate 2 to 3 targeted, specific sub-queries.
3. Every sub-query must be clear, keyword-rich, and targeted at gathering factual evidence.
4. Do NOT attempt to answer the question directly. Your role is strictly query planning for evidence retrieval.
"""


class QueryAnalysisNode:
    """Node that analyzes the original query and formulates sub-query retrieval plans."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker

    async def __call__(self, state: AgenticGraphState) -> dict[str, Any]:
        """Execute query analysis node."""
        original_query = state.get("original_query", "")
        company_id = state.get("company_id")
        workspace_id = state.get("workspace_id", "default")

        user_prompt = f"Original Query: {original_query}"
        if company_id:
            user_prompt += f"\nTarget Company ID: {company_id}"

        analysis = await self.llm_client.generate_structured(
            prompt=user_prompt,
            response_schema=QueryAnalysis,
            system_prompt=QUERY_ANALYSIS_SYSTEM_PROMPT,
            budget_tracker=self.budget_tracker,
        )

        # Guard: ensure at least one sub-query exists
        if not analysis.sub_queries:
            analysis.sub_queries = [
                SubQueryPlan(
                    sub_query=original_query,
                    target_company_ids=[company_id] if company_id else [],
                    rationale="Default single retrieval query",
                )
            ]

        logger.info(
            "Query analysis completed",
            decision=analysis.decision.value,
            sub_queries_count=len(analysis.sub_queries),
            hop=1,
        )

        return {
            "query_analysis": analysis,
            "sub_queries": analysis.sub_queries,
            "current_sub_queries": analysis.sub_queries,
            "hop_count": 1,
        }
