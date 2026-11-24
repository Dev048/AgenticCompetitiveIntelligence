"""LangGraph state machine assembly for Agentic RAG retrieval workflow."""

from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from competitive_intelligence.agentic_rag.graph.state import AgenticGraphState
from competitive_intelligence.agentic_rag.interfaces import BaseAgentWorkflow
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.agentic_rag.models import AgenticRAGState
from competitive_intelligence.agentic_rag.nodes import (
    CorrectiveRetrievalNode,
    ExecuteSubqueriesNode,
    QueryAnalysisNode,
    VerifyEvidenceNode,
)
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.retrieval_engine.interfaces import BaseRetriever
from competitive_intelligence.retrieval_engine.models import RetrievalQuery

logger = get_logger(__name__)


def should_continue(state: AgenticGraphState) -> str:
    """Evaluate whether to terminate or trigger another corrective retrieval hop."""
    if state.get("is_fallback"):
        return END

    if state.get("termination_reason"):
        return END

    latest_verification = state.get("latest_verification")
    if latest_verification is not None and latest_verification.is_sufficient:
        return END

    hop_count = state.get("hop_count", 1)
    max_hops = state.get("max_hops", 3)
    if hop_count >= max_hops:
        return END

    return "corrective_retrieval"


def build_agentic_rag_graph(
    retriever: BaseRetriever,
    llm_client: BaseLLMClient,
    budget_tracker: LLMBudgetTracker | None = None,
) -> Any:
    """Assemble and compile the bounded LangGraph state machine.
    
    Graph Topology:
    START -> analyze_query -> execute_subqueries -> verify_evidence
             ^                                           |
             |---------------- corrective_retrieval <----+ (if insufficient and hops < max_hops)
                                                         |
                                                        END (if sufficient or hops >= max_hops)
    """
    workflow = StateGraph(AgenticGraphState)

    # Initialize nodes
    workflow.add_node("analyze_query", QueryAnalysisNode(llm_client, budget_tracker))
    workflow.add_node("execute_subqueries", ExecuteSubqueriesNode(retriever, budget_tracker))
    workflow.add_node("verify_evidence", VerifyEvidenceNode(llm_client, budget_tracker))
    workflow.add_node("corrective_retrieval", CorrectiveRetrievalNode())

    # Deterministic graph edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "execute_subqueries")
    workflow.add_edge("execute_subqueries", "verify_evidence")

    # Conditional branching after verification
    workflow.add_conditional_edges(
        "verify_evidence",
        should_continue,
        {
            "corrective_retrieval": "corrective_retrieval",
            END: END,
        },
    )

    # Loop back from corrective retrieval to execute_subqueries
    workflow.add_edge("corrective_retrieval", "execute_subqueries")

    return workflow.compile()


class AgenticRAGWorkflow(BaseAgentWorkflow):
    """LangGraph-backed runner executing the bounded Agentic RAG graph."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: BaseLLMClient,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.budget_tracker = budget_tracker
        self.graph = build_agentic_rag_graph(retriever, llm_client, budget_tracker)

    async def run(
        self,
        query: RetrievalQuery,
        max_hops: int = 3,
    ) -> AgenticRAGState:
        """Run the LangGraph state machine from query input to accumulated retrieval evidence."""
        task_id = str(uuid4())
        workspace_id = query.metadata_filter.workspace_id
        company_id = query.metadata_filter.company_ids[0] if query.metadata_filter.company_ids else None

        initial_state: AgenticGraphState = {
            "task_id": task_id,
            "workspace_id": workspace_id,
            "company_id": company_id,
            "original_query": query.query_text,
            "metadata_filter": query.metadata_filter,
            "top_k": query.top_k,
            "max_hops": max_hops,
            "hop_count": 0,
            "accumulated_results": [],
            "verification_assessments": [],
            "sub_queries": [],
            "current_sub_queries": [],
            "missing_aspects": [],
            "is_fallback": False,
            "metadata": {},
        }

        final_state: dict[str, Any] = await self.graph.ainvoke(initial_state)

        return AgenticRAGState(
            task_id=final_state.get("task_id", task_id),
            workspace_id=workspace_id,
            company_id=company_id,
            original_query=query.query_text,
            metadata_filter=query.metadata_filter,
            query_analysis=final_state.get("query_analysis"),
            accumulated_evidence=final_state.get("accumulated_results", []),
            hops_executed=final_state.get("hop_count", 0),
            max_hops=max_hops,
            verifications=final_state.get("verification_assessments", []),
            missing_aspects=final_state.get("missing_aspects", []),
            corrective_query=final_state.get("corrective_query"),
            termination_reason=final_state.get("termination_reason"),
            is_fallback=final_state.get("is_fallback", False),
            metadata=final_state.get("metadata", {}),
        )
