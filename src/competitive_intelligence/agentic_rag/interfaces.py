"""Abstract interfaces for Agentic RAG subsystem.

Subsystem 4: Bounded Agentic RAG Retrieval Controller.
Defines interfaces for query analysis, evidence verification, and graph workflow orchestration.
"""

from abc import ABC, abstractmethod

from competitive_intelligence.agentic_rag.models import (
    AgenticRAGState,
    QueryAnalysis,
    VerificationAssessment,
)
from competitive_intelligence.retrieval_engine.models import (
    RetrievalQuery,
    RetrievalResult,
)


class BaseQueryAnalyzer(ABC):
    """Analyzes user query, decides retrieval strategy, and plans sub-queries."""

    @abstractmethod
    async def analyze_query(
        self,
        query: str,
        workspace_id: str,
        company_id: str | None = None,
    ) -> QueryAnalysis:
        """Decompose query and choose retrieval strategy."""
        pass


class BaseEvidenceVerifier(ABC):
    """Evaluates evidence sufficiency, detects gaps, and spots cross-source contradictions."""

    @abstractmethod
    async def verify(
        self,
        query: str,
        evidence: list[RetrievalResult],
    ) -> VerificationAssessment:
        """Evaluate if evidence is complete, identifying missing aspects and conflicts."""
        pass


class BaseAgentWorkflow(ABC):
    """LangGraph workflow runner for orchestrating the agentic RAG graph."""

    @abstractmethod
    async def run(
        self,
        query: RetrievalQuery,
        max_hops: int = 3,
    ) -> AgenticRAGState:
        """Execute state graph from query input to accumulated retrieval evidence."""
        pass
