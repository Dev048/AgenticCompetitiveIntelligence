"""Data models for Agentic RAG evaluation and benchmarking."""

from pydantic import BaseModel, Field


class AgenticBenchmarkMetrics(BaseModel):
    """Metrics produced during Agentic RAG evaluation."""

    retriever_name: str
    sample_size: int
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    average_latency_ms: float = 0.0
    retrieval_calls_per_query: float = 0.0
    llm_calls_per_query: float = 0.0
    fallback_rate: float = 0.0


class AgenticComparativeReport(BaseModel):
    """Side-by-side comparison between AdaptiveRetriever and AgenticRetriever."""

    dataset_id: str
    adaptive_metrics: AgenticBenchmarkMetrics
    agentic_metrics: AgenticBenchmarkMetrics
