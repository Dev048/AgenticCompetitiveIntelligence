from pydantic import BaseModel, Field

from competitive_intelligence.knowledge_pipeline.models import DocumentChunk
from competitive_intelligence.retrieval_engine.models import MetadataFilter


class EvaluationQuery(BaseModel):
    """A benchmark query with labeled ground-truth relevant chunks."""

    query_id: str
    query_text: str
    relevant_chunk_ids: dict[str, int] = Field(default_factory=dict)
    metadata_filter: MetadataFilter | None = None
    notes: str | None = None


class EvaluationDataset(BaseModel):
    """A self-contained dataset for retrieval evaluation."""

    dataset_id: str
    queries: list[EvaluationQuery]
    corpus: list[DocumentChunk]
