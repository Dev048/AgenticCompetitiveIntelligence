"""Abstract interfaces for lexical index components in Retrieval Engine subsystem."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from competitive_intelligence.knowledge_pipeline.models import (
    DocumentChunk,
    VectorSearchFilter,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


class BaseLexicalIndex(ABC):
    """Abstract interface for in-memory or persisted lexical search indexes (e.g. BM25)."""

    @abstractmethod
    def build_index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Build or rebuild the lexical index from a sequence of document chunks."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Execute lexical term search and return ranked RetrievalResult objects."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents and reset index state."""
        pass

    @property
    @abstractmethod
    def chunk_count(self) -> int:
        """Number of document chunks currently indexed."""
        pass
