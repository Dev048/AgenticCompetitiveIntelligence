"""In-memory BM25 lexical index implementation for Retrieval Engine subsystem.

Uses rank-bm25 for BM25Okapi scoring with deterministic entity tokenization,
pre-ranking metadata filtering, and domain RetrievalResult construction.
"""

from collections.abc import Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.models import (
    DocumentChunk,
    VectorSearchFilter,
)
from competitive_intelligence.retrieval_engine.lexical.interfaces import (
    BaseLexicalIndex,
)
from competitive_intelligence.retrieval_engine.lexical.tokenizer import (
    LexicalTokenizer,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


class BM25Index(BaseLexicalIndex):
    """In-memory BM25 index wrapping rank-bm25 BM25Okapi.

    Responsibilities:
    - Tokenizes DocumentChunk contents using LexicalTokenizer.
    - Builds an internal BM25Okapi term-frequency inverted structure.
    - Computes BM25 query scores.
    - Applies metadata filters before final top-k truncation.
    - Preserves score and document provenance in RetrievalResult models.
    - Completely isolates rank_bm25 types from outer domain interfaces.
    """

    def __init__(self, tokenizer: LexicalTokenizer | None = None) -> None:
        """Initialize BM25Index.

        Args:
            tokenizer: Optional LexicalTokenizer instance. Defaults to new LexicalTokenizer.
        """
        self._tokenizer = tokenizer or LexicalTokenizer()
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[set[str]] = []
        self._bm25: Any | None = None

    @property
    def tokenizer(self) -> LexicalTokenizer:
        """Configured lexical tokenizer."""
        return self._tokenizer

    @property
    def chunk_count(self) -> int:
        """Number of document chunks currently indexed."""
        return len(self._chunks)

    def build_index(self, chunks: Sequence[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from a sequence of DocumentChunk objects.

        Args:
            chunks: Sequence of DocumentChunk instances. Inputs are not mutated.
        """
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            self._tokenized_corpus = []
            return

        tokenized_lists = [self._tokenizer.tokenize(chunk.content) for chunk in self._chunks]
        self._tokenized_corpus = [set(tokens) for tokens in tokenized_lists]

        try:
            from rank_bm25 import BM25Plus

            self._bm25 = BM25Plus(tokenized_lists)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize BM25Plus index: {exc}") from exc

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
    ) -> list[RetrievalResult]:
        """Search the BM25 index for terms matching the query.

        Args:
            query: Search query string.
            top_k: Maximum number of ranked results to return.
            filters: Optional metadata filters applied before top_k selection.

        Returns:
            A list of RetrievalResult objects ordered by descending BM25 score.

        Raises:
            ValueError: If query is empty/whitespace or top_k is non-positive.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace-only")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if not self._chunks or self._bm25 is None:
            return []

        query_tokens = self._tokenizer.tokenize(query.strip())
        if not query_tokens:
            return []

        query_set = set(query_tokens)
        raw_scores = self._bm25.get_scores(query_tokens)

        candidates: list[tuple[float, DocumentChunk]] = []
        for idx, chunk in enumerate(self._chunks):
            # Verify term overlap between query and document
            if not (query_set & self._tokenized_corpus[idx]):
                continue

            score = float(raw_scores[idx])

            if filters is not None and not self._matches_filter(chunk, filters):
                continue

            candidates.append((score, chunk))

        if not candidates:
            return []

        # Sort descending by BM25 score
        candidates.sort(key=lambda pair: pair[0], reverse=True)

        selected = candidates[:top_k]
        return [
            RetrievalResult.from_chunk(chunk, score=score)
            for score, chunk in selected
        ]

    def clear(self) -> None:
        """Clear all indexed documents and reset index state."""
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None

    def _matches_filter(
        self, chunk: DocumentChunk, filters: VectorSearchFilter
    ) -> bool:
        """Check if a chunk satisfies the specified metadata filter criteria."""
        if (
            filters.workspace_id is not None
            and chunk.metadata.workspace_id != filters.workspace_id
        ):
            return False
        if (
            filters.company_id is not None
            and chunk.metadata.company_id != filters.company_id
        ):
            return False
        if (
            filters.source_id is not None
            and chunk.metadata.source_id != filters.source_id
        ):
            return False
        if (
            filters.source_category is not None
            and chunk.metadata.source_category != filters.source_category
        ):
            return False
        if (
            filters.content_type is not None
            and chunk.metadata.custom_tags.get("content_type")
            != filters.content_type
        ):
            return False
        if filters.version_id is not None and chunk.version_id != filters.version_id:
            return False
        if filters.document_id is not None and chunk.document_id != filters.document_id:
            return False
        if (
            filters.parent_chunk_id is not None
            and chunk.parent_chunk_id != filters.parent_chunk_id
        ):
            return False
        if filters.chunk_level is not None and chunk.chunk_level != filters.chunk_level:
            return False

        return True
