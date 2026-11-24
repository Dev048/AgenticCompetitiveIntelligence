"""Embedding service for Knowledge Pipeline subsystem.

Coordinates chunk eligibility filtering, batching, vector generation via BaseEmbeddingProvider,
shape validation, and mapping to canonical EmbeddingRecord domain models.
"""

from collections.abc import Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    DocumentChunk,
    EmbeddingRecord,
)

DEFAULT_BATCH_SIZE = 32


class EmbeddingService:
    """Application-level service that coordinates chunk embedding generation.

    Responsibilities:
    - Filters which chunks are eligible for embedding (e.g. child chunks vs parents in hierarchies).
    - Splits eligible chunks into configurable batches.
    - Calls the injected BaseEmbeddingProvider batch methods.
    - Validates that provider output count and dimensions match expected shapes.
    - Maps vectors back to input chunks and attaches model metadata.
    - Leaves input DocumentChunk models completely unmodified.
    - Strictly decouples application pipelines from concrete embedding models.
    """

    def __init__(
        self,
        provider: BaseEmbeddingProvider,
        default_batch_size: int = DEFAULT_BATCH_SIZE,
        embed_parents_with_children: bool = False,
    ) -> None:
        if provider is None or not isinstance(provider, BaseEmbeddingProvider):
            raise TypeError("EmbeddingService requires a valid BaseEmbeddingProvider instance")
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be a positive integer")

        self.provider = provider
        self.default_batch_size = default_batch_size
        self.embed_parents_with_children = embed_parents_with_children

    def filter_eligible_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embed_parents_with_children: bool | None = None,
    ) -> list[DocumentChunk]:
        """Filter chunks based on hierarchical parent-child eligibility rules.

        Rules:
        - If embed_parents_with_children is True: all input chunks are eligible.
        - If embed_parents_with_children is False (default):
          - Any chunk whose chunk_id is referenced as a parent_chunk_id by a child in the input set is skipped.
          - Child chunks (chunk_level > 0) and standalone chunks (chunk_level == 0 with no children) are eligible.

        Args:
            chunks: Sequence of DocumentChunk candidates.
            embed_parents_with_children: Optional override for parent embedding policy.

        Returns:
            A list of DocumentChunk instances that are eligible for vector embedding.
        """
        if not chunks:
            return []

        should_embed_parents = (
            embed_parents_with_children
            if embed_parents_with_children is not None
            else self.embed_parents_with_children
        )

        if should_embed_parents:
            return list(chunks)

        # Identify all parent IDs that have child references in this chunk set
        parent_ids_with_children = {
            c.parent_chunk_id for c in chunks if c.parent_chunk_id is not None
        }

        # Filter out chunks that serve as parent containers for existing children
        return [c for c in chunks if c.chunk_id not in parent_ids_with_children]

    def generate_embeddings(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int | None = None,
        embed_parents_with_children: bool | None = None,
    ) -> list[EmbeddingRecord]:
        """Generate EmbeddingRecords for eligible chunks using synchronous batching.

        Args:
            chunks: Sequence of DocumentChunk instances.
            batch_size: Optional batch size override (must be positive).
            embed_parents_with_children: Optional override for parent embedding policy.

        Returns:
            A list of EmbeddingRecord instances corresponding to eligible chunks.

        Raises:
            ValueError: If batch_size is non-positive or provider outputs fail validation.
        """
        if not chunks:
            return []

        actual_batch_size = batch_size if batch_size is not None else self.default_batch_size
        if actual_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        eligible_chunks = self.filter_eligible_chunks(
            chunks=chunks,
            embed_parents_with_children=embed_parents_with_children,
        )

        if not eligible_chunks:
            return []

        records: list[EmbeddingRecord] = []

        # Process in batches
        for i in range(0, len(eligible_chunks), actual_batch_size):
            batch = eligible_chunks[i : i + actual_batch_size]
            texts = [c.content for c in batch]

            vectors = self.provider.embed_texts(texts)

            # Validate provider output count
            if len(vectors) != len(batch):
                raise ValueError(
                    f"Provider output count mismatch: expected {len(batch)} vectors for batch, got {len(vectors)}"
                )

            # Validate vector dimensions and construct records
            for chunk, vector in zip(batch, vectors, strict=True):
                self._validate_vector_shape(vector)
                record = self._build_embedding_record(chunk=chunk, vector=vector)
                records.append(record)

        return records

    async def agenerate_embeddings(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int | None = None,
        embed_parents_with_children: bool | None = None,
    ) -> list[EmbeddingRecord]:
        """Asynchronously generate EmbeddingRecords for eligible chunks.

        Args:
            chunks: Sequence of DocumentChunk instances.
            batch_size: Optional batch size override.
            embed_parents_with_children: Optional override for parent embedding policy.

        Returns:
            A list of EmbeddingRecord instances corresponding to eligible chunks.
        """
        if not chunks:
            return []

        actual_batch_size = batch_size if batch_size is not None else self.default_batch_size
        if actual_batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        eligible_chunks = self.filter_eligible_chunks(
            chunks=chunks,
            embed_parents_with_children=embed_parents_with_children,
        )

        if not eligible_chunks:
            return []

        records: list[EmbeddingRecord] = []

        for i in range(0, len(eligible_chunks), actual_batch_size):
            batch = eligible_chunks[i : i + actual_batch_size]
            texts = [c.content for c in batch]

            vectors = await self.provider.aembed_texts(texts)

            if len(vectors) != len(batch):
                raise ValueError(
                    f"Provider output count mismatch: expected {len(batch)} vectors for batch, got {len(vectors)}"
                )

            for chunk, vector in zip(batch, vectors, strict=True):
                self._validate_vector_shape(vector)
                record = self._build_embedding_record(chunk=chunk, vector=vector)
                records.append(record)

        return records

    def _validate_vector_shape(self, vector: list[float]) -> None:
        """Validate that an individual vector is non-empty and matches expected provider dimensions."""
        if not vector:
            raise ValueError("Provider returned empty or null vector")
        if self.provider.dimensions is not None and len(vector) != self.provider.dimensions:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.provider.dimensions}, got {len(vector)}"
            )

    def _build_embedding_record(
        self, chunk: DocumentChunk, vector: list[float]
    ) -> EmbeddingRecord:
        """Construct a provenance-rich EmbeddingRecord mapping vector to chunk."""
        metadata: dict[str, Any] = {
            "chunk_index": chunk.chunk_index,
            "chunk_level": chunk.chunk_level,
            "parent_chunk_id": chunk.parent_chunk_id,
            "source_id": chunk.metadata.source_id,
            "company_id": chunk.metadata.company_id,
            "workspace_id": chunk.metadata.workspace_id,
            "source_url": chunk.metadata.source_url,
            "source_category": chunk.metadata.source_category,
        }

        return EmbeddingRecord(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            vector=vector,
            model_name=self.provider.model_name,
            model_version=self.provider.model_version,
            dimensions=len(vector),
            metadata=metadata,
        )
