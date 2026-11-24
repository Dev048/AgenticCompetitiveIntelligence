"""Semantic chunker implementation for Knowledge Pipeline subsystem using LlamaIndex.

Splits DocumentVersion content into semantic chunks based on embedding similarity
transitions using LlamaIndex SemanticSplitterNodeParser and our BaseEmbeddingProvider abstraction.
"""

from collections.abc import Sequence
from typing import Any
import uuid

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentChunker,
    BaseEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    ChunkingParameters,
    ChunkingStrategy,
    DocumentChunk,
    DocumentVersion,
)
from competitive_intelligence.knowledge_pipeline.normalizer import compute_content_hash
from competitive_intelligence.source_acquisition.models import Source

DEFAULT_SEMANTIC_CHUNK_SIZE = 800
DEFAULT_SEMANTIC_CHUNK_OVERLAP = 100
DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD = 95
DEFAULT_BUFFER_SIZE = 1


class _LlamaIndexEmbeddingAdapter(BaseEmbedding):
    """Adapter bridging our domain BaseEmbeddingProvider to LlamaIndex BaseEmbedding."""

    def __init__(self, provider: BaseEmbeddingProvider) -> None:
        super().__init__()
        self._provider = provider

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._provider.embed_query(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._provider.embed_query(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._provider.aembed_query(query)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed_texts(texts)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._provider.aembed_texts(texts)

    def get_text_embedding_batch(
        self, texts: Sequence[str], show_progress: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        return self._provider.embed_texts(texts)


def derive_semantic_chunk_id(
    version_id: str, chunk_index: int, chunk_size: int, chunk_overlap: int
) -> str:
    """Derive deterministic chunk UUID for semantic chunking using UUID v5."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"urn:ci:chunk:semantic:{version_id}:{chunk_index}:{chunk_size}:{chunk_overlap}",
        )
    )


class SemanticChunker(BaseDocumentChunker):
    """Semantic boundary chunker using LlamaIndex SemanticSplitterNodeParser.

    Responsibilities:
    - Receives BaseEmbeddingProvider via dependency injection.
    - Adapts provider to LlamaIndex embedding interface without importing concrete models.
    - Evaluates semantic distance between consecutive sentence groups.
    - Identifies topic transition breakpoints based on percentile distance thresholds.
    - Enforces size-safety via recursive sentence splitting for oversized semantic blocks.
    - Preserves sequential document order and full provenance metadata.
    - Strictly isolates LlamaIndex types from domain outputs.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        default_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
        default_chunk_overlap: int = DEFAULT_SEMANTIC_CHUNK_OVERLAP,
        default_breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE_THRESHOLD,
        default_buffer_size: int = DEFAULT_BUFFER_SIZE,
    ) -> None:
        if embedding_provider is None or not isinstance(embedding_provider, BaseEmbeddingProvider):
            raise TypeError("SemanticChunker requires a valid BaseEmbeddingProvider instance")

        self.embedding_provider = embedding_provider
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.default_breakpoint_percentile_threshold = default_breakpoint_percentile_threshold
        self.default_buffer_size = default_buffer_size
        self._embed_adapter = _LlamaIndexEmbeddingAdapter(embedding_provider)

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content across semantic topic boundaries.

        Args:
            version: The DocumentVersion to chunk.
            parameters: Optional chunking parameters (chunk_size, chunk_overlap, custom_params).
            source: Optional Source domain model for enriched provenance metadata.

        Returns:
            A list of DocumentChunk instances.

        Raises:
            ValueError: If version.content is empty or whitespace-only.
        """
        if not version.content or not version.content.strip():
            raise ValueError("Cannot chunk DocumentVersion with empty or whitespace-only content")

        chunk_size = (
            parameters.chunk_size
            if parameters is not None and parameters.chunk_size is not None
            else self.default_chunk_size
        )
        chunk_overlap = (
            parameters.chunk_overlap
            if parameters is not None and parameters.chunk_overlap is not None
            else self.default_chunk_overlap
        )

        threshold_val = (
            parameters.custom_params.get("breakpoint_percentile_threshold")
            if parameters is not None and "breakpoint_percentile_threshold" in parameters.custom_params
            else (
                parameters.custom_params.get("semantic_threshold")
                if parameters is not None and "semantic_threshold" in parameters.custom_params
                else self.default_breakpoint_percentile_threshold
            )
        )
        breakpoint_percentile_threshold: int = int(threshold_val) if threshold_val is not None else self.default_breakpoint_percentile_threshold

        buffer_size_val = (
            parameters.custom_params.get("buffer_size")
            if parameters is not None and "buffer_size" in parameters.custom_params
            else self.default_buffer_size
        )
        buffer_size: int = int(buffer_size_val) if buffer_size_val is not None else self.default_buffer_size

        # 1. Use LlamaIndex SemanticSplitterNodeParser
        splitter = SemanticSplitterNodeParser(
            embed_model=self._embed_adapter,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size,
        )

        llama_doc = LlamaDocument(text=version.content)
        nodes = splitter.get_nodes_from_documents([llama_doc])

        if not nodes:
            raw_splits = [version.content]
        else:
            raw_splits = [n.get_content().strip() for n in nodes if n.get_content().strip()]

        if not raw_splits:
            raw_splits = [version.content]

        # 2. Size safety: split any oversized semantic chunk using SentenceSplitter
        sentence_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        final_splits: list[str] = []
        for segment in raw_splits:
            if len(segment) > chunk_size:
                sub_chunks = sentence_splitter.split_text(segment)
                if sub_chunks:
                    final_splits.extend(sub_chunks)
                else:
                    final_splits.append(segment)
            else:
                final_splits.append(segment)

        total_chunks = len(final_splits)
        chunks: list[DocumentChunk] = []
        search_start = 0

        for i, split_text in enumerate(final_splits):
            chunk_id = derive_semantic_chunk_id(
                version_id=version.version_id,
                chunk_index=i,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Character offset calculation against original text
            start_char = version.content.find(split_text, search_start)
            if start_char == -1:
                start_char = version.content.find(split_text)
                if start_char == -1:
                    start_char = search_start
            end_char = start_char + len(split_text)
            search_start = max(0, start_char)

            metadata = self._build_chunk_metadata(
                version=version,
                source=source,
                chunk_index=i,
                total_chunks=total_chunks,
                start_char=start_char,
                end_char=end_char,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                buffer_size=buffer_size,
            )

            content_hash = compute_content_hash(split_text)

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=version.document_id,
                version_id=version.version_id,
                content=split_text,
                chunk_index=i,
                metadata=metadata,
                content_hash=content_hash,
                parent_chunk_id=None,
                chunk_level=0,
                created_at=version.acquired_at,
            )
            chunks.append(chunk)

        return chunks

    def _build_chunk_metadata(
        self,
        version: DocumentVersion,
        source: Source | None,
        chunk_index: int,
        total_chunks: int,
        start_char: int,
        end_char: int,
        chunk_size: int,
        chunk_overlap: int,
        breakpoint_percentile_threshold: int,
        buffer_size: int,
    ) -> ChunkMetadata:
        """Construct provenance ChunkMetadata for a semantic chunk."""
        if source is not None:
            source_id = source.identity.source_id
            company_id = source.identity.company_id
            workspace_id = source.identity.workspace_id
            source_url = source.location.url_or_identifier
            source_category = source.classification.category.value
        else:
            source_id = str(version.metadata.get("source_id", f"src_{version.document_id}"))
            company_id = str(version.metadata.get("company_id", "unknown_company"))
            workspace_id = str(version.metadata.get("workspace_id", "default_workspace"))
            source_url = str(version.metadata.get("source_url", "https://example.com/document"))
            source_category = str(version.metadata.get("source_category", "general"))

        custom_tags: dict[str, Any] = dict(version.metadata)
        custom_tags["strategy"] = ChunkingStrategy.SEMANTIC.value
        custom_tags["chunk_size"] = chunk_size
        custom_tags["chunk_overlap"] = chunk_overlap
        custom_tags["breakpoint_percentile_threshold"] = breakpoint_percentile_threshold
        custom_tags["buffer_size"] = buffer_size

        return ChunkMetadata(
            source_id=source_id,
            document_id=version.document_id,
            version_id=version.version_id,
            company_id=company_id,
            workspace_id=workspace_id,
            source_url=source_url,
            source_category=source_category,
            collected_at=version.acquired_at,
            start_char=start_char,
            end_char=end_char,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            language="en",
            custom_tags=custom_tags,
        )
