"""Unified chunking service for Knowledge Pipeline subsystem.

Coordinates document characteristic derivation, policy resolution via
ChunkingPolicyResolver, and dynamic dispatch across concrete chunking engines.
"""

import re
from typing import Any

from competitive_intelligence.knowledge_pipeline.chunkers import (
    FixedChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    WholeDocumentChunker,
)
from competitive_intelligence.knowledge_pipeline.chunking_policy import (
    ChunkingPolicyResolver,
)
from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseChunkingPolicyResolver,
    BaseDocumentChunker,
    BaseEmbeddingProvider,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkingParameters,
    ChunkingPolicy,
    ChunkingStrategy,
    DocumentCharacteristics,
    DocumentChunk,
    DocumentVersion,
    StructuralRichness,
)
from competitive_intelligence.source_acquisition.models import Source


class ChunkingService:
    """Application-level chunking service providing unified policy resolution and chunker dispatch.

    Responsibilities:
    - Derives multi-signal DocumentCharacteristics from DocumentVersion content and Source metadata.
    - Resolves optimal ChunkingPolicy using the injected BaseChunkingPolicyResolver.
    - Dynamically delegates to the appropriate chunking engine (WholeDocument, Recursive, ParentChild, Semantic, Fixed).
    - Preserves end-to-end chunk provenance, deterministic UUID chunk IDs, and hierarchy metadata.
    - Strictly isolates downstream callers from individual chunker instantiation and parameter management.
    """

    def __init__(
        self,
        policy_resolver: BaseChunkingPolicyResolver | None = None,
        resolver: BaseChunkingPolicyResolver | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        chunkers: dict[ChunkingStrategy, BaseDocumentChunker] | None = None,
    ) -> None:
        self.policy_resolver = policy_resolver or resolver or ChunkingPolicyResolver()
        self.embedding_provider = embedding_provider

        # Initialize chunker registry
        if chunkers is not None:
            self._chunkers = dict(chunkers)
        else:
            self._chunkers = {
                ChunkingStrategy.WHOLE_DOCUMENT: WholeDocumentChunker(),
                ChunkingStrategy.FIXED: FixedChunker(),
                ChunkingStrategy.RECURSIVE: RecursiveChunker(),
                ChunkingStrategy.RECURSIVE_CHARACTER: RecursiveChunker(),
                ChunkingStrategy.MARKDOWN_HEADER: RecursiveChunker(),
                ChunkingStrategy.SENTENCE_WINDOW: FixedChunker(),
                ChunkingStrategy.PARENT_CHILD: ParentChildChunker(),
            }
            if embedding_provider is not None:
                self._chunkers[ChunkingStrategy.SEMANTIC] = SemanticChunker(
                    embedding_provider=embedding_provider
                )

    def derive_characteristics(
        self,
        version: DocumentVersion,
        source: Source | None = None,
    ) -> DocumentCharacteristics:
        """Analyze DocumentVersion and Source to derive multi-signal DocumentCharacteristics."""
        content = version.content or ""
        char_count = len(content)

        # 1. Source category & content type
        source_category: str | None = None
        content_type: str | None = None

        if source is not None and hasattr(source, "classification"):
            if source.classification.category:
                source_category = source.classification.category.value
            if source.classification.content_type:
                content_type = source.classification.content_type.value
        else:
            source_category = str(version.metadata.get("source_category")) if version.metadata.get("source_category") else None
            content_type = str(version.metadata.get("content_type")) if version.metadata.get("content_type") else None

        # 2. Structural richness analysis
        has_headers = len(re.findall(r"^#{1,6}\s", content, re.MULTILINE))
        has_tables = "|" in content and "\n|-" in content
        has_code_blocks = "```" in content
        has_lists = len(re.findall(r"^[\s]*[-*+]\s|^[\s]*\d+\.\s", content, re.MULTILINE))

        if has_code_blocks or has_tables or has_headers >= 3:
            structural_richness = StructuralRichness.HIGH
        elif has_headers >= 1 or has_lists >= 3:
            structural_richness = StructuralRichness.MEDIUM
        else:
            structural_richness = StructuralRichness.LOW

        # 3. Semantic transitions indicator
        has_semantic_transitions = bool(
            version.metadata.get("has_semantic_transitions", False)
            or version.metadata.get("semantic_transitions_detected", False)
        )

        return DocumentCharacteristics(
            source_category=source_category,
            content_type=content_type,
            char_count=char_count,
            structural_richness=structural_richness,
            has_semantic_transitions=has_semantic_transitions,
        )

    def resolve_policy(
        self,
        version: DocumentVersion,
        source: Source | None = None,
        characteristics: DocumentCharacteristics | None = None,
    ) -> ChunkingPolicy:
        """Resolve the optimal ChunkingPolicy for a DocumentVersion."""
        chars = characteristics or self.derive_characteristics(version, source)
        return self.policy_resolver.resolve_policy(chars)

    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
        characteristics: DocumentCharacteristics | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content into DocumentChunks according to resolved policy.

        Args:
            version: The immutable DocumentVersion snapshot to chunk.
            parameters: Optional execution parameter overrides.
            source: Optional Source domain model providing provenance and classification.
            characteristics: Optional pre-computed DocumentCharacteristics.

        Returns:
            A list of normalized DocumentChunk instances with preserved provenance.

        Raises:
            ValueError: If version.content is empty or whitespace-only.
        """
        if not version.content or not version.content.strip():
            raise ValueError("Cannot chunk DocumentVersion with empty or whitespace-only content")

        # 1. Derive characteristics and resolve policy
        chars = characteristics or self.derive_characteristics(version, source)
        policy = self.policy_resolver.resolve_policy(chars)

        # 2. Merge policy parameters with caller overrides if provided
        effective_params = self._merge_parameters(policy.parameters, parameters)

        # 3. Select chunker by resolved strategy
        chunker = self._get_chunker(policy.strategy)

        # 4. Execute chunking
        return chunker.chunk(
            version=version,
            parameters=effective_params,
            source=source,
        )

    def _get_chunker(self, strategy: ChunkingStrategy) -> BaseDocumentChunker:
        """Select concrete chunker for strategy, falling back safely if needed."""
        if strategy in self._chunkers:
            return self._chunkers[strategy]

        if strategy == ChunkingStrategy.SEMANTIC:
            # If SemanticChunker not available due to missing embedding provider, fall back to RecursiveChunker
            if self.embedding_provider is not None:
                chunker = SemanticChunker(embedding_provider=self.embedding_provider)
                self._chunkers[ChunkingStrategy.SEMANTIC] = chunker
                return chunker
            return self._chunkers.get(ChunkingStrategy.RECURSIVE, RecursiveChunker())

        return self._chunkers.get(ChunkingStrategy.FIXED, FixedChunker())

    def _merge_parameters(
        self,
        policy_params: ChunkingParameters,
        override_params: ChunkingParameters | None,
    ) -> ChunkingParameters:
        """Merge base policy parameters with optional caller-provided overrides."""
        if override_params is None:
            return policy_params

        custom = dict(policy_params.custom_params)
        custom.update(override_params.custom_params)

        return ChunkingParameters(
            chunk_size=override_params.chunk_size if override_params.chunk_size is not None else policy_params.chunk_size,
            chunk_overlap=override_params.chunk_overlap if override_params.chunk_overlap is not None else policy_params.chunk_overlap,
            parent_chunk_size=override_params.parent_chunk_size if override_params.parent_chunk_size is not None else policy_params.parent_chunk_size,
            child_chunk_size=override_params.child_chunk_size if override_params.child_chunk_size is not None else policy_params.child_chunk_size,
            preserve_headers=override_params.preserve_headers or policy_params.preserve_headers,
            custom_params=custom,
        )
