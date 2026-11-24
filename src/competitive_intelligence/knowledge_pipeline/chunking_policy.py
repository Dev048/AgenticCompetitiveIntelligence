"""Chunking policy resolution for Knowledge Pipeline subsystem.

Determines the optimal chunking strategy and parameter configuration
based on multi-signal DocumentCharacteristics without running chunking algorithms.
"""

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseChunkingPolicyResolver,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkingParameters,
    ChunkingPolicy,
    ChunkingPolicyConfig,
    ChunkingStrategy,
    DocumentCharacteristics,
    StructuralRichness,
)


class ChunkingPolicyResolver(BaseChunkingPolicyResolver):
    """Deterministic resolver selecting the optimal chunking strategy and parameters.

    Responsibilities:
    - Evaluates document length, structural richness, category, content type, and semantic cues.
    - Selects one of:
        - WHOLE_DOCUMENT (short content below threshold)
        - PARENT_CHILD (richly structured documentation / technical content)
        - RECURSIVE (web, blog, pricing, marketing content)
        - SEMANTIC (explicit semantic/topic transition indicators)
        - FIXED (fallback baseline)
    - Returns ChunkingPolicy with rationale and execution parameters.
    - Purely deterministic; no LLM or external service dependencies.
    """

    def __init__(self, config: ChunkingPolicyConfig | None = None) -> None:
        self.config = config or ChunkingPolicyConfig()

    def resolve_policy(
        self, characteristics: DocumentCharacteristics
    ) -> ChunkingPolicy:
        """Resolve chunking strategy and parameters from document characteristics.

        Args:
            characteristics: Features observed from document content and source.

        Returns:
            ChunkingPolicy specifying strategy, rationale, and downstream parameters.
        """
        # 1. Short document constraint: Keep whole document to preserve complete context
        if characteristics.char_count < self.config.short_document_threshold_chars:
            return ChunkingPolicy(
                strategy=ChunkingStrategy.WHOLE_DOCUMENT,
                rationale=(
                    f"Document length ({characteristics.char_count} chars) is below "
                    f"threshold ({self.config.short_document_threshold_chars} chars); "
                    f"preserving whole document without splitting."
                ),
                parameters=ChunkingParameters(
                    chunk_size=characteristics.char_count,
                    chunk_overlap=0,
                ),
            )

        # 2. Explicit semantic transition signal
        if characteristics.has_semantic_transitions:
            return ChunkingPolicy(
                strategy=ChunkingStrategy.SEMANTIC,
                rationale=(
                    "Explicit semantic or topic transition indicators detected; "
                    "applying semantic chunking."
                ),
                parameters=ChunkingParameters(
                    chunk_size=self.config.semantic_chunk_size,
                    chunk_overlap=self.config.semantic_chunk_overlap,
                ),
            )

        # 3. Rich technical / documentation content: Parent-Child chunking
        category = (characteristics.source_category or "").lower()
        is_tech_category = category in (
            "documentation",
            "product_docs",
            "engineering",
            "api_docs",
            "changelog",
            "release_notes",
        )
        is_rich_structure = characteristics.structural_richness == StructuralRichness.HIGH

        if is_tech_category and is_rich_structure:
            return ChunkingPolicy(
                strategy=ChunkingStrategy.PARENT_CHILD,
                rationale=(
                    f"Technical/documentation category '{characteristics.source_category}' "
                    f"with high structural richness; applying parent-child chunking."
                ),
                parameters=ChunkingParameters(
                    parent_chunk_size=self.config.parent_chunk_size,
                    child_chunk_size=self.config.child_chunk_size,
                    chunk_overlap=self.config.parent_child_overlap,
                    preserve_headers=True,
                ),
            )

        # 4. Blog / Web / Marketing / General text content: Recursive character chunking
        is_web_or_blog = category in (
            "blog",
            "marketing",
            "pricing",
            "news",
            "community",
            "social",
            "product",
            "about",
        )
        content_type = (characteristics.content_type or "").lower()
        is_markdown_or_html = content_type in ("html", "markdown", "text/html", "text/markdown")

        if is_web_or_blog or is_markdown_or_html or is_tech_category:
            return ChunkingPolicy(
                strategy=ChunkingStrategy.RECURSIVE,
                rationale=(
                    f"Web/blog or general structured content (category: '{characteristics.source_category}', "
                    f"content_type: '{characteristics.content_type}'); applying recursive chunking."
                ),
                parameters=ChunkingParameters(
                    chunk_size=self.config.recursive_chunk_size,
                    chunk_overlap=self.config.recursive_chunk_overlap,
                ),
            )

        # 5. Fallback baseline: Fixed chunking
        return ChunkingPolicy(
            strategy=ChunkingStrategy.FIXED,
            rationale="Default fallback policy applied for unclassified or raw content.",
            parameters=ChunkingParameters(
                chunk_size=self.config.fixed_chunk_size,
                chunk_overlap=self.config.fixed_chunk_overlap,
            ),
        )
