"""Unit tests for Knowledge Pipeline ChunkingPolicyResolver.

Verifies deterministic strategy selection, multi-signal evaluation (length, richness,
category, content type, semantic transitions), parameter population, threshold configuration,
and LLM-independence.
"""

from pydantic import ValidationError
import pytest

from competitive_intelligence.knowledge_pipeline import (
    ChunkingPolicy,
    ChunkingPolicyConfig,
    ChunkingPolicyResolver,
    ChunkingStrategy,
    DocumentCharacteristics,
    StructuralRichness,
)


# ==============================================================================
# ChunkingPolicyResolver Unit Tests
# ==============================================================================


def test_short_document_resolves_to_whole_document() -> None:
    """Test that documents below the length threshold resolve to WHOLE_DOCUMENT strategy."""
    resolver = ChunkingPolicyResolver()

    # Short blog post
    char_blog = DocumentCharacteristics(
        source_category="blog",
        content_type="markdown",
        char_count=450,
        structural_richness=StructuralRichness.LOW,
    )
    policy_blog = resolver.resolve_policy(char_blog)

    assert policy_blog.strategy == ChunkingStrategy.WHOLE_DOCUMENT
    assert policy_blog.parameters.chunk_size == 450
    assert policy_blog.parameters.chunk_overlap == 0
    assert "below threshold" in policy_blog.rationale.lower()

    # Short technical documentation with high richness
    char_tech = DocumentCharacteristics(
        source_category="documentation",
        content_type="markdown",
        char_count=800,
        structural_richness=StructuralRichness.HIGH,
    )
    policy_tech = resolver.resolve_policy(char_tech)

    # Length constraint overrides technical richness
    assert policy_tech.strategy == ChunkingStrategy.WHOLE_DOCUMENT
    assert policy_tech.parameters.chunk_size == 800


def test_rich_technical_documentation_resolves_to_parent_child() -> None:
    """Test that long technical/documentation content with high structural richness resolves to PARENT_CHILD."""
    resolver = ChunkingPolicyResolver()

    chars = DocumentCharacteristics(
        source_category="documentation",
        content_type="markdown",
        char_count=4500,
        structural_richness=StructuralRichness.HIGH,
    )
    policy = resolver.resolve_policy(chars)

    assert policy.strategy == ChunkingStrategy.PARENT_CHILD
    assert policy.parameters.parent_chunk_size == 2000
    assert policy.parameters.child_chunk_size == 400
    assert policy.parameters.preserve_headers is True
    assert "parent-child" in policy.rationale.lower()


def test_typical_blog_and_web_content_resolves_to_recursive() -> None:
    """Test that standard web/blog/marketing articles resolve to RECURSIVE strategy."""
    resolver = ChunkingPolicyResolver()

    chars = DocumentCharacteristics(
        source_category="blog",
        content_type="html",
        char_count=3200,
        structural_richness=StructuralRichness.MEDIUM,
    )
    policy = resolver.resolve_policy(chars)

    assert policy.strategy == ChunkingStrategy.RECURSIVE
    assert policy.parameters.chunk_size == 1000
    assert policy.parameters.chunk_overlap == 150
    assert "recursive" in policy.rationale.lower()


def test_explicit_semantic_signal_resolves_to_semantic() -> None:
    """Test that documents with explicit semantic transition flags resolve to SEMANTIC strategy."""
    resolver = ChunkingPolicyResolver()

    chars = DocumentCharacteristics(
        source_category="generic_text",
        content_type="text/plain",
        char_count=2800,
        has_semantic_transitions=True,
    )
    policy = resolver.resolve_policy(chars)

    assert policy.strategy == ChunkingStrategy.SEMANTIC
    assert policy.parameters.chunk_size == 800
    assert policy.parameters.chunk_overlap == 100
    assert "semantic" in policy.rationale.lower()


def test_unclassified_fallback_resolves_to_fixed() -> None:
    """Test that unclassified, raw, or unstructured content falls back to FIXED strategy."""
    resolver = ChunkingPolicyResolver()

    chars = DocumentCharacteristics(
        source_category=None,
        content_type="application/octet-stream",
        char_count=2000,
        structural_richness=StructuralRichness.LOW,
        has_semantic_transitions=False,
    )
    policy = resolver.resolve_policy(chars)

    assert policy.strategy == ChunkingStrategy.FIXED
    assert policy.parameters.chunk_size == 500
    assert policy.parameters.chunk_overlap == 50
    assert "fallback" in policy.rationale.lower()


def test_strategy_selection_considers_multiple_signals() -> None:
    """Test that policy decision depends on combinations of category, richness, and length."""
    resolver = ChunkingPolicyResolver()

    # 1. Documentation category with LOW richness -> Falls to RECURSIVE (not PARENT_CHILD)
    chars_low_tech = DocumentCharacteristics(
        source_category="documentation",
        content_type="text/plain",
        char_count=2500,
        structural_richness=StructuralRichness.LOW,
    )
    policy_low_tech = resolver.resolve_policy(chars_low_tech)
    assert policy_low_tech.strategy == ChunkingStrategy.RECURSIVE

    # 2. Documentation category with HIGH richness -> PARENT_CHILD
    chars_high_tech = DocumentCharacteristics(
        source_category="documentation",
        content_type="markdown",
        char_count=2500,
        structural_richness=StructuralRichness.HIGH,
    )
    policy_high_tech = resolver.resolve_policy(chars_high_tech)
    assert policy_high_tech.strategy == ChunkingStrategy.PARENT_CHILD


def test_custom_configuration_thresholds_honored() -> None:
    """Test that ChunkingPolicyResolver respects custom ChunkingPolicyConfig thresholds."""
    custom_config = ChunkingPolicyConfig(
        short_document_threshold_chars=500,
        recursive_chunk_size=1500,
        recursive_chunk_overlap=200,
    )
    resolver = ChunkingPolicyResolver(config=custom_config)

    # 600 chars would be WHOLE_DOCUMENT under default 1000 threshold, but RECURSIVE under 500
    chars = DocumentCharacteristics(
        source_category="blog",
        content_type="markdown",
        char_count=600,
    )
    policy = resolver.resolve_policy(chars)

    assert policy.strategy == ChunkingStrategy.RECURSIVE
    assert policy.parameters.chunk_size == 1500
    assert policy.parameters.chunk_overlap == 200


def test_invalid_characteristics_rejected() -> None:
    """Test that negative char_count raises ValidationError."""
    with pytest.raises(ValidationError, match="char_count cannot be negative"):
        DocumentCharacteristics(
            char_count=-10,
        )


def test_deterministic_resolution() -> None:
    """Test that evaluating the same characteristics repeatedly produces identical policies."""
    resolver = ChunkingPolicyResolver()
    chars = DocumentCharacteristics(
        source_category="product_docs",
        content_type="markdown",
        char_count=5000,
        structural_richness=StructuralRichness.HIGH,
    )

    policy1 = resolver.resolve_policy(chars)
    policy2 = resolver.resolve_policy(chars)

    assert policy1.strategy == policy2.strategy
    assert policy1.rationale == policy2.rationale
    assert policy1.parameters == policy2.parameters


def test_no_llm_or_external_dependencies() -> None:
    """Verify that chunking policy module contains zero LLM imports or network calls."""
    import competitive_intelligence.knowledge_pipeline.chunking_policy as cp_mod

    mod_file = getattr(cp_mod, "__file__", "")
    assert mod_file != ""

    with open(mod_file, "r", encoding="utf-8") as f:
        code = f.read()

    assert "openai" not in code.lower()
    assert "anthropic" not in code.lower()
    assert "httpx" not in code.lower()
    assert "requests" not in code.lower()
