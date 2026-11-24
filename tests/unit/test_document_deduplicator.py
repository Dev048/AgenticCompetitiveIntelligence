"""Unit tests for Knowledge Pipeline DocumentDeduplicator.

Verifies deterministic duplicate detection on content hashes and canonical URLs,
distinguishes content duplicates from URL candidates, tests multi-candidate matching,
and asserts non-destructive evidence preservation.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    DeduplicationResult,
    Document,
    DocumentDeduplicator,
    DocumentVersion,
    DuplicateMatch,
    DuplicateType,
    compute_content_hash,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_doc_and_version(
    doc_id: str = "doc_1",
    source_id: str = "src_1",
    canonical_url: str = "https://example.com/page",
    content: str = "Standard content payload",
    version_id: str = "ver_1",
) -> tuple[Document, DocumentVersion]:
    """Helper to create a matched Document and DocumentVersion pair."""
    doc = Document(
        document_id=doc_id,
        source_id=source_id,
        company_id="comp_1",
        workspace_id="ws_1",
        canonical_url=canonical_url,
    )
    ver = DocumentVersion(
        version_id=version_id,
        document_id=doc_id,
        content=content,
        content_hash=compute_content_hash(content),
        acquired_at=datetime.now(timezone.utc),
    )
    return doc, ver


# ==============================================================================
# DocumentDeduplicator Unit Tests
# ==============================================================================


def test_exact_content_duplicate_different_url() -> None:
    """Test that matching content_hash with differing URL produces EXACT_CONTENT duplicate."""
    deduplicator = DocumentDeduplicator()

    doc_new, ver_new = create_doc_and_version(
        doc_id="doc_new",
        canonical_url="https://mirror.example.com/article",
        content="Shared article content across mirrors",
    )
    doc_existing, ver_existing = create_doc_and_version(
        doc_id="doc_existing",
        canonical_url="https://original.example.com/article",
        content="Shared article content across mirrors",
    )

    result = deduplicator.find_duplicates(
        document=doc_new,
        version=ver_new,
        candidates=[(doc_existing, ver_existing)],
    )

    assert isinstance(result, DeduplicationResult)
    assert result.duplicate_type == DuplicateType.EXACT_CONTENT
    assert result.has_content_duplicate is True
    assert result.has_url_duplicate is False
    assert result.is_duplicate is True
    assert result.is_unique is False
    assert len(result.matches) == 1

    match = result.matches[0]
    assert match.candidate_document_id == "doc_existing"
    assert match.candidate_version_id == ver_existing.version_id
    assert match.match_type == DuplicateType.EXACT_CONTENT
    assert match.matched_content_hash is True
    assert match.matched_canonical_url is False


def test_canonical_url_duplicate_different_content() -> None:
    """Test that matching canonical_url with different content produces CANONICAL_URL duplicate."""
    deduplicator = DocumentDeduplicator()

    doc_new, ver_new = create_doc_and_version(
        doc_id="doc_v2",
        canonical_url="https://example.com/pricing",
        content="# Pricing\nPro Tier: $30/mo",
    )
    doc_existing, ver_existing = create_doc_and_version(
        doc_id="doc_v1",
        canonical_url="https://example.com/pricing",
        content="# Pricing\nPro Tier: $20/mo",
    )

    result = deduplicator.find_duplicates(
        document=doc_new,
        version=ver_new,
        candidates=[(doc_existing, ver_existing)],
    )

    assert result.duplicate_type == DuplicateType.CANONICAL_URL
    assert result.has_url_duplicate is True
    assert result.has_content_duplicate is False
    assert result.is_duplicate is True
    assert len(result.matches) == 1

    match = result.matches[0]
    assert match.candidate_document_id == "doc_v1"
    assert match.matched_canonical_url is True
    assert match.matched_content_hash is False
    assert match.match_type == DuplicateType.CANONICAL_URL


def test_exact_content_and_canonical_url_duplicate() -> None:
    """Test that matching both canonical_url and content_hash produces EXACT_CONTENT_AND_URL duplicate."""
    deduplicator = DocumentDeduplicator()

    doc_new, ver_new = create_doc_and_version(
        doc_id="doc_b",
        canonical_url="https://example.com/blog/post-1",
        content="Exact blog post body",
    )
    doc_existing, ver_existing = create_doc_and_version(
        doc_id="doc_a",
        canonical_url="https://example.com/blog/post-1",
        content="Exact blog post body",
    )

    result = deduplicator.find_duplicates(
        document=doc_new,
        version=ver_new,
        candidates=[(doc_existing, ver_existing)],
    )

    assert result.duplicate_type == DuplicateType.EXACT_CONTENT_AND_URL
    assert result.has_content_duplicate is True
    assert result.has_url_duplicate is True
    assert result.is_duplicate is True
    assert len(result.matches) == 1
    assert result.matches[0].match_type == DuplicateType.EXACT_CONTENT_AND_URL


def test_completely_distinct_document() -> None:
    """Test that a document with distinct URL and distinct content produces NO_DUPLICATE."""
    deduplicator = DocumentDeduplicator()

    doc_new, ver_new = create_doc_and_version(
        doc_id="doc_alpha",
        canonical_url="https://company-a.com/features",
        content="Features of Product Alpha",
    )
    doc_existing, ver_existing = create_doc_and_version(
        doc_id="doc_beta",
        canonical_url="https://company-b.com/pricing",
        content="Pricing of Product Beta",
    )

    result = deduplicator.find_duplicates(
        document=doc_new,
        version=ver_new,
        candidates=[(doc_existing, ver_existing)],
    )

    assert result.duplicate_type == DuplicateType.NO_DUPLICATE
    assert result.has_content_duplicate is False
    assert result.has_url_duplicate is False
    assert result.is_duplicate is False
    assert result.is_unique is True
    assert len(result.matches) == 0


def test_empty_candidates_collection() -> None:
    """Test evaluating a document with an empty candidate collection returns NO_DUPLICATE."""
    deduplicator = DocumentDeduplicator()
    doc_new, ver_new = create_doc_and_version()

    result = deduplicator.find_duplicates(
        document=doc_new,
        version=ver_new,
        candidates=[],
    )

    assert result.duplicate_type == DuplicateType.NO_DUPLICATE
    assert result.is_unique is True
    assert result.matches == []


def test_multiple_candidates_matching_different_signals() -> None:
    """Test evaluating against multiple candidates with distinct match types."""
    deduplicator = DocumentDeduplicator()

    doc_target, ver_target = create_doc_and_version(
        doc_id="doc_target",
        canonical_url="https://news.com/release-1",
        content="Official press release text.",
    )

    # Candidate 1: Same URL, different content (e.g. updated version)
    c1_doc, c1_ver = create_doc_and_version(
        doc_id="cand_1",
        canonical_url="https://news.com/release-1",
        content="Old press release draft.",
    )

    # Candidate 2: Different URL (mirror/syndication), same content
    c2_doc, c2_ver = create_doc_and_version(
        doc_id="cand_2",
        canonical_url="https://syndicate.com/release-1",
        content="Official press release text.",
    )

    # Candidate 3: Unrelated candidate (no match)
    c3_doc, c3_ver = create_doc_and_version(
        doc_id="cand_3",
        canonical_url="https://unrelated.com/about",
        content="About our company.",
    )

    # Candidate 4: Exact content AND URL match
    c4_doc, c4_ver = create_doc_and_version(
        doc_id="cand_4",
        canonical_url="https://news.com/release-1",
        content="Official press release text.",
    )

    result = deduplicator.find_duplicates(
        document=doc_target,
        version=ver_target,
        candidates=[
            (c1_doc, c1_ver),
            (c2_doc, c2_ver),
            (c3_doc, c3_ver),
            (c4_doc, c4_ver),
        ],
    )

    assert result.duplicate_type == DuplicateType.EXACT_CONTENT_AND_URL
    assert result.has_url_duplicate is True
    assert result.has_content_duplicate is True
    assert len(result.matches) == 3

    match_types = {m.candidate_document_id: m.match_type for m in result.matches}
    assert match_types["cand_1"] == DuplicateType.CANONICAL_URL
    assert match_types["cand_2"] == DuplicateType.EXACT_CONTENT
    assert match_types["cand_4"] == DuplicateType.EXACT_CONTENT_AND_URL
    assert "cand_3" not in match_types


def test_source_and_event_identity_preserved_without_merging() -> None:
    """Test that two separate news outlets reporting on the same event with different text and URLs remain distinct."""
    deduplicator = DocumentDeduplicator()

    doc_techcrunch, ver_techcrunch = create_doc_and_version(
        doc_id="doc_tc",
        source_id="src_techcrunch",
        canonical_url="https://techcrunch.com/2026/08/competitor-raises-series-b",
        content="TechCrunch reports Competitor raised $50M Series B led by VC Firm.",
    )
    doc_bloomberg, ver_bloomberg = create_doc_and_version(
        doc_id="doc_bb",
        source_id="src_bloomberg",
        canonical_url="https://bloomberg.com/news/competitor-series-b-funding",
        content="Bloomberg News: AI startup Competitor secures $50M in new financing round.",
    )

    result = deduplicator.find_duplicates(
        document=doc_bloomberg,
        version=ver_bloomberg,
        candidates=[(doc_techcrunch, ver_techcrunch)],
    )

    # Documents discuss the same real-world event but are distinct articles/sources
    assert result.duplicate_type == DuplicateType.NO_DUPLICATE
    assert result.is_unique is True
    assert result.is_duplicate is False
    assert len(result.matches) == 0


def test_deduplicator_does_not_mutate_inputs() -> None:
    """Verify that find_duplicates does not alter input Document or DocumentVersion instances."""
    deduplicator = DocumentDeduplicator()

    doc_in, ver_in = create_doc_and_version(doc_id="doc_in", content="Input text")
    doc_cand, ver_cand = create_doc_and_version(doc_id="doc_cand", content="Input text")

    doc_in_dump = doc_in.model_dump()
    ver_in_dump = ver_in.model_dump()
    doc_cand_dump = doc_cand.model_dump()
    ver_cand_dump = ver_cand.model_dump()

    _ = deduplicator.find_duplicates(
        document=doc_in,
        version=ver_in,
        candidates=[(doc_cand, ver_cand)],
    )

    assert doc_in.model_dump() == doc_in_dump
    assert ver_in.model_dump() == ver_in_dump
    assert doc_cand.model_dump() == doc_cand_dump
    assert ver_cand.model_dump() == ver_cand_dump


def test_mismatched_version_document_id_raises_value_error() -> None:
    """Test that providing a DocumentVersion whose document_id does not match the Document raises ValueError."""
    deduplicator = DocumentDeduplicator()

    doc = Document(
        document_id="doc_expected_123",
        source_id="src_1",
        company_id="comp_1",
        workspace_id="ws_1",
        canonical_url="https://example.com",
    )
    ver = DocumentVersion(
        version_id="ver_1",
        document_id="doc_mismatched_456",
        content="Content",
        content_hash=compute_content_hash("Content"),
        acquired_at=datetime.now(timezone.utc),
    )

    with pytest.raises(ValueError, match="does not match Document document_id"):
        deduplicator.find_duplicates(document=doc, version=ver, candidates=[])
