"""Unit tests for Knowledge Pipeline ChangeDetector.

Verifies deterministic change detection, temporal rule adherence,
multi-version state progressions, document boundary enforcement, and immutability.
"""

from datetime import datetime, timezone

import pytest

from competitive_intelligence.knowledge_pipeline import (
    ChangeDetectionError,
    ChangeDetector,
    ChangeResult,
    ChangeType,
    DocumentVersion,
    MismatchedDocumentError,
    compute_content_hash,
)


# ==============================================================================
# Helper Factories
# ==============================================================================


def create_version(
    document_id: str = "doc_test_123",
    version_id: str = "ver_1",
    content: str = "# Pricing: $20/mo",
    acquired_at: datetime | None = None,
) -> DocumentVersion:
    """Helper to create a valid DocumentVersion with deterministic content hash."""
    return DocumentVersion(
        version_id=version_id,
        document_id=document_id,
        content=content,
        content_hash=compute_content_hash(content),
        acquired_at=acquired_at or datetime.now(timezone.utc),
        metadata={"source": "test"},
    )


# ==============================================================================
# ChangeDetector Unit Tests
# ==============================================================================


def test_first_version_is_initial() -> None:
    """Test that evaluating a version with no preceding version produces INITIAL state."""
    detector = ChangeDetector()
    v1 = create_version(version_id="ver_1", content="Initial Content")

    result = detector.detect_change(current_version=v1, previous_version=None)

    assert isinstance(result, ChangeResult)
    assert result.document_id == v1.document_id
    assert result.current_version_id == "ver_1"
    assert result.previous_version_id is None
    assert result.change_type == ChangeType.INITIAL
    assert result.is_initial is True
    assert result.is_changed is False
    assert result.is_unchanged is False
    assert result.current_content_hash == v1.content_hash
    assert result.previous_content_hash is None
    assert result.detected_at.tzinfo is not None


def test_same_content_hash_is_unchanged() -> None:
    """Test that identical content hashes between consecutive versions produce UNCHANGED state."""
    detector = ChangeDetector()
    v1 = create_version(version_id="ver_1", content="Identical Content")
    v2 = create_version(version_id="ver_2", content="Identical Content")

    result = detector.detect_change(current_version=v2, previous_version=v1)

    assert result.change_type == ChangeType.UNCHANGED
    assert result.is_unchanged is True
    assert result.is_changed is False
    assert result.is_initial is False
    assert result.document_id == v1.document_id
    assert result.current_version_id == "ver_2"
    assert result.previous_version_id == "ver_1"
    assert result.current_content_hash == v2.content_hash
    assert result.previous_content_hash == v1.content_hash


def test_different_content_hash_is_changed() -> None:
    """Test that differing content hashes between consecutive versions produce CHANGED state."""
    detector = ChangeDetector()
    v1 = create_version(version_id="ver_1", content="Plan A: $20/mo")
    v2 = create_version(version_id="ver_2", content="Plan A: $25/mo")

    result = detector.detect_change(current_version=v2, previous_version=v1)

    assert result.change_type == ChangeType.CHANGED
    assert result.is_changed is True
    assert result.is_unchanged is False
    assert result.is_initial is False
    assert result.current_version_id == "ver_2"
    assert result.previous_version_id == "ver_1"
    assert result.current_content_hash != result.previous_content_hash


def test_temporal_rule_compares_immediately_preceding_version() -> None:
    """Test that comparison strictly checks immediately preceding version, not older historical matches.

    Sequence:
    V1: $20 (initial)
    V2: $30 (changed vs V1)
    V3: $20 (changed vs V2, even though content matches V1)
    """
    detector = ChangeDetector()
    v1 = create_version(version_id="ver_1", content="Price: $20")
    v2 = create_version(version_id="ver_2", content="Price: $30")
    v3 = create_version(version_id="ver_3", content="Price: $20")

    # V3 compared to V2 must be CHANGED
    result_v3_vs_v2 = detector.detect_change(current_version=v3, previous_version=v2)

    assert result_v3_vs_v2.change_type == ChangeType.CHANGED
    assert result_v3_vs_v2.is_changed is True
    assert result_v3_vs_v2.previous_version_id == "ver_2"
    assert result_v3_vs_v2.current_content_hash == v1.content_hash  # Same content as V1
    assert result_v3_vs_v2.current_content_hash != v2.content_hash  # Differs from V2


def test_sequential_version_lifecycle_progression() -> None:
    """Test the full lifecycle progression: INITIAL -> CHANGED -> CHANGED -> UNCHANGED.

    V1: $20 → INITIAL
    V2: $30 → CHANGED
    V3: $20 → CHANGED
    V4: $20 → UNCHANGED
    """
    detector = ChangeDetector()
    v1 = create_version(version_id="v1", content="Tier 1: $20")
    v2 = create_version(version_id="v2", content="Tier 1: $30")
    v3 = create_version(version_id="v3", content="Tier 1: $20")
    v4 = create_version(version_id="v4", content="Tier 1: $20")

    r1 = detector.detect_change(current_version=v1, previous_version=None)
    assert r1.change_type == ChangeType.INITIAL

    r2 = detector.detect_change(current_version=v2, previous_version=v1)
    assert r2.change_type == ChangeType.CHANGED

    r3 = detector.detect_change(current_version=v3, previous_version=v2)
    assert r3.change_type == ChangeType.CHANGED

    r4 = detector.detect_change(current_version=v4, previous_version=v3)
    assert r4.change_type == ChangeType.UNCHANGED


def test_mismatched_document_ids_rejected() -> None:
    """Test that comparing versions from different documents raises MismatchedDocumentError."""
    detector = ChangeDetector()
    v1 = create_version(document_id="doc_apple_pricing", version_id="v1", content="Content")
    v2 = create_version(document_id="doc_google_pricing", version_id="v2", content="Content")

    with pytest.raises(MismatchedDocumentError) as exc_info:
        detector.detect_change(current_version=v2, previous_version=v1)

    assert exc_info.value.current_doc_id == "doc_google_pricing"
    assert exc_info.value.previous_doc_id == "doc_apple_pricing"
    assert issubclass(MismatchedDocumentError, ChangeDetectionError)


def test_detector_does_not_mutate_versions() -> None:
    """Verify that detect_change preserves complete immutability of input DocumentVersions."""
    detector = ChangeDetector()
    v1 = create_version(version_id="v1", content="Content V1")
    v2 = create_version(version_id="v2", content="Content V2")

    v1_dump_before = v1.model_dump()
    v2_dump_before = v2.model_dump()

    _ = detector.detect_change(current_version=v2, previous_version=v1)

    assert v1.model_dump() == v1_dump_before
    assert v2.model_dump() == v2_dump_before
