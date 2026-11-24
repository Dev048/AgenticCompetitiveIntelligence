"""Deterministic change detection for Knowledge Pipeline subsystem.

Compares consecutive DocumentVersion instances of the same Document to classify
state transitions into INITIAL, UNCHANGED, or CHANGED based on content hashes.
"""

from competitive_intelligence.knowledge_pipeline.interfaces import BaseChangeDetector
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    ChangeType,
    DocumentVersion,
)


class ChangeDetectionError(Exception):
    """Base exception for change detection failures."""

    pass


class MismatchedDocumentError(ChangeDetectionError):
    """Raised when attempting to compare versions belonging to different documents."""

    def __init__(self, current_doc_id: str, previous_doc_id: str) -> None:
        self.current_doc_id = current_doc_id
        self.previous_doc_id = previous_doc_id
        super().__init__(
            f"Cannot compare versions belonging to different documents: "
            f"current='{current_doc_id}', previous='{previous_doc_id}'"
        )


class ChangeDetector(BaseChangeDetector):
    """Deterministic change detector comparing consecutive DocumentVersion snapshots.

    Responsibilities:
    - Compares current DocumentVersion against its immediately preceding DocumentVersion.
    - Classifies state transition:
        - None previous -> INITIAL
        - Equal content_hash -> UNCHANGED
        - Different content_hash -> CHANGED
    - Enforces document boundary (mismatched document_id raises MismatchedDocumentError).
    - Preserves immutability (no mutation of input versions).
    """

    def detect_change(
        self,
        current_version: DocumentVersion,
        previous_version: DocumentVersion | None = None,
    ) -> ChangeResult:
        """Detect content change between current and immediately preceding document version.

        Args:
            current_version: The newly acquired/normalized DocumentVersion.
            previous_version: The immediately preceding DocumentVersion for the same document,
                or None if this is the first version.

        Returns:
            ChangeResult containing the change classification and version provenance.

        Raises:
            MismatchedDocumentError: If previous_version belongs to a different document_id.
        """
        if previous_version is not None and previous_version.document_id != current_version.document_id:
            raise MismatchedDocumentError(
                current_doc_id=current_version.document_id,
                previous_doc_id=previous_version.document_id,
            )

        if previous_version is None:
            return ChangeResult(
                document_id=current_version.document_id,
                current_version_id=current_version.version_id,
                previous_version_id=None,
                change_type=ChangeType.INITIAL,
                current_content_hash=current_version.content_hash,
                previous_content_hash=None,
            )

        if previous_version.content_hash == current_version.content_hash:
            change_type = ChangeType.UNCHANGED
        else:
            change_type = ChangeType.CHANGED

        return ChangeResult(
            document_id=current_version.document_id,
            current_version_id=current_version.version_id,
            previous_version_id=previous_version.version_id,
            change_type=change_type,
            current_content_hash=current_version.content_hash,
            previous_content_hash=previous_version.content_hash,
        )
