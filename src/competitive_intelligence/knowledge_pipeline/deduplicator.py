"""Deterministic document deduplication for Knowledge Pipeline subsystem.

Identifies exact content duplicates (via SHA-256 content hashes) and canonical URL
duplicate candidates without performing destructive operations, entity resolution,
or semantic merging.
"""

from collections.abc import Sequence

from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseDocumentDeduplicator,
)
from competitive_intelligence.knowledge_pipeline.models import (
    DeduplicationResult,
    Document,
    DocumentVersion,
    DuplicateMatch,
    DuplicateType,
)


class DocumentDeduplicator(BaseDocumentDeduplicator):
    """Deterministic deduplicator matching on content hashes and canonical URLs.

    Responsibilities:
    - Evaluates incoming Document and DocumentVersion against in-memory candidate records.
    - Classifies candidate matches into EXACT_CONTENT, CANONICAL_URL, or EXACT_CONTENT_AND_URL.
    - Aggregates overall duplicate status and preserves individual candidate match details.
    - Preserves evidence: does not merge, overwrite, or delete documents or versions.
    - Operates purely deterministically with zero LLM or embedding dependencies.
    """

    def find_duplicates(
        self,
        document: Document,
        version: DocumentVersion,
        candidates: Sequence[tuple[Document, DocumentVersion]],
    ) -> DeduplicationResult:
        """Find content and URL duplicate matches among existing document candidates.

        Args:
            document: The logical Document to evaluate.
            version: The specific DocumentVersion snapshot to evaluate.
            candidates: Sequence of existing (Document, DocumentVersion) candidate pairs.

        Returns:
            DeduplicationResult summarizing duplicate classifications and candidate matches.

        Raises:
            ValueError: If version.document_id does not match document.document_id.
        """
        if version.document_id != document.document_id:
            raise ValueError(
                f"Version document_id '{version.document_id}' does not match "
                f"Document document_id '{document.document_id}'"
            )

        matches: list[DuplicateMatch] = []

        for cand_doc, cand_ver in candidates:
            matched_url = cand_doc.canonical_url == document.canonical_url
            matched_hash = cand_ver.content_hash == version.content_hash

            if not matched_url and not matched_hash:
                continue

            if matched_url and matched_hash:
                match_type = DuplicateType.EXACT_CONTENT_AND_URL
            elif matched_hash:
                match_type = DuplicateType.EXACT_CONTENT
            else:
                match_type = DuplicateType.CANONICAL_URL

            matches.append(
                DuplicateMatch(
                    candidate_document_id=cand_doc.document_id,
                    candidate_version_id=cand_ver.version_id,
                    match_type=match_type,
                    matched_canonical_url=matched_url,
                    matched_content_hash=matched_hash,
                    candidate_canonical_url=cand_doc.canonical_url,
                    candidate_content_hash=cand_ver.content_hash,
                )
            )

        has_content_duplicate = any(m.matched_content_hash for m in matches)
        has_url_duplicate = any(m.matched_canonical_url for m in matches)
        is_duplicate = len(matches) > 0

        if has_content_duplicate and has_url_duplicate:
            overall_duplicate_type = DuplicateType.EXACT_CONTENT_AND_URL
        elif has_content_duplicate:
            overall_duplicate_type = DuplicateType.EXACT_CONTENT
        elif has_url_duplicate:
            overall_duplicate_type = DuplicateType.CANONICAL_URL
        else:
            overall_duplicate_type = DuplicateType.NO_DUPLICATE

        return DeduplicationResult(
            document_id=document.document_id,
            version_id=version.version_id,
            duplicate_type=overall_duplicate_type,
            has_content_duplicate=has_content_duplicate,
            has_url_duplicate=has_url_duplicate,
            is_duplicate=is_duplicate,
            matches=matches,
        )
