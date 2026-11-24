"""Knowledge Ingestion Coordinator for Knowledge Pipeline subsystem.

Orchestrates the complete lifecycle:
AcquisitionResult -> Normalize -> Deduplicate -> Change Detect -> Persist Document/Version
-> Chunk -> Persist Chunks -> Embed -> Qdrant Upsert.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
import uuid

from competitive_intelligence.core.logger import get_logger
from competitive_intelligence.knowledge_pipeline.change_detector import (
    ChangeDetector,
)
from competitive_intelligence.knowledge_pipeline.chunking_service import (
    ChunkingService,
)
from competitive_intelligence.knowledge_pipeline.deduplicator import (
    DocumentDeduplicator,
)
from competitive_intelligence.knowledge_pipeline.embeddings.embedding_service import (
    EmbeddingService,
)
from competitive_intelligence.knowledge_pipeline.interfaces import (
    BaseChangeDetector,
    BaseDocumentDeduplicator,
    BaseDocumentNormalizer,
    BaseDocumentRepository,
    BaseEmbeddingProvider,
    BaseVectorStore,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    ChangeType,
    ChunkingParameters,
    DeduplicationResult,
    Document,
    DocumentChunk,
    DocumentVersion,
    IngestionResult,
    IngestionStatus,
    VectorStoreEntry,
)
from competitive_intelligence.knowledge_pipeline.normalizer import (
    DocumentNormalizer,
)
from competitive_intelligence.source_acquisition.models import (
    AcquisitionResult,
    Source,
)

logger = get_logger(__name__)


class KnowledgeIngestionService:
    """Production ingestion service for the Knowledge Pipeline subsystem.

    Coordinates:
    - Document normalization (Source + AcquisitionResult -> Document + DocumentVersion).
    - Database-backed deduplication lookup.
    - Change detection between consecutive versions.
    - PostgreSQL persistence of Document, DocumentVersion, and DocumentChunks.
    - Chunk generation via unified ChunkingService with dynamic policy resolution.
    - Embedding generation via EmbeddingService.
    - Vector upsert into BaseVectorStore (e.g. Qdrant) with full provenance metadata.

    Guarantees:
    - PostgreSQL is the canonical system of record.
    - Qdrant is the searchable vector projection.
    - If content is UNCHANGED and vector indexing is complete, processing is skipped.
    - If content or chunks exist but vector indexing is incomplete, missing vectors are retried.
    - Concurrency collisions on (document_id, content_hash) are handled idempotently.
    - Strict provenance is preserved across Document -> Version -> Chunks -> Vector Points.
    """

    def __init__(
        self,
        document_repo: BaseDocumentRepository | None = None,
        doc_repo: BaseDocumentRepository | None = None,
        chunking_service: ChunkingService | None = None,
        normalizer: BaseDocumentNormalizer | None = None,
        deduplicator: BaseDocumentDeduplicator | None = None,
        change_detector: BaseChangeDetector | None = None,
        embedding_service: EmbeddingService | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        repo = document_repo or doc_repo
        if repo is None:
            raise ValueError("document_repo is required")
        self.document_repo = repo
        self.chunking_service = chunking_service or ChunkingService()
        self.normalizer = normalizer or DocumentNormalizer()
        self.deduplicator = deduplicator or DocumentDeduplicator()
        self.change_detector = change_detector or ChangeDetector()
        self.embedding_service: EmbeddingService | None
        if embedding_service is not None:
            self.embedding_service = embedding_service
        elif embedding_provider is not None:
            self.embedding_service = EmbeddingService(provider=embedding_provider)
        else:
            self.embedding_service = None
        self.vector_store = vector_store

    async def _is_indexing_complete(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> bool:
        """Check if all eligible chunks for a version are indexed in the vector store."""
        if self.embedding_service is None or self.vector_store is None:
            return True
        if not chunks:
            return False

        eligible_chunks = self.embedding_service.filter_eligible_chunks(chunks)
        if not eligible_chunks:
            return True

        try:
            await self.vector_store.aensure_collection()
            for chunk in eligible_chunks:
                point = await self.vector_store.aget_by_id(chunk.chunk_id)
                if point is None:
                    return False
            return True
        except Exception:
            return False

    async def ingest(
        self,
        source: Source,
        result: AcquisitionResult,
        chunking_parameters: ChunkingParameters | None = None,
    ) -> IngestionResult:
        """Execute the full knowledge ingestion lifecycle for an acquired source snapshot.

        Args:
            source: Monitored Source domain model.
            result: Successful AcquisitionResult containing content and metadata.
            chunking_parameters: Optional execution parameter overrides for chunking.

        Returns:
            Structured IngestionResult summarizing document, version, change detection,
            chunks, and indexing outcomes.
        """
        # 1. Normalize
        normalized_doc, normalized_ver = self.normalizer.normalize(
            source=source, result=result
        )

        workspace_id = source.identity.workspace_id
        company_id = source.identity.company_id
        source_id = source.identity.source_id

        # 2. Resolve canonical Document in repository
        existing_doc = await self.document_repo.get_document_by_canonical_url(
            source_id=source_id,
            canonical_url=normalized_doc.canonical_url,
        )
        if existing_doc is None:
            existing_doc = await self.document_repo.get_document(normalized_doc.document_id)

        target_document = existing_doc or normalized_doc

        # Ensure normalized version references the target document_id
        if normalized_ver.document_id != target_document.document_id:
            normalized_ver = normalized_ver.model_copy(
                update={"document_id": target_document.document_id}
            )

        # 3. Retrieve candidates for deduplication
        existing_versions = await self.document_repo.list_versions(target_document.document_id)
        candidates: list[tuple[Document, DocumentVersion]] = [
            (target_document, v) for v in existing_versions
        ]

        dedup_result: DeduplicationResult | None = None
        if candidates:
            dedup_result = self.deduplicator.find_duplicates(
                document=target_document,
                version=normalized_ver,
                candidates=candidates,
            )

        # 4. Change Detection
        latest_version = await self.document_repo.get_latest_version(target_document.document_id)
        change_result = self.change_detector.detect_change(
            current_version=normalized_ver,
            previous_version=latest_version,
        )

        logger.info(
            "INGESTION_CHANGE_DETECTED",
            workspace_id=workspace_id,
            company_id=company_id,
            source_id=source_id,
            document_id=target_document.document_id,
            change_type=change_result.change_type.value,
        )

        # 5. Handle UNCHANGED content (skip chunking, embedding, and vector upsert ONLY if indexing complete)
        if change_result.change_type == ChangeType.UNCHANGED and latest_version is not None:
            existing_chunks = await self.document_repo.get_chunks_for_version(
                latest_version.version_id
            )
            if existing_chunks and await self._is_indexing_complete(existing_chunks):
                return IngestionResult(
                    document=target_document,
                    version=latest_version,
                    change_result=change_result,
                    deduplication_result=dedup_result,
                    chunks=[],
                    embeddings_count=0,
                    vectors_count=0,
                    is_skipped_unchanged=True,
                    is_duplicate=dedup_result.is_duplicate if dedup_result else False,
                    status=IngestionStatus.SKIPPED_UNCHANGED,
                )

        # 6. Persist Document (if not already existing)
        if existing_doc is None:
            try:
                await self.document_repo.create_document(target_document)
            except Exception as exc:
                # Handle possible concurrent creation collision
                existing_doc_lookup = await self.document_repo.get_document_by_canonical_url(
                    source_id=source_id,
                    canonical_url=target_document.canonical_url,
                )
                if existing_doc_lookup is not None:
                    target_document = existing_doc_lookup
                else:
                    raise exc

        # 7. Persist DocumentVersion (with unique constraint collision recovery)
        persisted_version: DocumentVersion
        existing_version_by_hash = await self.document_repo.get_version_by_hash(
            document_id=target_document.document_id,
            content_hash=normalized_ver.content_hash,
        )

        if existing_version_by_hash is not None:
            persisted_version = existing_version_by_hash
        else:
            try:
                persisted_version = await self.document_repo.create_version(normalized_ver)
            except Exception as exc:
                # Check for concurrent insert collision on (document_id, content_hash)
                existing_ver_lookup = await self.document_repo.get_version_by_hash(
                    document_id=target_document.document_id,
                    content_hash=normalized_ver.content_hash,
                )
                if existing_ver_lookup is not None:
                    persisted_version = existing_ver_lookup
                else:
                    raise exc

        # 8 & 9. Resolve or Partition and Persist Chunks in PostgreSQL (reuse existing chunks without duplicate insertion)
        existing_chunks = await self.document_repo.get_chunks_for_version(
            persisted_version.version_id
        )
        if existing_chunks:
            persisted_chunks = existing_chunks
        else:
            chunks = self.chunking_service.chunk(
                version=persisted_version,
                parameters=chunking_parameters,
                source=source,
            )
            persisted_chunks = await self.document_repo.create_chunks(chunks)

        # 10. Generate Embeddings & Upsert to Vector Store (if configured)
        embeddings_count = 0
        vectors_count = 0

        if self.embedding_service is not None and self.vector_store is not None and persisted_chunks:
            # Ensure Qdrant collection exists
            await self.vector_store.aensure_collection()

            eligible_chunks = self.embedding_service.filter_eligible_chunks(persisted_chunks)

            # Determine chunks requiring vector indexing
            chunks_to_index: list[DocumentChunk] = []
            for chunk in eligible_chunks:
                try:
                    point = await self.vector_store.aget_by_id(chunk.chunk_id)
                    if point is None:
                        chunks_to_index.append(chunk)
                except Exception:
                    chunks_to_index.append(chunk)

            if chunks_to_index:
                # Generate dense embeddings (batching and hierarchical filtering handled internally)
                embedding_records = await self.embedding_service.agenerate_embeddings(chunks_to_index)
                embeddings_count = len(embedding_records)

                if embedding_records:
                    # Map chunks to VectorStoreEntry points
                    emb_by_chunk_id = {emb.chunk_id: emb for emb in embedding_records}
                    entries = [
                        VectorStoreEntry.from_chunk_and_embedding(chunk, emb_by_chunk_id[chunk.chunk_id])
                        for chunk in chunks_to_index
                        if chunk.chunk_id in emb_by_chunk_id
                    ]

                    # Upsert entries to VectorStore
                    await self.vector_store.aupsert_entries(entries)
                    vectors_count = len(entries)

        logger.info(
            "INGESTION_COMPLETED",
            workspace_id=workspace_id,
            company_id=company_id,
            source_id=source_id,
            document_id=target_document.document_id,
            version_id=persisted_version.version_id,
            chunks_count=len(persisted_chunks),
            embeddings_count=embeddings_count,
            vectors_count=vectors_count,
        )

        return IngestionResult(
            document=target_document,
            version=persisted_version,
            change_result=change_result,
            deduplication_result=dedup_result,
            chunks=persisted_chunks,
            embeddings_count=embeddings_count,
            vectors_count=vectors_count,
            is_skipped_unchanged=False,
            is_duplicate=False,
            status=IngestionStatus.COMPLETED,
        )
