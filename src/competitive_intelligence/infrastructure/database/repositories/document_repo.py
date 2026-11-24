"""SQL repository for canonical Document, DocumentVersion, and DocumentChunk persistence."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from competitive_intelligence.application.interfaces import BaseDocumentRepository
from competitive_intelligence.infrastructure.database.models import (
    DocumentChunkOrm,
    DocumentOrm,
    DocumentVersionOrm,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentVersion,
)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlDocumentRepository(BaseDocumentRepository):
    """Repository managing canonical Documents, append-only Versions, and Chunks."""

    def __init__(
        self,
        session: AsyncSession | async_sessionmaker[AsyncSession] | DatabaseSessionManager | None = None,
    ) -> None:
        self._session: AsyncSession | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None

        if isinstance(session, AsyncSession):
            self._session = session
        elif isinstance(session, DatabaseSessionManager):
            self._sessionmaker = session.sessionmaker
        elif session is not None:
            self._sessionmaker = session
        else:
            self._sessionmaker = get_db_manager().sessionmaker

    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._session is not None:
            yield self._session
        elif self._sessionmaker is not None:
            async with self._sessionmaker() as session:
                yield session
                await session.commit()
        else:
            raise RuntimeError("SqlDocumentRepository has no active session or sessionmaker")

    # --------------------------------------------------------------------------
    # Document Operations
    # --------------------------------------------------------------------------

    async def create_document(self, document: Document) -> Document:
        """Persist a new canonical Document identity."""
        orm_doc = DocumentOrm(
            document_id=document.document_id,
            source_id=document.source_id,
            workspace_id=document.workspace_id,
            company_id=document.company_id,
            canonical_url=document.canonical_url,
        )
        async with self._get_session() as session:
            session.add(orm_doc)
            await session.flush()
        return document

    async def get_document(self, document_id: str) -> Document | None:
        """Fetch a canonical Document by primary document_id."""
        return await self.get_document_by_id(document_id)

    async def get_document_by_id(self, document_id: str) -> Document | None:
        """Fetch a canonical Document by primary document_id."""
        stmt = select(DocumentOrm).where(DocumentOrm.document_id == document_id)
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return Document(
                document_id=row.document_id,
                source_id=row.source_id,
                workspace_id=row.workspace_id,
                company_id=row.company_id,
                canonical_url=row.canonical_url,
            )

    async def get_document_by_canonical_url(
        self, source_id: str, canonical_url: str
    ) -> Document | None:
        """Fetch a canonical Document by source_id and canonical_url."""
        stmt = (
            select(DocumentOrm)
            .where(DocumentOrm.source_id == source_id)
            .where(DocumentOrm.canonical_url == canonical_url)
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return Document(
                document_id=row.document_id,
                source_id=row.source_id,
                workspace_id=row.workspace_id,
                company_id=row.company_id,
                canonical_url=row.canonical_url,
            )

    async def list_documents(
        self,
        workspace_id: str,
        company_id: str | None = None,
        source_id: str | None = None,
    ) -> list[Document]:
        """List canonical documents in workspace with optional filters."""
        stmt = select(DocumentOrm).where(DocumentOrm.workspace_id == workspace_id)
        if company_id:
            stmt = stmt.where(DocumentOrm.company_id == company_id)
        if source_id:
            stmt = stmt.where(DocumentOrm.source_id == source_id)
        stmt = stmt.order_by(DocumentOrm.created_at.asc())
        async with self._get_session() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                Document(
                    document_id=r.document_id,
                    source_id=r.source_id,
                    workspace_id=r.workspace_id,
                    company_id=r.company_id,
                    canonical_url=r.canonical_url,
                )
                for r in rows
            ]

    async def list_documents_by_company(self, company_id: str) -> list[Document]:
        """List all canonical documents tracked for a competitor."""
        stmt = (
            select(DocumentOrm)
            .where(DocumentOrm.company_id == company_id)
            .order_by(DocumentOrm.created_at.asc())
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                Document(
                    document_id=r.document_id,
                    source_id=r.source_id,
                    workspace_id=r.workspace_id,
                    company_id=r.company_id,
                    canonical_url=r.canonical_url,
                )
                for r in rows
            ]

    # --------------------------------------------------------------------------
    # DocumentVersion Operations (Append-Only)
    # --------------------------------------------------------------------------

    async def create_version(self, version: DocumentVersion) -> DocumentVersion:
        """Persist an immutable, append-only DocumentVersion snapshot."""
        existing = await self.get_version_by_id(version.version_id)
        if existing is not None:
            return existing

        orm_ver = DocumentVersionOrm(
            version_id=version.version_id,
            document_id=version.document_id,
            content=version.content,
            title=version.title,
            content_hash=version.content_hash,
            acquired_at=version.acquired_at,
            metadata_=version.metadata,
        )
        async with self._get_session() as session:
            session.add(orm_ver)
            await session.flush()
        return version

    async def get_version(self, version_id: str) -> DocumentVersion | None:
        """Alias for get_version_by_id conforming to BaseDocumentRepository."""
        return await self.get_version_by_id(version_id)

    async def get_version_by_id(self, version_id: str) -> DocumentVersion | None:
        """Fetch a DocumentVersion by primary version_id."""
        stmt = select(DocumentVersionOrm).where(DocumentVersionOrm.version_id == version_id)
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            acq_at = _ensure_utc(row.acquired_at)
            assert acq_at is not None
            return DocumentVersion(
                version_id=row.version_id,
                document_id=row.document_id,
                content=row.content,
                title=row.title,
                content_hash=row.content_hash,
                acquired_at=acq_at,
                metadata=row.metadata_,
            )

    async def get_version_by_hash(
        self, document_id: str, content_hash: str
    ) -> DocumentVersion | None:
        """Fetch version by document ID and content hash."""
        stmt = (
            select(DocumentVersionOrm)
            .where(DocumentVersionOrm.document_id == document_id)
            .where(DocumentVersionOrm.content_hash == content_hash)
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            acq_at = _ensure_utc(row.acquired_at)
            assert acq_at is not None
            return DocumentVersion(
                version_id=row.version_id,
                document_id=row.document_id,
                content=row.content,
                title=row.title,
                content_hash=row.content_hash,
                acquired_at=acq_at,
                metadata=row.metadata_,
            )

    async def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        """Fetch the most recently acquired DocumentVersion for a document."""
        stmt = (
            select(DocumentVersionOrm)
            .where(DocumentVersionOrm.document_id == document_id)
            .order_by(DocumentVersionOrm.acquired_at.desc())
            .limit(1)
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            acq_at = _ensure_utc(row.acquired_at)
            assert acq_at is not None
            return DocumentVersion(
                version_id=row.version_id,
                document_id=row.document_id,
                content=row.content,
                title=row.title,
                content_hash=row.content_hash,
                acquired_at=acq_at,
                metadata=row.metadata_,
            )

    async def list_versions(self, document_id: str) -> list[DocumentVersion]:
        """Alias for list_versions_for_document conforming to BaseDocumentRepository."""
        return await self.list_versions_for_document(document_id)

    async def list_versions_for_document(self, document_id: str) -> list[DocumentVersion]:
        """List all chronological versions of a document."""
        stmt = (
            select(DocumentVersionOrm)
            .where(DocumentVersionOrm.document_id == document_id)
            .order_by(DocumentVersionOrm.acquired_at.asc())
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            rows = result.scalars().all()
            versions: list[DocumentVersion] = []
            for r in rows:
                acq_at = _ensure_utc(r.acquired_at)
                assert acq_at is not None
                versions.append(
                    DocumentVersion(
                        version_id=r.version_id,
                        document_id=r.document_id,
                        content=r.content,
                        title=r.title,
                        content_hash=r.content_hash,
                        acquired_at=acq_at,
                        metadata=r.metadata_,
                    )
                )
            return versions

    # --------------------------------------------------------------------------
    # DocumentChunk Operations
    # --------------------------------------------------------------------------

    async def create_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Persist a single DocumentChunk."""
        existing = await self.get_chunk_by_id(chunk.chunk_id)
        if existing is not None:
            return existing

        orm_chunk = DocumentChunkOrm(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            content_hash=chunk.content_hash,
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_level=chunk.chunk_level,
            start_char=chunk.metadata.start_char,
            end_char=chunk.metadata.end_char,
            total_chunks=chunk.metadata.total_chunks,
            language=chunk.metadata.language,
            custom_tags=chunk.metadata.custom_tags,
            created_at=chunk.created_at,
        )
        async with self._get_session() as session:
            session.add(orm_chunk)
            await session.flush()
        return chunk

    async def create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Bulk persist a list of DocumentChunks."""
        async with self._get_session() as session:
            for chunk in chunks:
                existing = await session.get(DocumentChunkOrm, chunk.chunk_id)
                if existing is not None:
                    continue
                orm_chunk = DocumentChunkOrm(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    parent_chunk_id=chunk.parent_chunk_id,
                    chunk_level=chunk.chunk_level,
                    start_char=chunk.metadata.start_char,
                    end_char=chunk.metadata.end_char,
                    total_chunks=chunk.metadata.total_chunks,
                    language=chunk.metadata.language,
                    custom_tags=chunk.metadata.custom_tags,
                    created_at=chunk.created_at,
                )
                session.add(orm_chunk)
            await session.flush()
        return chunks

    async def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        """Fetch a DocumentChunk by primary chunk_id with joined document and version info."""
        stmt = (
            select(DocumentChunkOrm, DocumentOrm, DocumentVersionOrm)
            .join(DocumentOrm, DocumentChunkOrm.document_id == DocumentOrm.document_id)
            .join(DocumentVersionOrm, DocumentChunkOrm.version_id == DocumentVersionOrm.version_id)
            .where(DocumentChunkOrm.chunk_id == chunk_id)
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            row = result.first()
            if row is None:
                return None
            c_orm, d_orm, v_orm = row
            return self._orm_to_chunk(c_orm, d_orm, v_orm)

    async def get_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        """Alias for list_chunks_for_version conforming to BaseDocumentRepository."""
        return await self.list_chunks_for_version(version_id)

    async def list_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        """List all chunks belonging to a document version in index order."""
        stmt = (
            select(DocumentChunkOrm, DocumentOrm, DocumentVersionOrm)
            .join(DocumentOrm, DocumentChunkOrm.document_id == DocumentOrm.document_id)
            .join(DocumentVersionOrm, DocumentChunkOrm.version_id == DocumentVersionOrm.version_id)
            .where(DocumentChunkOrm.version_id == version_id)
            .order_by(DocumentChunkOrm.chunk_index.asc())
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            rows = result.all()
            return [self._orm_to_chunk(c, d, v) for c, d, v in rows]

    async def get_child_chunks(self, parent_chunk_id: str) -> list[DocumentChunk]:
        """List child chunks of a hierarchical parent chunk."""
        stmt = (
            select(DocumentChunkOrm, DocumentOrm, DocumentVersionOrm)
            .join(DocumentOrm, DocumentChunkOrm.document_id == DocumentOrm.document_id)
            .join(DocumentVersionOrm, DocumentChunkOrm.version_id == DocumentVersionOrm.version_id)
            .where(DocumentChunkOrm.parent_chunk_id == parent_chunk_id)
            .order_by(DocumentChunkOrm.chunk_index.asc())
        )
        async with self._get_session() as session:
            result = await session.execute(stmt)
            rows = result.all()
            return [self._orm_to_chunk(c, d, v) for c, d, v in rows]

    @staticmethod
    def _orm_to_chunk(
        c: DocumentChunkOrm, d: DocumentOrm, v: DocumentVersionOrm
    ) -> DocumentChunk:
        """Helper to construct DocumentChunk domain model from ORM entities."""
        collected_at = _ensure_utc(v.acquired_at)
        assert collected_at is not None
        created_at = _ensure_utc(c.created_at)
        assert created_at is not None

        return DocumentChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            version_id=c.version_id,
            content=c.content,
            chunk_index=c.chunk_index,
            content_hash=c.content_hash,
            parent_chunk_id=c.parent_chunk_id,
            chunk_level=c.chunk_level,
            metadata=ChunkMetadata(
                source_id=d.source_id,
                document_id=d.document_id,
                version_id=v.version_id,
                company_id=d.company_id,
                workspace_id=d.workspace_id,
                source_url=d.canonical_url,
                source_category=c.custom_tags.get("source_category", "website"),
                collected_at=collected_at,
                start_char=c.start_char or 0,
                end_char=c.end_char or 0,
                chunk_index=c.chunk_index,
                total_chunks=c.total_chunks,
                language=c.language,
                custom_tags=c.custom_tags,
            ),
            created_at=created_at,
        )
