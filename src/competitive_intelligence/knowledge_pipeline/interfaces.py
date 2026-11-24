"""Abstract interfaces for Knowledge Pipeline subsystem.

Defines the formal contracts for normalization, deduplication, change detection,
chunking, embeddings generation, vector storage, and ingestion orchestration.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    ChunkingParameters,
    ChunkingPolicy,
    ChunkingStrategy,
    DeduplicationResult,
    Document,
    DocumentCharacteristics,
    DocumentChunk,
    DocumentVersion,
    IngestionJob,
    SearchConfiguration,
    VectorSearchFilter,
    VectorStoreEntry,
    VectorSearchResult,
)
from competitive_intelligence.source_acquisition.models import AcquisitionResult, Source


class BaseChangeDetector(ABC):
    """Abstract interface for detecting changes between consecutive document versions."""

    @abstractmethod
    def detect_change(
        self,
        current_version: DocumentVersion,
        previous_version: DocumentVersion | None = None,
    ) -> ChangeResult:
        """Compare current version with immediately preceding version."""
        pass


class BaseChunkingPolicyResolver(ABC):
    """Abstract interface for resolving chunking strategy based on document characteristics."""

    @abstractmethod
    def resolve_policy(
        self, characteristics: DocumentCharacteristics
    ) -> ChunkingPolicy:
        """Resolve the optimal chunking policy for the given document characteristics."""
        pass


class BaseDocumentChunker(ABC):
    """Abstract interface for chunking a DocumentVersion into DocumentChunks using specified parameters."""

    @abstractmethod
    def chunk(
        self,
        version: DocumentVersion,
        parameters: ChunkingParameters | None = None,
        source: Source | None = None,
    ) -> list[DocumentChunk]:
        """Partition DocumentVersion content into normalized DocumentChunk units with preserved provenance."""
        pass


# Domain alias for the chunker abstraction
DocumentChunker = BaseDocumentChunker


class BaseDocumentDeduplicator(ABC):
    """Abstract interface for deterministic document deduplication."""

    @abstractmethod
    def find_duplicates(
        self,
        document: Document,
        version: DocumentVersion,
        candidates: Sequence[tuple[Document, DocumentVersion]],
    ) -> DeduplicationResult:
        """Evaluate candidate documents and versions for URL and content duplicates."""
        pass


class BaseDocumentNormalizer(ABC):
    """Abstract interface for transforming Source and AcquisitionResult into canonical Document and DocumentVersion."""

    @abstractmethod
    def normalize(
        self, source: Source, result: AcquisitionResult
    ) -> tuple[Document, DocumentVersion]:
        """Normalize acquired snapshot into a Document and DocumentVersion pair."""
        pass


class BaseDocumentRepository(ABC):
    """Abstract repository for Documents, Versions, and Chunks."""

    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Create canonical document identity."""
        pass

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        """Fetch canonical document by ID."""
        pass

    @abstractmethod
    async def get_document_by_canonical_url(
        self, source_id: str, canonical_url: str
    ) -> Document | None:
        """Fetch document by source and canonical URL."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        workspace_id: str,
        company_id: str | None = None,
        source_id: str | None = None,
    ) -> list[Document]:
        """List documents in workspace."""
        pass

    @abstractmethod
    async def create_version(self, version: DocumentVersion) -> DocumentVersion:
        """Persist immutable document version snapshot."""
        pass

    @abstractmethod
    async def get_version(self, version_id: str) -> DocumentVersion | None:
        """Fetch version snapshot by ID."""
        pass

    @abstractmethod
    async def get_latest_version(self, document_id: str) -> DocumentVersion | None:
        """Fetch latest version for a document."""
        pass

    @abstractmethod
    async def get_version_by_hash(
        self, document_id: str, content_hash: str
    ) -> DocumentVersion | None:
        """Fetch version by content hash."""
        pass

    @abstractmethod
    async def list_versions(self, document_id: str) -> list[DocumentVersion]:
        """List version history snapshots for a document."""
        pass

    @abstractmethod
    async def create_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Persist document chunks."""
        pass

    @abstractmethod
    async def get_chunks_for_version(self, version_id: str) -> list[DocumentChunk]:
        """List chunks for a specific version."""
        pass

    @abstractmethod
    async def get_child_chunks(self, parent_chunk_id: str) -> list[DocumentChunk]:
        """List child chunks of a parent chunk."""
        pass


class BaseNormalizer(ABC):
    """Abstract interface for cleaning and standardizing raw document content."""

    @abstractmethod
    def normalize(self, raw_content: str, content_type: str) -> str:
        """Strip boilerplate and convert content to clean markdown/text."""
        pass


class BaseDeduplicator(ABC):
    """Abstract interface for hash and semantic deduplication."""

    @abstractmethod
    async def is_duplicate_or_unchanged(
        self, content_hash: str, document_id: str
    ) -> bool:
        """Check if content hash matches existing version."""
        pass


class BaseChunker(ABC):
    """Abstract interface for partitioning normalized text into DocumentChunks."""

    @abstractmethod
    def chunk(
        self,
        document_version: Any,
        source: Source,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    ) -> list[DocumentChunk]:
        """Split document into chunks with preserved provenance offsets."""
        pass


class BaseEmbeddingProvider(ABC):
    """Abstract interface for provider-agnostic dense text embedding generation."""

    @property
    def model_name(self) -> str:
        """Name/identifier of the underlying embedding model."""
        return "base_embedding_provider"

    @property
    def model_version(self) -> str | None:
        """Version of the underlying embedding model if available."""
        return None

    @property
    def dimensions(self) -> int | None:
        """Vector dimensionality of the embedding model if known in advance."""
        return None

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute dense vector embeddings for a sequence of texts.

        Args:
            texts: Sequence of string texts to embed.

        Returns:
            A list of float vectors with length equal to len(texts).
        """
        pass

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Convenience alias for embed_texts."""
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        """Compute dense vector embedding for a single search query or text."""
        results = self.embed_texts([text])
        if not results:
            raise ValueError("Embedding provider returned empty result for query")
        return results[0]

    async def aembed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Async variant for computing dense vector embeddings."""
        return self.embed_texts(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async variant for computing a single text embedding."""
        return self.embed_query(text)


# Domain aliases
EmbeddingProvider = BaseEmbeddingProvider
BaseEmbedder = BaseEmbeddingProvider


class BaseVectorStore(ABC):
    """Abstract interface for vector database operations (e.g. Qdrant)."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create or verify the vector collection exists with expected dimensions and metric."""
        pass

    @abstractmethod
    def upsert_entries(self, entries: Sequence[VectorStoreEntry]) -> None:
        """Upsert a batch of VectorStoreEntry points into the collection."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
        search_config: SearchConfiguration | None = None,
    ) -> list[VectorSearchResult]:
        """Perform vector similarity search against the collection."""
        pass

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> VectorSearchResult | None:
        """Retrieve a single vector entry by its chunk_id."""
        pass

    @abstractmethod
    def delete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        """Delete specific points by their chunk_ids."""
        pass

    @abstractmethod
    def delete_by_filter(self, filters: VectorSearchFilter) -> None:
        """Delete points matching payload filter conditions."""
        pass

    async def aensure_collection(self) -> None:
        """Async variant for ensuring collection."""
        return self.ensure_collection()

    async def aupsert_entries(self, entries: Sequence[VectorStoreEntry]) -> None:
        """Async variant for upserting entries."""
        return self.upsert_entries(entries)

    async def asearch(
        self,
        query_vector: Sequence[float],
        top_k: int = 10,
        filters: VectorSearchFilter | None = None,
        search_config: SearchConfiguration | None = None,
    ) -> list[VectorSearchResult]:
        """Async variant for vector search."""
        return self.search(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            search_config=search_config,
        )

    async def aget_by_id(self, chunk_id: str) -> VectorSearchResult | None:
        """Async variant for retrieving point by id."""
        return self.get_by_id(chunk_id)

    async def adelete_by_ids(self, chunk_ids: Sequence[str]) -> None:
        """Async variant for deleting points by ids."""
        return self.delete_by_ids(chunk_ids)

    async def adelete_by_filter(self, filters: VectorSearchFilter) -> None:
        """Async variant for deleting points by filter."""
        return self.delete_by_filter(filters)


# Domain aliases
VectorStore = BaseVectorStore


class BasePipelineOrchestrator(ABC):
    """Abstract interface for running the knowledge ingestion pipeline."""

    @abstractmethod
    async def process_version(
        self, document_version: Any, source: Source
    ) -> IngestionJob:
        """Execute full transformation: normalize -> dedup -> chunk -> embed -> index."""
        pass
