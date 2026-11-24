"""Domain models for Knowledge Pipeline subsystem.

Defines canonical document models (Document, DocumentVersion), chunking models,
and ingestion tracking entities.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import urllib.parse
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Canonical Document Domain Models
# ==============================================================================


class Document(BaseModel):
    """Represents the stable logical identity of a tracked document across its versions."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    company_id: str
    workspace_id: str
    canonical_url: str

    @property
    def id(self) -> str:
        """Alias for document_id."""
        return self.document_id

    @property
    def url(self) -> str:
        """Alias for canonical_url."""
        return self.canonical_url

    @property
    def canonical_source_id(self) -> str:
        """Alias for source_id."""
        return self.source_id

    @field_validator("document_id", "source_id", "company_id", "workspace_id")
    @classmethod
    def validate_non_empty_identifiers(cls, value: str) -> str:
        """Ensure required identifiers are non-empty strings."""
        if not value or not value.strip():
            raise ValueError("Identifier cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("canonical_url")
    @classmethod
    def validate_canonical_url(cls, value: str) -> str:
        """Ensure canonical_url is a valid HTTP/HTTPS URL with host/domain."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("canonical_url cannot be empty or whitespace-only")
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"Invalid URL format '{cleaned}'. Expected valid HTTP/HTTPS URL with host/domain."
            )
        return cleaned


# Backwards compatibility and ontology alias
SourceDocument = Document


class DocumentVersion(BaseModel):
    """Represents an immutable snapshot of content, hash, and metadata at a specific timestamp."""

    version_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    content: str
    title: str | None = None
    content_hash: str
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias for version_id."""
        return self.version_id

    @field_validator("version_id", "document_id", "content_hash")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure required string identifiers and hashes are non-empty."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, value: str) -> str:
        """Ensure document content is not empty or whitespace-only."""
        if not value or not value.strip():
            raise ValueError("Document content cannot be empty or whitespace-only")
        return value

    @field_validator("acquired_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure acquired_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


# ==============================================================================
# Change Detection Domain Models
# ==============================================================================


class ChangeType(str, Enum):
    """Classification of content change between consecutive document versions."""

    INITIAL = "initial"
    UNCHANGED = "unchanged"
    CHANGED = "changed"


class ChangeResult(BaseModel):
    """Outcome of comparing a DocumentVersion against its immediately preceding version."""

    document_id: str
    current_version_id: str
    previous_version_id: str | None = None
    change_type: ChangeType
    current_content_hash: str
    previous_content_hash: str | None = None
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("document_id", "current_version_id", "current_content_hash")
    @classmethod
    def validate_non_empty_required_fields(cls, value: str) -> str:
        """Ensure required identifiers and hashes are non-empty strings."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("previous_version_id", "previous_content_hash")
    @classmethod
    def validate_optional_non_empty(cls, value: str | None) -> str | None:
        """Ensure optional identifiers and hashes are non-empty if present."""
        if value is not None and not value.strip():
            raise ValueError("Field cannot be whitespace-only")
        return value.strip() if value is not None else None

    @field_validator("detected_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure detected_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def is_initial(self) -> bool:
        """Whether this is the initial version of the document."""
        return self.change_type == ChangeType.INITIAL

    @property
    def is_changed(self) -> bool:
        """Whether the content changed compared with the previous version."""
        return self.change_type == ChangeType.CHANGED

    @property
    def is_unchanged(self) -> bool:
        """Whether the content is identical to the previous version."""
        return self.change_type == ChangeType.UNCHANGED


# ==============================================================================
# Deduplication Domain Models
# ==============================================================================


class DuplicateType(str, Enum):
    """Type of duplicate relationship identified between documents and versions."""

    NO_DUPLICATE = "no_duplicate"
    EXACT_CONTENT = "exact_content"
    CANONICAL_URL = "canonical_url"
    EXACT_CONTENT_AND_URL = "exact_content_and_url"


class DuplicateMatch(BaseModel):
    """Details of a candidate match against a specific existing document/version."""

    candidate_document_id: str
    candidate_version_id: str | None = None
    match_type: DuplicateType
    matched_canonical_url: bool
    matched_content_hash: bool
    candidate_canonical_url: str
    candidate_content_hash: str | None = None

    @field_validator("candidate_document_id", "candidate_canonical_url")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        """Ensure candidate identifiers and URLs are non-empty."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()


class DeduplicationResult(BaseModel):
    """Outcome of evaluating a Document and DocumentVersion against existing candidate records."""

    document_id: str
    version_id: str
    duplicate_type: DuplicateType
    has_content_duplicate: bool
    has_url_duplicate: bool
    is_duplicate: bool
    matches: list[DuplicateMatch] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("document_id", "version_id")
    @classmethod
    def validate_non_empty_identifiers(cls, value: str) -> str:
        """Ensure document and version IDs are non-empty."""
        if not value or not value.strip():
            raise ValueError("Identifier cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("evaluated_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure evaluated_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def is_unique(self) -> bool:
        """Whether the evaluated document/version is completely unique without any matches."""
        return not self.is_duplicate


# ==============================================================================
# Chunking and Ingestion Models
# ==============================================================================


class ChunkingStrategy(str, Enum):
    """Strategies evaluated for partitioning documents into searchable chunks."""

    WHOLE_DOCUMENT = "whole_document"
    FIXED = "fixed"
    RECURSIVE = "recursive"
    RECURSIVE_CHARACTER = "recursive_character"
    SEMANTIC = "semantic"
    PARENT_CHILD = "parent_child"
    MARKDOWN_HEADER = "markdown_header"
    SENTENCE_WINDOW = "sentence_window"


class StructuralRichness(str, Enum):
    """Level of structural formatting present in document (headings, tables, lists, code)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocumentCharacteristics(BaseModel):
    """Observed structural, categorical, and volumetric features of a document."""

    source_category: str | None = None
    content_type: str | None = None
    char_count: int
    structural_richness: StructuralRichness = StructuralRichness.LOW
    has_semantic_transitions: bool = False

    @field_validator("char_count")
    @classmethod
    def validate_char_count_non_negative(cls, value: int) -> int:
        """Ensure character count is non-negative."""
        if value < 0:
            raise ValueError("char_count cannot be negative")
        return value


class ChunkingParameters(BaseModel):
    """Parameters passed to downstream chunking execution engines."""

    chunk_size: int | None = None
    chunk_overlap: int | None = None
    parent_chunk_size: int | None = None
    child_chunk_size: int | None = None
    preserve_headers: bool = False
    custom_params: dict[str, Any] = Field(default_factory=dict)


class ChunkingPolicyConfig(BaseModel):
    """Configurable thresholds and parameter defaults for chunking policy resolution."""

    short_document_threshold_chars: int = 1000
    fixed_chunk_size: int = 500
    fixed_chunk_overlap: int = 50
    recursive_chunk_size: int = 1000
    recursive_chunk_overlap: int = 150
    parent_chunk_size: int = 2000
    child_chunk_size: int = 400
    parent_child_overlap: int = 50
    semantic_chunk_size: int = 800
    semantic_chunk_overlap: int = 100


class ChunkingPolicy(BaseModel):
    """Deterministic policy resolution deciding chunking strategy and execution parameters."""

    strategy: ChunkingStrategy
    rationale: str
    parameters: ChunkingParameters = Field(default_factory=ChunkingParameters)
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("resolved_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure resolved_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class ChunkMetadata(BaseModel):
    """Provenance and context metadata attached to every document chunk."""

    source_id: str
    document_id: str
    version_id: str
    company_id: str
    workspace_id: str
    source_url: str
    source_category: str
    collected_at: datetime
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    language: str = "en"
    custom_tags: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_id", "document_id", "version_id", "company_id", "workspace_id", "source_url"
    )
    @classmethod
    def validate_non_empty_metadata_strings(cls, value: str) -> str:
        """Ensure required provenance identifiers and URLs are non-empty."""
        if not value or not value.strip():
            raise ValueError("Metadata field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("start_char", "end_char", "chunk_index", "total_chunks")
    @classmethod
    def validate_non_negative_offsets(cls, value: int) -> int:
        """Ensure character offsets and chunk indices are non-negative."""
        if value < 0:
            raise ValueError("Numeric offset/index cannot be negative")
        return value

    @field_validator("collected_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure collected_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


class DocumentChunk(BaseModel):
    """Normalized, searchable unit of text with vector and provenance bindings."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    version_id: str = ""
    content: str
    chunk_index: int = 0
    metadata: ChunkMetadata
    content_hash: str = ""
    parent_chunk_id: str | None = None
    chunk_level: int = 0
    dense_embedding: list[float] | None = None
    sparse_embedding: dict[int, float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def sync_provenance_and_hash(cls, data: Any) -> Any:
        """Sync document_id, version_id, chunk_index from metadata and compute content_hash if needed."""
        if isinstance(data, dict):
            content = data.get("content", "")
            meta = data.get("metadata")
            if isinstance(meta, ChunkMetadata):
                if not data.get("document_id"):
                    data["document_id"] = meta.document_id
                if not data.get("version_id"):
                    data["version_id"] = meta.version_id
                if "chunk_index" not in data or data.get("chunk_index") == 0:
                    data["chunk_index"] = meta.chunk_index
            elif isinstance(meta, dict):
                if not data.get("document_id") and meta.get("document_id"):
                    data["document_id"] = meta["document_id"]
                if not data.get("version_id") and meta.get("version_id"):
                    data["version_id"] = meta["version_id"]
                if ("chunk_index" not in data or data.get("chunk_index") == 0) and meta.get("chunk_index") is not None:
                    data["chunk_index"] = meta["chunk_index"]

            if not data.get("content_hash") and isinstance(content, str) and content.strip():
                import hashlib
                data["content_hash"] = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        return data

    @field_validator("content")
    @classmethod
    def validate_content_non_empty(cls, value: str) -> str:
        """Ensure chunk content is not empty or whitespace-only."""
        if not value or not value.strip():
            raise ValueError("DocumentChunk content cannot be empty or whitespace-only")
        return value

    @field_validator("chunk_id", "document_id", "version_id")
    @classmethod
    def validate_non_empty_identifiers(cls, value: str) -> str:
        """Ensure chunk and provenance identifiers are non-empty."""
        if not value or not value.strip():
            raise ValueError("Identifier cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("parent_chunk_id")
    @classmethod
    def validate_parent_chunk_id(cls, value: str | None) -> str | None:
        """Ensure parent_chunk_id is non-empty if present."""
        if value is not None and not value.strip():
            raise ValueError("parent_chunk_id cannot be whitespace-only")
        return value.strip() if value is not None else None

    @field_validator("chunk_index", "chunk_level")
    @classmethod
    def validate_non_negative_levels(cls, value: int) -> int:
        """Ensure index and hierarchy level are non-negative."""
        if value < 0:
            raise ValueError("Index/level cannot be negative")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure created_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def id(self) -> str:
        """Alias for chunk_id."""
        return self.chunk_id


class IngestionStatus(str, Enum):
    """Status of an ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED_UNCHANGED = "skipped_unchanged"


class IngestionJob(BaseModel):
    """Tracks the end-to-end transformation of a DocumentVersion into vector index chunks."""

    job_id: str = Field(default_factory=lambda: str(uuid4()))
    version_id: str
    document_id: str
    status: IngestionStatus = IngestionStatus.PENDING
    chunks_created: int = 0
    error_message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class IngestionResult(BaseModel):
    """Structured outcome of executing end-to-end knowledge ingestion for a document."""

    document: Document
    version: DocumentVersion
    change_result: ChangeResult
    deduplication_result: DeduplicationResult | None = None
    chunks: list[DocumentChunk] = Field(default_factory=list)
    embeddings_count: int = 0
    vectors_count: int = 0
    is_skipped_unchanged: bool = False
    is_duplicate: bool = False
    status: IngestionStatus = IngestionStatus.COMPLETED
    error_message: str | None = None

    @property
    def document_id(self) -> str:
        """Convenience identifier for the canonical Document."""
        return self.document.document_id

    @property
    def version_id(self) -> str:
        """Convenience identifier for the current DocumentVersion."""
        return self.version.version_id

    @property
    def change_type(self) -> ChangeType:
        """Change classification type."""
        return self.change_result.change_type

    @property
    def is_new_document(self) -> bool:
        """Whether this ingestion created a new canonical document."""
        return self.change_result.is_initial

    @property
    def is_new_version(self) -> bool:
        """Whether this ingestion created a new document version."""
        return self.change_result.is_initial or self.change_result.is_changed

    @property
    def chunks_created_count(self) -> int:
        """Number of chunks created during this ingestion."""
        return len(self.chunks)

    @property
    def vector_points_upserted_count(self) -> int:
        """Number of vector points upserted into vector store."""
        return self.vectors_count

    @property
    def chunk_ids(self) -> list[str]:
        """List of chunk IDs created during this ingestion."""
        return [c.chunk_id for c in self.chunks]



class EmbeddingRecord(BaseModel):
    """Normalized embedding vector and provenance associated with a DocumentChunk."""

    record_id: str = Field(default_factory=lambda: str(uuid4()))
    chunk_id: str
    document_id: str
    version_id: str
    vector: list[float]
    model_name: str
    model_version: str | None = None
    dimensions: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("record_id", "chunk_id", "document_id", "version_id", "model_name")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure string identifiers are not empty or whitespace-only."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("model_version")
    @classmethod
    def validate_optional_model_version(cls, value: str | None) -> str | None:
        """Clean optional model version."""
        if value is not None and not value.strip():
            return None
        return value.strip() if value is not None else None

    @field_validator("dimensions")
    @classmethod
    def validate_positive_dimensions(cls, value: int) -> int:
        """Ensure dimensions count is strictly positive."""
        if value <= 0:
            raise ValueError("dimensions must be a positive integer")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure created_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_vector_length_matches_dimensions(self) -> "EmbeddingRecord":
        """Ensure vector length matches specified dimensions."""
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"Vector length ({len(self.vector)}) does not match declared dimensions ({self.dimensions})"
            )
        return self


class DistanceMetric(str, Enum):
    """Vector similarity metric for vector storage."""

    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"


class VectorStoreConfig(BaseModel):
    """Configuration for vector database collection and client connection."""

    collection_name: str = "ci_knowledge_chunks"
    vector_dimensions: int = 1024
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_model_version: str | None = "1.0.0"
    location: str | None = ":memory:"
    url: str | None = None
    api_key: str | None = None
    path: str | None = None
    timeout: int = 10

    @field_validator("collection_name", "embedding_model_name")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Ensure collection and model names are non-empty."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @field_validator("vector_dimensions")
    @classmethod
    def validate_positive_dims(cls, value: int) -> int:
        """Ensure vector dimensions is strictly positive."""
        if value <= 0:
            raise ValueError("vector_dimensions must be a positive integer")
        return value


class VectorSearchFilter(BaseModel):
    """Payload filter for vector similarity search queries."""

    workspace_id: str | None = None
    company_id: str | None = None
    source_id: str | None = None
    source_category: str | None = None
    content_type: str | None = None
    version_id: str | None = None
    document_id: str | None = None
    parent_chunk_id: str | None = None
    chunk_level: int | None = None


class VectorStoreEntry(BaseModel):
    """Canonical vector point entry prepared for storage in vector database."""

    chunk_id: str
    document_id: str
    version_id: str
    vector: list[float]
    text: str
    source_id: str | None = None
    company_id: str | None = None
    workspace_id: str | None = None
    source_category: str | None = None
    content_type: str | None = None
    canonical_url: str | None = None
    chunk_index: int = 0
    parent_chunk_id: str | None = None
    chunk_level: int = 0
    collected_at: datetime | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "version_id", "text")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        """Ensure core identifiers and text are non-empty."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()

    @classmethod
    def from_chunk_and_embedding(
        cls, chunk: "DocumentChunk", embedding: "EmbeddingRecord"
    ) -> "VectorStoreEntry":
        """Construct a VectorStoreEntry by joining a DocumentChunk and EmbeddingRecord."""
        if chunk.chunk_id != embedding.chunk_id:
            raise ValueError(
                f"Chunk ID mismatch: chunk.chunk_id='{chunk.chunk_id}' vs embedding.chunk_id='{embedding.chunk_id}'"
            )

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            version_id=chunk.version_id,
            vector=embedding.vector,
            text=chunk.content,
            source_id=chunk.metadata.source_id,
            company_id=chunk.metadata.company_id,
            workspace_id=chunk.metadata.workspace_id,
            source_category=chunk.metadata.source_category,
            content_type=chunk.metadata.custom_tags.get("content_type"),
            canonical_url=chunk.metadata.source_url,
            chunk_index=chunk.chunk_index,
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_level=chunk.chunk_level,
            collected_at=chunk.metadata.collected_at,
            title=chunk.metadata.custom_tags.get("title"),
            metadata=dict(chunk.metadata.custom_tags),
        )


class VectorSearchResult(BaseModel):
    """Scored vector retrieval result returned from vector store."""

    chunk_id: str
    score: float
    text: str
    document_id: str
    version_id: str
    source_id: str | None = None
    company_id: str | None = None
    workspace_id: str | None = None
    source_category: str | None = None
    content_type: str | None = None
    canonical_url: str | None = None
    chunk_index: int = 0
    parent_chunk_id: str | None = None
    chunk_level: int = 0
    collected_at: datetime | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchConfiguration(BaseModel):
    """Application-level configuration for vector similarity search and ANN parameters."""

    top_k: int = 10
    hnsw_ef: int | None = None
    exact_search: bool = False
    score_threshold: float | None = None

    @field_validator("top_k")
    @classmethod
    def validate_positive_top_k(cls, value: int) -> int:
        """Ensure top_k is strictly positive."""
        if value <= 0:
            raise ValueError("top_k must be a positive integer")
        return value

    @field_validator("hnsw_ef")
    @classmethod
    def validate_positive_hnsw_ef(cls, value: int | None) -> int | None:
        """Ensure hnsw_ef is positive if provided."""
        if value is not None and value <= 0:
            raise ValueError("hnsw_ef must be a positive integer if specified")
        return value
