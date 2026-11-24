"""Domain models for Source Acquisition and Monitoring subsystem.

Implements the Source Registry domain model representing independently monitored sources.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import urllib.parse
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Enumerations
# ==============================================================================


class SourceCategory(str, Enum):
    """Categorization of external data sources."""

    WEBSITE = "website"
    PRICING = "pricing"
    DOCUMENTATION = "documentation"
    BLOG = "blog"
    CAREERS = "careers"
    GITHUB = "github"
    YOUTUBE = "youtube"
    TWITTER_X = "twitter_x"
    REDDIT = "reddit"
    NEWS = "news"
    OTHER = "other"


class PlatformType(str, Enum):
    """Target platform or ecosystem hosting the source."""

    WEB = "web"
    GITHUB = "github"
    YOUTUBE = "youtube"
    TWITTER_X = "twitter_x"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    RSS = "rss"
    CUSTOM_API = "custom_api"
    OTHER = "other"


class ContentType(str, Enum):
    """Data format of acquired content."""

    HTML = "text/html"
    MARKDOWN = "text/markdown"
    JSON = "application/json"
    TEXT = "text/plain"
    PDF = "application/pdf"


class AccessPattern(str, Enum):
    """Network access pattern or protocol for acquiring content."""

    PUBLIC_WEB = "public_web"
    AUTHENTICATED_WEB = "authenticated_web"
    API = "api"
    RSS_FEED = "rss_feed"
    COMMUNITY_FEED = "community_feed"
    REPOSITORY = "repository"
    STREAM = "stream"


class AcquisitionMethod(str, Enum):
    """Underlying tool or protocol used for data acquisition."""

    CRAWL4AI = "crawl4ai"
    AGENT_REACH = "agent_reach"
    RSS_FEED = "rss_feed"
    REST_API = "rest_api"
    DIRECT_HTTP = "direct_http"


class FetchStatus(str, Enum):
    """Status of the most recent acquisition attempt."""

    NEVER_FETCHED = "never_fetched"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    RATE_LIMITED = "rate_limited"
    FORBIDDEN_OR_BLOCKED = "forbidden_or_blocked"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    PARSING_ERROR = "parsing_error"
    FAILURE = "failure"


class AcquisitionStatus(str, Enum):
    """Status of a single source acquisition attempt."""

    SUCCESS = "success"
    NOT_FOUND = "not_found"
    BLOCKED = "blocked"
    RATE_LIMITED = "rate_limited"
    AUTH_REQUIRED = "auth_required"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    PARSER_ERROR = "parser_error"
    SERVER_ERROR = "server_error"
    FAILURE = "failure"


class SourceTier(str, Enum):
    """Tiering of sources by authority and verification level."""

    PRIMARY = "primary"  # First-party verified (e.g. official pricing page, release notes)
    SECONDARY = "secondary"  # Direct company channel (e.g. official blog, jobs portal)
    COMMUNITY = "community"  # Third-party / public discussion (e.g. Reddit, Twitter, news)


class ToSStatus(str, Enum):
    """Observed terms of service and access status."""

    COMPLIANT = "compliant"
    REVIEW_REQUIRED = "review_required"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


class DataSensitivity(str, Enum):
    """Data sensitivity and confidentiality level."""

    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class LifecycleStatus(str, Enum):
    """Operational lifecycle state of a monitored source."""

    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DEGRADED = "degraded"
    ERROR = "error"
    REPLACED = "replaced"


# Backwards compatibility aliases
GovernanceState = ToSStatus
LifecycleState = LifecycleStatus


# ==============================================================================
# Nested Component Models
# ==============================================================================


class SourceIdentity(BaseModel):
    """Identity and workspace ownership attributes of a source."""

    source_id: str = Field(default_factory=lambda: str(uuid4()))
    company_id: str
    workspace_id: str
    display_name: str

    @field_validator("source_id", "company_id", "workspace_id", "display_name")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Ensure required identifiers are not empty or blank."""
        if not value or not value.strip():
            raise ValueError("Identifier or display name cannot be empty or whitespace-only")
        return value.strip()


class SourceClassification(BaseModel):
    """Classification metadata determining the nature and type of source content."""

    category: SourceCategory
    platform: PlatformType = PlatformType.WEB
    content_type: ContentType = ContentType.HTML


class SourceLocation(BaseModel):
    """Network address or platform identifier and access configuration."""

    url_or_identifier: str
    access_pattern: AccessPattern = AccessPattern.PUBLIC_WEB
    requires_javascript: bool = False

    @field_validator("url_or_identifier")
    @classmethod
    def validate_location_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Location url_or_identifier cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_url_format_if_applicable(self) -> "SourceLocation":
        """Validate URL syntax when access pattern or address indicates a web/feed URL."""
        val = self.url_or_identifier
        parsed = urllib.parse.urlparse(val)

        is_web_pattern = self.access_pattern in (
            AccessPattern.PUBLIC_WEB,
            AccessPattern.AUTHENTICATED_WEB,
            AccessPattern.RSS_FEED,
        )

        if is_web_pattern or parsed.scheme in ("http", "https"):
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError(
                    f"Invalid URL format '{val}' for access pattern '{self.access_pattern.value}'. "
                    "Expected valid HTTP/HTTPS URL with host/domain."
                )
        return self


class ToolChain(BaseModel):
    """Defines primary and fallback acquisition mechanisms."""

    primary: AcquisitionMethod
    fallbacks: list[AcquisitionMethod] = Field(default_factory=list)


class RateLimitConstraints(BaseModel):
    """Rate limit and concurrency boundaries for a source."""

    requests_per_minute: int = Field(default=10, ge=1)
    min_interval_seconds: float = Field(default=0.0, ge=0.0)
    max_concurrent_requests: int = Field(default=1, ge=1)


class AcquisitionConfig(BaseModel):
    """Acquisition strategy, toolchain, and recent fetch metrics."""

    acquisition_method: AcquisitionMethod = AcquisitionMethod.CRAWL4AI
    tool_chain: ToolChain | None = None
    requires_authentication: bool = False
    rate_limit_constraints: RateLimitConstraints = Field(default_factory=RateLimitConstraints)
    last_fetch_status: FetchStatus = FetchStatus.NEVER_FETCHED
    consecutive_failure_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def sync_and_validate_tool_chain(self) -> "AcquisitionConfig":
        """Ensure tool_chain is populated and matches acquisition_method without contradiction."""
        if self.tool_chain is None:
            self.tool_chain = ToolChain(primary=self.acquisition_method)
        elif self.tool_chain.primary != self.acquisition_method:
            raise ValueError(
                f"Contradictory acquisition configuration: acquisition_method '{self.acquisition_method.value}' "
                f"does not match tool_chain.primary '{self.tool_chain.primary.value}'"
            )
        return self


class MonitoringPolicy(BaseModel):
    """Operational monitoring policy and observed volatility."""

    enabled: bool = True
    frequency: int = Field(default=86400, ge=60, description="Monitoring interval in seconds")
    priority: int = Field(default=1, ge=1, le=10)
    volatility_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Observed change frequency [0.0, 1.0]"
    )
    last_checked_at: datetime | None = None
    last_changed_at: datetime | None = None

    @field_validator("last_checked_at", "last_changed_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime | None) -> datetime | None:
        """Ensure timestamp is timezone-aware if present."""
        if value is not None and value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def frequency_seconds(self) -> int:
        """Alias for frequency in seconds."""
        return self.frequency


class SourceReliability(BaseModel):
    """Reliability, credibility, and authority ratings."""

    credibility_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_tier: SourceTier = SourceTier.PRIMARY
    historical_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Observed historical accuracy [0.0, 1.0], None if unobserved",
    )


class SourceGovernance(BaseModel):
    """Acquisition governance policy and data handling status."""

    terms_of_service_status: ToSStatus = ToSStatus.UNKNOWN
    robots_txt_respected: bool = True
    data_sensitivity: DataSensitivity = DataSensitivity.PUBLIC


class SourceLifecycle(BaseModel):
    """Lifecycle tracking and replacement references."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: LifecycleStatus = LifecycleStatus.ACTIVE
    replaced_by_source_id: str | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value


# ==============================================================================
# Composite Source Model
# ==============================================================================


class Source(BaseModel):
    """Represents an independently monitored source.

    Composed of structured nested components. Contains NO document content.
    """

    identity: SourceIdentity
    classification: SourceClassification
    location: SourceLocation
    acquisition: AcquisitionConfig
    monitoring: MonitoringPolicy = Field(default_factory=MonitoringPolicy)
    reliability: SourceReliability = Field(default_factory=SourceReliability)
    governance: SourceGovernance = Field(default_factory=SourceGovernance)
    lifecycle: SourceLifecycle = Field(default_factory=SourceLifecycle)

    # Convenience properties
    @property
    def source_id(self) -> str:
        return self.identity.source_id

    @property
    def company_id(self) -> str:
        return self.identity.company_id

    @property
    def workspace_id(self) -> str:
        return self.identity.workspace_id

    @property
    def display_name(self) -> str:
        return self.identity.display_name


# ==============================================================================
# Acquisition Outcome Model
# ==============================================================================


class AcquisitionResult(BaseModel):
    """Outcome of a single acquisition attempt for a Source.

    Represents the isolated result of acquiring content from a source provider before
    further normalization, versioning, or pipeline processing.
    """

    status: AcquisitionStatus
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    method: AcquisitionMethod | None = None
    provider_name: str | None = None
    status_code: int | None = None
    error_message: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] | None = None

    @field_validator("acquired_at")
    @classmethod
    def validate_timezone_aware(cls, value: datetime) -> datetime:
        """Ensure acquired_at timestamp is timezone-aware."""
        if value.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware")
        return value

    @property
    def is_success(self) -> bool:
        """Whether the acquisition attempt succeeded."""
        return self.status == AcquisitionStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Whether the acquisition attempt failed."""
        return not self.is_success

    @property
    def has_content(self) -> bool:
        """Whether acquired content is present and non-empty."""
        return self.content is not None and len(self.content) > 0
