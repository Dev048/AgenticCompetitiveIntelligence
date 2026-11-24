"""Unit tests for Source Registry domain models in Source Acquisition subsystem."""

from datetime import datetime, timezone

from pydantic import ValidationError
import pytest

from competitive_intelligence.source_acquisition.models import (
    AccessPattern,
    AcquisitionConfig,
    AcquisitionMethod,
    ContentType,
    DataSensitivity,
    FetchStatus,
    LifecycleStatus,
    MonitoringPolicy,
    PlatformType,
    RateLimitConstraints,
    Source,
    SourceCategory,
    SourceClassification,
    SourceGovernance,
    SourceIdentity,
    SourceLifecycle,
    SourceLocation,
    SourceReliability,
    SourceTier,
    ToolChain,
    ToSStatus,
)


def test_valid_website_source() -> None:
    """Test creating a valid website source (e.g. Anthropic Pricing)."""
    identity = SourceIdentity(
        company_id="company_anthropic",
        workspace_id="workspace_default",
        display_name="Anthropic Pricing Page",
    )
    classification = SourceClassification(
        category=SourceCategory.PRICING,
        platform=PlatformType.WEB,
        content_type=ContentType.HTML,
    )
    location = SourceLocation(
        url_or_identifier="https://www.anthropic.com/pricing",
        access_pattern=AccessPattern.PUBLIC_WEB,
        requires_javascript=False,
    )
    acquisition = AcquisitionConfig(
        acquisition_method=AcquisitionMethod.CRAWL4AI,
        tool_chain=ToolChain(
            primary=AcquisitionMethod.CRAWL4AI,
            fallbacks=[AcquisitionMethod.DIRECT_HTTP],
        ),
        rate_limit_constraints=RateLimitConstraints(requests_per_minute=20),
    )

    source = Source(
        identity=identity,
        classification=classification,
        location=location,
        acquisition=acquisition,
    )

    # Invariants & properties
    assert source.source_id is not None
    assert source.company_id == "company_anthropic"
    assert source.workspace_id == "workspace_default"
    assert source.display_name == "Anthropic Pricing Page"
    assert source.location.url_or_identifier == "https://www.anthropic.com/pricing"
    assert source.acquisition.tool_chain is not None
    assert source.acquisition.tool_chain.primary == AcquisitionMethod.CRAWL4AI
    assert source.acquisition.tool_chain.fallbacks == [AcquisitionMethod.DIRECT_HTTP]
    assert source.monitoring.enabled is True
    assert source.reliability.credibility_score == 1.0
    assert source.reliability.historical_accuracy is None  # Defaults to None for unobserved sources
    assert source.governance.terms_of_service_status == ToSStatus.UNKNOWN  # Defaults to UNKNOWN
    assert source.lifecycle.status == LifecycleStatus.ACTIVE


def test_valid_github_platform_source() -> None:
    """Test creating a valid platform source (e.g. Anthropic GitHub Repository)."""
    source = Source(
        identity=SourceIdentity(
            company_id="company_anthropic",
            workspace_id="workspace_default",
            display_name="Anthropic SDK GitHub Repo",
        ),
        classification=SourceClassification(
            category=SourceCategory.GITHUB,
            platform=PlatformType.GITHUB,
            content_type=ContentType.JSON,
        ),
        location=SourceLocation(
            url_or_identifier="anthropics/anthropic-sdk-python",
            access_pattern=AccessPattern.REPOSITORY,
            requires_javascript=False,
        ),
        acquisition=AcquisitionConfig(
            acquisition_method=AcquisitionMethod.AGENT_REACH,
            tool_chain=ToolChain(
                primary=AcquisitionMethod.AGENT_REACH,
                fallbacks=[AcquisitionMethod.REST_API],
            ),
        ),
        reliability=SourceReliability(
            credibility_score=0.95,
            source_tier=SourceTier.PRIMARY,
            historical_accuracy=0.98,
        ),
    )

    assert source.classification.platform == PlatformType.GITHUB
    assert source.location.url_or_identifier == "anthropics/anthropic-sdk-python"
    assert source.location.access_pattern == AccessPattern.REPOSITORY
    assert source.acquisition.acquisition_method == AcquisitionMethod.AGENT_REACH
    assert source.reliability.credibility_score == 0.95
    assert source.reliability.historical_accuracy == 0.98


def test_invalid_required_identity_fields() -> None:
    """Test that empty or whitespace-only identity fields raise validation error."""
    with pytest.raises(ValidationError):
        SourceIdentity(
            company_id="",
            workspace_id="ws_1",
            display_name="Valid Name",
        )

    with pytest.raises(ValidationError):
        SourceIdentity(
            company_id="comp_1",
            workspace_id="   ",
            display_name="Valid Name",
        )

    with pytest.raises(ValidationError):
        SourceIdentity(
            company_id="comp_1",
            workspace_id="ws_1",
            display_name="",
        )

    with pytest.raises(ValidationError):
        SourceIdentity(
            source_id="   ",
            company_id="comp_1",
            workspace_id="ws_1",
            display_name="Valid Name",
        )


def test_source_reliability_defaults_and_bounds() -> None:
    """Test SourceReliability defaults to None for historical accuracy and enforces score bounds."""
    reliability = SourceReliability()
    assert reliability.credibility_score == 1.0
    assert reliability.source_tier == SourceTier.PRIMARY
    assert reliability.historical_accuracy is None

    # Invalid credibility_score
    with pytest.raises(ValidationError):
        SourceReliability(credibility_score=1.5)

    with pytest.raises(ValidationError):
        SourceReliability(credibility_score=-0.1)

    # Invalid historical_accuracy
    with pytest.raises(ValidationError):
        SourceReliability(historical_accuracy=1.2)

    with pytest.raises(ValidationError):
        SourceReliability(historical_accuracy=-0.01)


def test_source_governance_default_tos_unknown() -> None:
    """Test that terms_of_service_status defaults to UNKNOWN for new unreviewed sources."""
    gov = SourceGovernance()
    assert gov.terms_of_service_status == ToSStatus.UNKNOWN
    assert gov.robots_txt_respected is True
    assert gov.data_sensitivity == DataSensitivity.PUBLIC


def test_negative_failure_count() -> None:
    """Test that consecutive_failure_count cannot be negative."""
    with pytest.raises(ValidationError):
        AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            consecutive_failure_count=-1,
        )

    # Valid non-negative
    config = AcquisitionConfig(
        acquisition_method=AcquisitionMethod.CRAWL4AI,
        consecutive_failure_count=3,
        last_fetch_status=FetchStatus.RATE_LIMITED,
    )
    assert config.consecutive_failure_count == 3


def test_acquisition_config_tool_chain_validation() -> None:
    """Test that tool_chain is synced when omitted and rejects contradictory configurations."""
    # Omitted tool_chain auto-populates matching primary
    config_auto = AcquisitionConfig(acquisition_method=AcquisitionMethod.AGENT_REACH)
    assert config_auto.tool_chain is not None
    assert config_auto.tool_chain.primary == AcquisitionMethod.AGENT_REACH
    assert config_auto.tool_chain.fallbacks == []

    # Valid explicit matching tool_chain
    config_explicit = AcquisitionConfig(
        acquisition_method=AcquisitionMethod.CRAWL4AI,
        tool_chain=ToolChain(
            primary=AcquisitionMethod.CRAWL4AI,
            fallbacks=[AcquisitionMethod.DIRECT_HTTP],
        ),
    )
    assert config_explicit.acquisition_method == AcquisitionMethod.CRAWL4AI
    assert config_explicit.tool_chain is not None
    assert config_explicit.tool_chain.primary == AcquisitionMethod.CRAWL4AI

    # Contradictory tool_chain.primary != acquisition_method
    with pytest.raises(ValidationError, match="Contradictory acquisition configuration"):
        AcquisitionConfig(
            acquisition_method=AcquisitionMethod.CRAWL4AI,
            tool_chain=ToolChain(
                primary=AcquisitionMethod.AGENT_REACH,
            ),
        )


def test_url_validation_for_web_sources() -> None:
    """Test URL validation for web and feed access patterns."""
    # Invalid missing protocol
    with pytest.raises(ValidationError, match="Invalid URL format"):
        SourceLocation(
            url_or_identifier="not-a-valid-url",
            access_pattern=AccessPattern.PUBLIC_WEB,
        )

    # Invalid non-http scheme
    with pytest.raises(ValidationError, match="Invalid URL format"):
        SourceLocation(
            url_or_identifier="ftp://example.com/feed",
            access_pattern=AccessPattern.RSS_FEED,
        )

    # Invalid empty host
    with pytest.raises(ValidationError, match="Invalid URL format"):
        SourceLocation(
            url_or_identifier="https://",
            access_pattern=AccessPattern.PUBLIC_WEB,
        )

    # Valid HTTPS URL
    loc = SourceLocation(
        url_or_identifier="https://competitor.com/blog/feed.xml",
        access_pattern=AccessPattern.RSS_FEED,
    )
    assert loc.url_or_identifier == "https://competitor.com/blog/feed.xml"

    # Non-web identifier allowed for REPOSITORY or COMMUNITY_FEED
    repo_loc = SourceLocation(
        url_or_identifier="owner/repo-name",
        access_pattern=AccessPattern.REPOSITORY,
    )
    assert repo_loc.url_or_identifier == "owner/repo-name"


def test_lifecycle_replacement() -> None:
    """Test updating lifecycle status and tracking replacement source."""
    lifecycle = SourceLifecycle(
        status=LifecycleStatus.REPLACED,
        replaced_by_source_id="source_new_uuid_999",
    )
    assert lifecycle.status == LifecycleStatus.REPLACED
    assert lifecycle.replaced_by_source_id == "source_new_uuid_999"


def test_default_monitoring_state() -> None:
    """Test default values of MonitoringPolicy."""
    policy = MonitoringPolicy()
    assert policy.enabled is True
    assert policy.frequency == 86400
    assert policy.frequency_seconds == 86400
    assert policy.priority == 1
    assert policy.volatility_score == 0.0
    assert policy.last_checked_at is None
    assert policy.last_changed_at is None


def test_monitoring_timezone_awareness() -> None:
    """Test that timestamps in MonitoringPolicy must be timezone-aware."""
    # Timezone-naive datetime
    naive_dt = datetime(2026, 8, 20, 12, 0, 0)
    with pytest.raises(ValidationError, match="timezone-aware"):
        MonitoringPolicy(last_checked_at=naive_dt)

    # Timezone-aware datetime
    aware_dt = datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)
    policy = MonitoringPolicy(last_checked_at=aware_dt, last_changed_at=aware_dt)
    assert policy.last_checked_at == aware_dt
    assert policy.last_changed_at == aware_dt
