"""Document change analyzer component for Intelligence Analysis subsystem.

Transforms deterministic DocumentVersion change diffs from the Knowledge Pipeline
into grounded, structured IntelligenceEvent objects.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import difflib
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.intelligence.interfaces import BaseIntelligenceAnalyzer
from competitive_intelligence.intelligence.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    IntelligenceEvent,
    IntelligenceEventType,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TimePrecision,
)
from competitive_intelligence.knowledge_pipeline.models import (
    ChangeResult,
    ChangeType,
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


# ==============================================================================
# Intermediate Interpreter Payloads
# ==============================================================================


class EvidenceSnippet(BaseModel):
    """Raw evidence excerpt extracted by a diff interpreter."""

    extracted_text: str
    is_current_version: bool = True
    role: EvidenceRole = EvidenceRole.PRIMARY_DIRECT
    locator: str | None = None


class RawEventPayload(BaseModel):
    """Intermediate candidate event payload produced by a diff interpreter."""

    event_type: IntelligenceEventType
    title: str
    summary: str
    subject: str
    previous_state: dict[str, Any] | str | None = None
    new_state: dict[str, Any] | str | None = None
    occurred_at: datetime | None = None
    time_precision: TimePrecision = TimePrecision.UNKNOWN
    approximate_period: str | None = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    confidence_rationale: str | None = None
    evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Diff Interpreter Contract & Reference Implementation
# ==============================================================================


class BaseDiffInterpreter(ABC):
    """Abstract contract for interpreting raw text differences into candidate event payloads."""

    @abstractmethod
    async def interpret_diff(
        self,
        previous_content: str | None,
        current_content: str,
        source_category: str,
        source_perspective: SourcePerspective,
        metadata: dict[str, Any],
    ) -> list[RawEventPayload]:
        """Extract candidate event payloads from version content differences.

        Args:
            previous_content: Prior document content (None for INITIAL).
            current_content: Current document content.
            source_category: Source category (e.g., 'pricing', 'news', 'blog').
            source_perspective: Authority/viewpoint classification.
            metadata: Additional document metadata.

        Returns:
            List of structured candidate RawEventPayload objects.
        """
        pass


class RuleBasedDiffInterpreter(BaseDiffInterpreter):
    """Minimal deterministic reference interpreter demonstrating the BaseDiffInterpreter boundary.

    Supports Phase 1 validation scenarios (pricing changes, launches, releases, hiring,
    acquisitions, deprecations, cosmetic changes) without attempting to be an exhaustive
    NLP/regex engine. Designed to be swappable with a future constrained LLM interpreter.
    """

    async def interpret_diff(
        self,
        previous_content: str | None,
        current_content: str,
        source_category: str,
        source_perspective: SourcePerspective,
        metadata: dict[str, Any],
    ) -> list[RawEventPayload]:
        # Handle INITIAL documents (no prior version)
        if previous_content is None:
            return self._interpret_initial_content(
                current_content, source_category, source_perspective, metadata
            )

        events: list[RawEventPayload] = []

        # Check for cosmetic / non-event changes
        if self._is_purely_cosmetic_diff(previous_content, current_content):
            return []

        # 1. Pricing Changes (e.g. $20/mo -> $30/mo or tier price changes)
        pricing_events = self._extract_pricing_changes(previous_content, current_content)
        events.extend(pricing_events)

        # 2. Product Launches / Introductions
        launch_events = self._extract_product_launches(previous_content, current_content)
        events.extend(launch_events)

        # 3. Software Releases & Patch Notes
        release_events = self._extract_release_notes(previous_content, current_content)
        events.extend(release_events)

        # 4. Hiring / Executive Appointments
        hiring_events = self._extract_hiring_updates(previous_content, current_content)
        events.extend(hiring_events)

        # 5. Mergers & Acquisitions
        acquisition_events = self._extract_acquisitions(
            previous_content, current_content, source_perspective
        )
        events.extend(acquisition_events)

        # 6. Product / API Deprecations
        deprecation_events = self._extract_deprecations(previous_content, current_content)
        events.extend(deprecation_events)

        return events

    def _is_purely_cosmetic_diff(self, prev: str, curr: str) -> bool:
        """Check if differences are trivial formatting or copyright updates."""
        prev_clean = re.sub(r"©\s*\d{4}", "", prev).strip()
        curr_clean = re.sub(r"©\s*\d{4}", "", curr).strip()
        # Strip all whitespace for comparison
        prev_no_ws = "".join(prev_clean.split())
        curr_no_ws = "".join(curr_clean.split())
        return prev_no_ws == curr_no_ws

    def _interpret_initial_content(
        self,
        content: str,
        source_category: str,
        source_perspective: SourcePerspective,
        metadata: dict[str, Any],
    ) -> list[RawEventPayload]:
        """Conservatively extract standalone events from newly monitored initial documents."""
        events: list[RawEventPayload] = []

        # Check for standalone release note
        release_match = re.search(r"(?:###\s*)?(v\d+\.\d+(?:\.\d+)?)\b", content, re.IGNORECASE)
        if release_match and ("release" in content.lower() or "changelog" in source_category.lower()):
            version = release_match.group(1)
            date_match = re.search(
                r"(?:released|published|date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
                content,
                re.IGNORECASE,
            )
            occurred_at = None
            precision = TimePrecision.UNKNOWN
            if date_match:
                occurred_at = self._parse_date_string(date_match.group(1))
                if occurred_at:
                    precision = TimePrecision.DATE_ONLY

            events.append(
                RawEventPayload(
                    event_type=IntelligenceEventType.RELEASE,
                    title=f"Release {version}",
                    summary=f"Observable software release {version} documented in initial crawl.",
                    subject=version,
                    occurred_at=occurred_at,
                    time_precision=precision,
                    confidence_score=0.95,
                    confidence_level=ConfidenceLevel.VERY_HIGH,
                    confidence_rationale="Explicit release note header in initial document crawl.",
                    evidence_snippets=[
                        EvidenceSnippet(
                            extracted_text=content[:200].strip(),
                            is_current_version=True,
                            role=EvidenceRole.PRIMARY_DIRECT,
                        )
                    ],
                )
            )

        # Check for standalone press release announcement
        if (
            "introducing" in content.lower()
            or "announces" in content.lower()
            or "announcing" in content.lower()
            or "acquires" in content.lower()
        ):
            intro_match = re.search(
                r"(?:introducing|announcing|announces|announced)\s+([A-Za-z0-9_\-\s]{3,40}?)(?:\s+for|\.|\n|:|,|$)",
                content,
                re.IGNORECASE,
            )
            if intro_match:
                product_name = intro_match.group(1).strip()
                events.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.PRODUCT_LAUNCH,
                        title=f"Announcement: {product_name}",
                        summary=f"Initial announcement regarding {product_name}.",
                        subject=product_name,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Explicit announcement statement in initial crawl.",
                        evidence_snippets=[
                            EvidenceSnippet(
                                extracted_text=intro_match.group(0).strip(),
                                is_current_version=True,
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                    )
                )

        return events

    def _extract_pricing_changes(self, prev: str, curr: str) -> list[RawEventPayload]:
        """Identify price increases, decreases, or tier updates between versions."""
        events: list[RawEventPayload] = []

        # Look for pattern: Tier / Plan name followed by $price
        prev_prices = self._find_plan_prices(prev)
        curr_prices = self._find_plan_prices(curr)

        for plan_name, curr_price in curr_prices.items():
            if plan_name in prev_prices:
                prev_price = prev_prices[plan_name]
                if prev_price["amount"] != curr_price["amount"]:
                    p_amt = prev_price["amount"]
                    c_amt = curr_price["amount"]
                    action = "Increased" if c_amt > p_amt else "Decreased"
                    title = f"{plan_name} Pricing {action} from ${p_amt:.0f} to ${c_amt:.0f}"
                    summary = (
                        f"{plan_name} updated from ${p_amt:.0f}/{prev_price['period']} "
                        f"to ${c_amt:.0f}/{curr_price['period']}."
                    )
                    events.append(
                        RawEventPayload(
                            event_type=IntelligenceEventType.PRICING_CHANGE,
                            title=title,
                            summary=summary,
                            subject=plan_name,
                            previous_state={
                                "price_usd": p_amt,
                                "billing_period": prev_price["period"],
                            },
                            new_state={
                                "price_usd": c_amt,
                                "billing_period": curr_price["period"],
                            },
                            confidence_score=1.0,
                            confidence_level=ConfidenceLevel.VERY_HIGH,
                            confidence_rationale="Deterministic price differential between versions.",
                            evidence_snippets=[
                                EvidenceSnippet(
                                    extracted_text=curr_price["raw_text"],
                                    is_current_version=True,
                                    role=EvidenceRole.PRIMARY_DIRECT,
                                ),
                                EvidenceSnippet(
                                    extracted_text=prev_price["raw_text"],
                                    is_current_version=False,
                                    role=EvidenceRole.PRIMARY_DIRECT,
                                ),
                            ],
                        )
                    )
        return events

    def _find_plan_prices(self, text: str) -> dict[str, dict[str, Any]]:
        """Extract plan names and associated numeric prices from text."""
        results: dict[str, dict[str, Any]] = {}
        # Pattern: (Plan Name) ... $(Amount) / (month|user|yr|mo)
        patterns = [
            r"([A-Za-z0-9\s]{2,20}?)\s*(?:Plan|Tier|Edition)?\s*[:\-–]\s*\$(\d+(?:\.\d{2})?)\s*(?:/|\s*per\s*)?\s*(mo|month|user|yr|year|seat)?",
            r"([A-Za-z0-9\s]{2,20}?)\s+pricing\s+from\s+\$(\d+(?:\.\d{2})?)",
        ]
        for line in text.splitlines():
            for pat in patterns:
                for match in re.finditer(pat, line, re.IGNORECASE):
                    plan_name = match.group(1).strip()
                    amount = float(match.group(2))
                    period = (
                        match.group(3).lower()
                        if len(match.groups()) >= 3 and match.group(3)
                        else "month"
                    )
                    if len(plan_name) >= 2 and not plan_name.lower().startswith("price"):
                        results[plan_name] = {
                            "amount": amount,
                            "period": period,
                            "raw_text": match.group(0).strip(),
                        }
        return results

    def _extract_product_launches(self, prev: str, curr: str) -> list[RawEventPayload]:
        """Detect newly introduced products or major platform offerings."""
        events: list[RawEventPayload] = []
        added_lines = self._get_added_lines(prev, curr)

        for line in added_lines:
            match = re.search(
                r"\b(?:introducing|announcing|announced|launched|launching)\s+([A-Za-z0-9_\-\s]{3,35}?)(?:\.|\n|:|,|$)",
                line,
                re.IGNORECASE,
            )
            if match:
                product_name = match.group(1).strip()
                # Exclude trivial matches like pronouns or common phrases
                if product_name.lower() not in ("a new", "our new", "the new", "new features", "support for"):
                    events.append(
                        RawEventPayload(
                            event_type=IntelligenceEventType.PRODUCT_LAUNCH,
                            title=f"New Product Announcement: {product_name}",
                            summary=f"Announcement regarding {product_name}.",
                            subject=product_name,
                            confidence_score=0.95,
                            confidence_level=ConfidenceLevel.VERY_HIGH,
                            confidence_rationale="Explicit introduction statement in added text.",
                            evidence_snippets=[
                                EvidenceSnippet(
                                    extracted_text=line.strip(),
                                    is_current_version=True,
                                    role=EvidenceRole.PRIMARY_DIRECT,
                                )
                            ],
                        )
                    )
        return events

    def _extract_release_notes(self, prev: str, curr: str) -> list[RawEventPayload]:
        """Extract software releases and changelog updates."""
        events: list[RawEventPayload] = []
        added_lines = self._get_added_lines(prev, curr)

        for line in added_lines:
            version_match = re.search(r"\b(v\d+\.\d+(?:\.\d+)?)\b", line, re.IGNORECASE)
            if version_match:
                version = version_match.group(1)
                date_match = re.search(
                    r"(?:released|published|date)[:\s\-–]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
                    line,
                    re.IGNORECASE,
                )
                occurred_at = None
                precision = TimePrecision.UNKNOWN
                if date_match:
                    occurred_at = self._parse_date_string(date_match.group(1))
                    if occurred_at:
                        precision = TimePrecision.DATE_ONLY

                events.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.RELEASE,
                        title=f"Release {version}",
                        summary=f"Software release {version} published in release notes.",
                        subject=version,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        confidence_score=0.98,
                        confidence_level=ConfidenceLevel.VERY_HIGH,
                        confidence_rationale="Explicit version identifier in changelog diff.",
                        evidence_snippets=[
                            EvidenceSnippet(
                                extracted_text=line.strip(),
                                is_current_version=True,
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                    )
                )
        return events

    def _extract_hiring_updates(self, prev: str, curr: str) -> list[RawEventPayload]:
        """Extract executive or leadership hiring announcements."""
        events: list[RawEventPayload] = []
        added_lines = self._get_added_lines(prev, curr)

        for line in added_lines:
            hiring_match = re.search(
                r"\b(?:welcoming|welcome|appointed|appoints|hired|hires)\s+([A-Za-z\s]{3,30}?)\s+as\s+([A-Za-z0-9_\-\s]{3,40}?)(?:\.|\n|,|$)",
                line,
                re.IGNORECASE,
            )
            if hiring_match:
                person = hiring_match.group(1).strip()
                title = hiring_match.group(2).strip()
                events.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
                        title=f"Leadership Appointment: {person} as {title}",
                        summary=f"Appointed {person} to the position of {title}.",
                        subject=title,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Explicit executive appointment announcement.",
                        evidence_snippets=[
                            EvidenceSnippet(
                                extracted_text=line.strip(),
                                is_current_version=True,
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                    )
                )
        return events

    def _extract_acquisitions(
        self, prev: str, curr: str, perspective: SourcePerspective
    ) -> list[RawEventPayload]:
        """Extract acquisition announcements."""
        events: list[RawEventPayload] = []
        added_lines = self._get_added_lines(prev, curr)

        for line in added_lines:
            acq_match = re.search(
                r"\b([A-Za-z0-9_\-\s]{2,30}?)\s+(?:acquires|acquired|to acquire|acquisition of)\s+([A-Za-z0-9_\-\s]{2,30}?)(?:\s+for\s+(\$[\d\.]+[MBKmbk]?))?(?:\.|\n|,|$)",
                line,
                re.IGNORECASE,
            )
            if acq_match:
                acquirer = acq_match.group(1).strip()
                target = acq_match.group(2).strip()
                price = acq_match.group(3) if len(acq_match.groups()) >= 3 else None
                summary = f"{acquirer} acquired {target}" + (f" for {price}." if price else ".")
                confidence_score = 0.95 if perspective == SourcePerspective.FIRST_PARTY else 0.85
                confidence_level = (
                    ConfidenceLevel.VERY_HIGH
                    if perspective == SourcePerspective.FIRST_PARTY
                    else ConfidenceLevel.HIGH
                )
                events.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.ACQUISITION,
                        title=f"Acquisition: {acquirer} Acquired {target}",
                        summary=summary,
                        subject=target,
                        new_state={"acquirer": acquirer, "target": target, "deal_size": price} if price else None,
                        confidence_score=confidence_score,
                        confidence_level=confidence_level,
                        confidence_rationale="Grounded acquisition statement in document diff.",
                        evidence_snippets=[
                            EvidenceSnippet(
                                extracted_text=line.strip(),
                                is_current_version=True,
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                    )
                )
        return events

    def _extract_deprecations(self, prev: str, curr: str) -> list[RawEventPayload]:
        """Extract product or API deprecation notices."""
        events: list[RawEventPayload] = []
        added_lines = self._get_added_lines(prev, curr)

        for line in added_lines:
            dep_match = re.search(
                r"\b(?:deprecating|deprecated|deprecation of|sunsetting|sunset of)\s+([A-Za-z0-9_\-\s]{3,35}?)(?:\s+in favor of|\.|\n|,|$)",
                line,
                re.IGNORECASE,
            )
            if dep_match:
                feature = dep_match.group(1).strip()
                events.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.PRODUCT_DEPRECATION,
                        title=f"Deprecation Notice: {feature}",
                        summary=f"Deprecation notice issued for {feature}.",
                        subject=feature,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Explicit deprecation notice in version diff.",
                        evidence_snippets=[
                            EvidenceSnippet(
                                extracted_text=line.strip(),
                                is_current_version=True,
                                role=EvidenceRole.PRIMARY_DIRECT,
                            )
                        ],
                    )
                )
        return events

    def _get_added_lines(self, prev: str, curr: str) -> list[str]:
        """Return lines present in curr that were not present in prev."""
        prev_lines = {line.strip() for line in prev.splitlines() if line.strip()}
        curr_lines = [line.strip() for line in curr.splitlines() if line.strip()]
        return [line for line in curr_lines if line not in prev_lines]

    def _parse_date_string(self, date_str: str) -> datetime | None:
        """Parse common date formats into timezone-aware UTC datetime."""
        clean = date_str.strip().replace(",", "")
        formats = [
            "%B %d %Y",
            "%b %d %Y",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(clean, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None


# ==============================================================================
# Document Change Analyzer Component
# ==============================================================================


class DocumentChangeAnalyzer(BaseIntelligenceAnalyzer):
    """Production component transforming document version changes into grounded IntelligenceEvents.

    Adheres strictly to the Knowledge Pipeline authoritative contracts and grounding rules:
    - Never modifies or fabricates provenance identifiers.
    - Preserves strict separation between observed facts, factual confidence, and strategic significance.
    - Guarantees: No evidence -> no event.
    """

    def __init__(self, diff_interpreter: BaseDiffInterpreter | None = None) -> None:
        self.diff_interpreter = diff_interpreter or RuleBasedDiffInterpreter()

    async def analyze_document_change(
        self,
        change: ChangeResult,
        current_version: DocumentVersion,
        previous_version: DocumentVersion | None,
        context_evidence: list[RetrievalResult] | None = None,
    ) -> list[IntelligenceEvent]:
        """Analyze a document version change and extract structured IntelligenceEvents."""
        # 1. Authority Verification: UNCHANGED documents emit 0 events
        if change.change_type == ChangeType.UNCHANGED:
            return []

        if previous_version is not None and current_version.content_hash == previous_version.content_hash:
            return []

        # 2. Extract Document & Provenance Metadata
        metadata = current_version.metadata
        workspace_id = metadata.get("workspace_id", "default_workspace")
        company_id = metadata.get("company_id", "unknown_company")
        source_id = metadata.get("source_id", "unknown_source")
        canonical_url = metadata.get("canonical_url") or metadata.get("source_url") or "https://example.com"
        source_category = metadata.get("source_category", "other")
        source_perspective = self._resolve_source_perspective(metadata)

        previous_content = previous_version.content if previous_version else None
        current_content = current_version.content

        # 3. Interpret Content Differences
        raw_payloads = await self.diff_interpreter.interpret_diff(
            previous_content=previous_content,
            current_content=current_content,
            source_category=source_category,
            source_perspective=source_perspective,
            metadata=metadata,
        )

        # 4. Assemble Grounded Intelligence Events
        intelligence_events: list[IntelligenceEvent] = []

        for payload in raw_payloads:
            # Enforce Grounding Invariant: No Evidence -> No Event
            if not payload.evidence_snippets:
                continue

            evidence_refs: list[EvidenceReference] = []
            for snippet in payload.evidence_snippets:
                is_curr = snippet.is_current_version
                target_ver = current_version if is_curr else (previous_version or current_version)
                evidence_refs.append(
                    EvidenceReference(
                        source_id=source_id,
                        document_id=target_ver.document_id,
                        version_id=target_ver.version_id,
                        source_url=canonical_url,
                        source_category=source_category,
                        source_perspective=source_perspective,
                        extracted_text=snippet.extracted_text,
                        acquired_at=target_ver.acquired_at,
                        role=snippet.role,
                    )
                )

            # Build EventTime
            event_time = EventTime(
                occurred_at=payload.occurred_at,
                precision=payload.time_precision,
                approximate_period=payload.approximate_period,
                acquired_at=current_version.acquired_at,
                detected_at=datetime.now(timezone.utc),
            )

            # Build Confidence & Significance Assessments
            confidence = ConfidenceAssessment(
                score=payload.confidence_score,
                level=payload.confidence_level,
                rationale=payload.confidence_rationale,
                is_corroborated=len(evidence_refs) > 1,
            )
            significance = SignificanceAssessment(
                status=SignificanceStatus.UNASSESSED,
                level=SignificanceLevel.UNKNOWN,
            )

            event = IntelligenceEvent(
                workspace_id=workspace_id,
                company_id=company_id,
                event_type=payload.event_type,
                title=payload.title,
                summary=payload.summary,
                subject=payload.subject,
                previous_state=payload.previous_state,
                new_state=payload.new_state,
                event_time=event_time,
                confidence=confidence,
                significance=significance,
                evidence=evidence_refs,
                primary_perspective=source_perspective,
                metadata=payload.metadata,
            )
            intelligence_events.append(event)

        return intelligence_events

    async def analyze_retrieved_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
    ) -> list[IntelligenceEvent]:
        """Subsystem 5 Phase 2 capability placeholder."""
        return []

    def _resolve_source_perspective(self, metadata: dict[str, Any]) -> SourcePerspective:
        """Resolve SourcePerspective enum from version metadata or fallback."""
        persp_val = metadata.get("source_perspective")
        if isinstance(persp_val, SourcePerspective):
            return persp_val
        if isinstance(persp_val, str):
            try:
                return SourcePerspective(persp_val.lower())
            except ValueError:
                pass

        # Infer from category
        cat = metadata.get("source_category", "").lower()
        if cat in ("website", "pricing", "documentation", "blog", "careers", "github"):
            return SourcePerspective.FIRST_PARTY
        if cat in ("news", "press_release"):
            return SourcePerspective.THIRD_PARTY_NEWS
        if cat in ("reddit", "twitter_x", "community"):
            return SourcePerspective.COMMUNITY_DISCUSSION
        return SourcePerspective.UNKNOWN
