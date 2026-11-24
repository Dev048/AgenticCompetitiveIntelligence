"""Retrieved evidence analyzer component for Intelligence Analysis subsystem.

Transforms search / topic retrieval evidence chunks into grounded, structured
IntelligenceEvent objects, performing deterministic clustering and evidence role assignment.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
import re
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient
from competitive_intelligence.core.budgets import LLMBudgetTracker
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
    DocumentVersion,
)
from competitive_intelligence.retrieval_engine.models import RetrievalResult


# ==============================================================================
# Intermediate Evidence Snippets & Payloads
# ==============================================================================


class RetrievedEvidenceSnippet(BaseModel):
    """Evidence snippet extracted from a specific RetrievalResult with full provenance."""

    chunk_id: str
    document_id: str
    version_id: str
    source_id: str = "unknown_source"
    source_url: str = "https://example.com"
    source_category: str = "other"
    source_perspective: SourcePerspective = SourcePerspective.UNKNOWN
    extracted_text: str
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrieval_score: float | None = None
    role: EvidenceRole = EvidenceRole.PRIMARY_DIRECT
    locator: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawEventPayload(BaseModel):
    """Intermediate candidate event payload produced by an evidence interpreter."""

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
    evidence_snippets: list[RetrievedEvidenceSnippet] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Event Signature Strategy (Deterministic Deduplication Boundary)
# ==============================================================================


class BaseEventSignatureStrategy(ABC):
    """Abstract strategy for computing deterministic clustering signatures for candidate events."""

    @abstractmethod
    def compute_signature(self, payload: RawEventPayload, company_id: str) -> str:
        """Generate a deterministic clustering signature string."""
        pass


class DeterministicEventSignatureStrategy(BaseEventSignatureStrategy):
    """Default deterministic signature combining event_type, normalized subject, temporal key, and state key."""

    def compute_signature(self, payload: RawEventPayload, company_id: str) -> str:
        # Normalize subject (alphanumeric only)
        norm_subj = re.sub(r"[^a-z0-9]", "", payload.subject.lower().strip())

        # Temporal key: exact date, coarse period, or unknown
        if payload.occurred_at is not None:
            temp_key = payload.occurred_at.strftime("%Y-%m-%d")
        elif payload.approximate_period:
            temp_key = re.sub(r"[^a-z0-9]", "", payload.approximate_period.lower().strip())
        else:
            temp_key = "unknown_time"

        # State key for stateful changes (e.g. pricing)
        state_key = ""
        if payload.event_type == IntelligenceEventType.PRICING_CHANGE and isinstance(payload.new_state, dict):
            state_key = str(payload.new_state.get("price_usd", ""))

        return f"{company_id.strip().lower()}::{payload.event_type.value}::{norm_subj}::{temp_key}::{state_key}"


# ==============================================================================
# Evidence Interpreter Contract & Reference Implementations
# ==============================================================================


class BaseEvidenceInterpreter(ABC):
    """Abstract contract for interpreting retrieved chunks into candidate event payloads."""

    @abstractmethod
    async def interpret_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[RawEventPayload]:
        """Extract candidate event payloads from retrieved evidence chunks."""
        pass


class RuleBasedEvidenceInterpreter(BaseEvidenceInterpreter):
    """Minimal deterministic reference interpreter.

    Demonstrates the BaseEvidenceInterpreter boundary without requiring external ML/LLMs.
    Extracts announcements, releases, hiring, acquisitions, pricing changes, partnerships, reviews, and community buzz.
    """

    async def interpret_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[RawEventPayload]:
        candidate_payloads: list[RawEventPayload] = []

        for item in evidence:
            text = item.text.strip()
            if not text:
                continue

            # Extract source perspective and category
            perspective = self._resolve_perspective(item)
            category = item.source_category or "other"

            # Parse temporal info
            occurred_at, precision, approx_period = self._extract_temporal_info(text)

            # Build base snippet with full provenance and metadata
            base_snippet = RetrievedEvidenceSnippet(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                version_id=item.version_id,
                source_id=item.source_id or "unknown_source",
                source_url=item.canonical_url or "https://example.com",
                source_category=category,
                source_perspective=perspective,
                extracted_text=text,
                retrieval_score=item.score,
                role=EvidenceRole.PRIMARY_DIRECT,
                metadata=dict(item.metadata),
            )

            # 1. Conflicting / Delay Statements
            if self._is_contradictory_statement(text):
                delay_match = re.search(
                    r"\b(?:delayed|postponed|cancelled|delay of|rollout delayed)(?:\s+for|\s+of)?\s+([A-Za-z0-9_\-\s]{3,30}?)(?:\s+due to|\s+pending|\.|\n|,|$)",
                    text,
                    re.IGNORECASE,
                )
                raw_subject = delay_match.group(1).strip() if delay_match else "Feature Rollout"
                subject = re.sub(r"^(?:for|of)\s+", "", raw_subject, flags=re.IGNORECASE).strip()
                base_snippet.role = EvidenceRole.CONTRADICTING
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.PRODUCT_CHANGE,
                        title=f"Disputed / Delayed Rollout: {subject}",
                        summary=f"Report indicates delay or postponement regarding {subject}.",
                        subject=subject,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.60,
                        confidence_level=ConfidenceLevel.LOW,
                        confidence_rationale="Contradicting statement reporting rollout delay.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue


            # 2. Pricing Changes (State Transitions)
            price_match = re.search(
                r"\b(?:reverted|updated|changed|lowered|reduced|increased|raised|cut)\s+([A-Za-z0-9_\-\s]{3,30}?)\s+(?:pricing|price|tier|plan)?\s*(?:from\s+(\$[\d\.]+|\d+)\s+)?(?:back\s+)?to\s+(\$[\d\.]+|\d+)",
                text,
                re.IGNORECASE,
            )
            if not price_match:
                price_match = re.search(
                    r"\b(?:prices?|pricing|tier|plan)\s+(?:for\s+([A-Za-z0-9_\-\s]{3,30}?)\s+)?(?:reverted|updated|changed|increased|decreased|dropped|raised|cut)\s+(?:from\s+(\$[\d\.]+|\d+)\s+)?(?:back\s+)?to\s+(\$[\d\.]+|\d+)",
                    text,
                    re.IGNORECASE,
                )
            if not price_match:
                price_match = re.search(
                    r"\b(?:announced|introduced|updated|new)\s+([A-Za-z0-9_\-\s]{3,30}?)\s+(?:plan\s+)?pricing\s+at\s+(\$[\d\.]+(?:/\w+)?)",
                    text,
                    re.IGNORECASE,
                )
            if price_match:
                groups = price_match.groups()
                subj = (groups[0] or "Enterprise Plan").strip()
                if len(groups) >= 3 and groups[1] and groups[2]:
                    prev_p = groups[1].strip()
                    new_p = groups[2].strip()
                elif len(groups) >= 2 and groups[1]:
                    prev_p = None
                    new_p = groups[1].strip()
                else:
                    prev_p = None
                    new_p = groups[0].strip()

                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.PRICING_CHANGE,
                        title=f"Pricing Adjustment: {subj}",
                        summary=f"Pricing for {subj} adjusted" + (f" from {prev_p} to {new_p}." if prev_p else f" to {new_p}."),
                        subject=subj,
                        previous_state={"price_usd": prev_p} if prev_p else None,
                        new_state={"price_usd": new_p},
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Grounded pricing change statement in evidence chunk.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 3. Product Announcements / Launches
            if (
                "introducing" in text.lower()
                or "announcing" in text.lower()
                or "announced" in text.lower()
                or "launched" in text.lower()
            ):
                intro_match = re.search(
                    r"\b(?:introducing|announcing|announced|launched|launching)\s+([A-Za-z0-9_\-\s]{3,35}?)(?:\s+for|\.|\n|:|,|$)",
                    text,
                    re.IGNORECASE,
                )
                if intro_match:
                    product_name = intro_match.group(1).strip()
                    if product_name.lower() not in ("a new", "our new", "the new", "new features", "support for"):
                        candidate_payloads.append(
                            RawEventPayload(
                                event_type=IntelligenceEventType.PRODUCT_LAUNCH,
                                title=f"New Product Announcement: {product_name}",
                                summary=f"Announcement regarding {product_name}.",
                                subject=product_name,
                                occurred_at=occurred_at,
                                time_precision=precision,
                                approximate_period=approx_period,
                                confidence_score=0.95,
                                confidence_level=ConfidenceLevel.VERY_HIGH,
                                confidence_rationale="Direct product announcement statement in evidence text.",
                                evidence_snippets=[base_snippet],
                                metadata={"extraction_source": "rule_based", "fallback_used": False},
                            )
                        )
                        continue

            # 4. Software Releases
            release_match = re.search(r"\b(v\d+\.\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
            if release_match and ("release" in text.lower() or "changelog" in category.lower()):
                version = release_match.group(1)
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.RELEASE,
                        title=f"Release {version}",
                        summary=f"Software release {version} documented in retrieved evidence.",
                        subject=version,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.98,
                        confidence_level=ConfidenceLevel.VERY_HIGH,
                        confidence_rationale="Grounded version identifier in retrieved text.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 5. Executive Hiring
            hiring_match = re.search(
                r"\b(?:welcoming|welcome|appointed|appoints|hired|hires)\s+([A-Za-z\s]{3,30}?)\s+as\s+([A-Za-z0-9_\-\s]{3,40}?)(?:\.|\n|,|$)",
                text,
                re.IGNORECASE,
            )
            if hiring_match:
                person = hiring_match.group(1).strip()
                title = hiring_match.group(2).strip()
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.HIRING_OR_HEADCOUNT,
                        title=f"Leadership Appointment: {person} as {title}",
                        summary=f"Appointed {person} to the position of {title}.",
                        subject=title,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Executive hiring appointment in evidence chunk.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 6. Mergers & Acquisitions
            acq_match = re.search(
                r"\b(?:acquires|acquired|to acquire|acquisition of)\s+([A-Za-z0-9_\-\s]{2,30}?)(?:\s+for\s+(\$[\d\.]+[MBKmbk]?))?(?:\s+\bon\b|\s+\bto\b|\s+\bin\b|\.|\n|,|$)",
                text,
                re.IGNORECASE,
            )

            if acq_match:
                target = acq_match.group(1).strip()
                price = acq_match.group(2) if len(acq_match.groups()) >= 2 else None
                acquirer = company_id.replace("_", " ").title()
                summary = f"{acquirer} acquired {target}" + (f" for {price}." if price else ".")
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.ACQUISITION,
                        title=f"Acquisition: {acquirer} Acquired {target}",
                        summary=summary,
                        subject=target,
                        new_state={"acquirer": acquirer, "target": target, "deal_size": price} if price else None,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Acquisition statement in evidence chunk.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 7. Strategic Partnerships
            partner_match = re.search(
                r"\b(?:partnered with|partners with|strategic partnership with)\s+([A-Za-z0-9_\-\s]{3,35}?)(?:\.|\n|,|$)",
                text,
                re.IGNORECASE,
            )
            if partner_match:
                partner_name = partner_match.group(1).strip()
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.PARTNERSHIP,
                        title=f"Partnership with {partner_name}",
                        summary=f"Established partnership with {partner_name}.",
                        subject=partner_name,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.90,
                        confidence_level=ConfidenceLevel.HIGH,
                        confidence_rationale="Partnership announcement in evidence chunk.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 8. Third-Party Reviews & Product Opinions
            if perspective == SourcePerspective.EXTERNAL_OPINION or "review" in category.lower():
                opinion_match = re.search(
                    r"\b([A-Za-z0-9_\-\s]{3,30}?)\s+(?:review|feedback|benchmark|performance|sluggish|excellent|ratings?)(?:\.|\n|:|,|$)",
                    text,
                    re.IGNORECASE,
                )
                subj = opinion_match.group(1).strip() if opinion_match else "Product Performance"
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.EXTERNAL_OPINION,
                        title=f"External Review on {subj}",
                        summary=f"Third-party user review / commentary on {subj}.",
                        subject=subj,
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.75,
                        confidence_level=ConfidenceLevel.MEDIUM,
                        confidence_rationale="Third-party user review / opinion in evidence chunk.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

            # 9. Community Discussion & Forum Speculation
            if perspective == SourcePerspective.COMMUNITY_DISCUSSION or "reddit" in category.lower():
                candidate_payloads.append(
                    RawEventPayload(
                        event_type=IntelligenceEventType.OTHER,
                        title="Community Discussion / Market Speculation",
                        summary=f"Community sentiment and discussion: {text[:100]}...",
                        subject="Community Discussion",
                        occurred_at=occurred_at,
                        time_precision=precision,
                        approximate_period=approx_period,
                        confidence_score=0.50,
                        confidence_level=ConfidenceLevel.LOW,
                        confidence_rationale="Community forum post / speculative discussion.",
                        evidence_snippets=[base_snippet],
                        metadata={"extraction_source": "rule_based", "fallback_used": False},
                    )
                )
                continue

        return candidate_payloads

    def _is_contradictory_statement(self, text: str) -> bool:
        """Check if text contains language indicating a delay or contradiction."""
        t_low = text.lower()
        return "delayed" in t_low or "postponed" in t_low or "cancelled" in t_low or "rollout delay" in t_low

    def _resolve_perspective(self, item: RetrievalResult) -> SourcePerspective:
        """Resolve SourcePerspective from result metadata or category."""
        persp = item.metadata.get("source_perspective")
        if isinstance(persp, SourcePerspective):
            return persp
        if isinstance(persp, str):
            try:
                return SourcePerspective(persp.lower())
            except ValueError:
                pass

        cat = (item.source_category or "").lower()
        if cat in ("website", "pricing", "documentation", "blog", "careers", "github"):
            return SourcePerspective.FIRST_PARTY
        if cat in ("news", "press_release"):
            return SourcePerspective.THIRD_PARTY_NEWS
        if cat in ("reddit", "twitter_x", "community"):
            return SourcePerspective.COMMUNITY_DISCUSSION
        if cat in ("review", "opinion", "benchmark"):
            return SourcePerspective.EXTERNAL_OPINION
        if cat in ("market_analysis", "analyst"):
            return SourcePerspective.MARKET_ANALYSIS
        return SourcePerspective.UNKNOWN

    def _extract_temporal_info(
        self, text: str
    ) -> tuple[datetime | None, TimePrecision, str | None]:
        """Extract explicit dates, coarse periods, or relative dates without inventing facts."""
        # 1. Explicit dates (e.g. August 15, 2026 or 2026-08-15)
        date_match = re.search(
            r"\b(?:on|date|published|released)?[:\s]*([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})\b",
            text,
            re.IGNORECASE,
        )
        if date_match:
            clean = date_match.group(1).strip().replace(",", "")
            for fmt in ("%B %d %Y", "%b %d %Y", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(clean, fmt)
                    return dt.replace(tzinfo=timezone.utc), TimePrecision.DATE_ONLY, None
                except ValueError:
                    pass

        # 2. Coarse periods (e.g. Q3 2026, H1 2025)
        quarter_match = re.search(r"\b(Q[1-4]\s+\d{4}|H[1-2]\s+\d{4})\b", text, re.IGNORECASE)
        if quarter_match:
            return None, TimePrecision.QUARTER_YEAR, quarter_match.group(1).strip()

        # 3. Relative expressions (e.g. yesterday, last quarter, last month)
        rel_match = re.search(r"\b(yesterday|last quarter|last month|last week)\b", text, re.IGNORECASE)
        if rel_match:
            return None, TimePrecision.APPROXIMATE, rel_match.group(1).strip().lower()

        return None, TimePrecision.UNKNOWN, None


# ==============================================================================
# LLM Evidence Interpreter (Isolated Structured Extraction with Fallback)
# ==============================================================================


class LLMExtractedEventItem(BaseModel):
    """Structured candidate event item emitted by an LLM."""

    event_type: IntelligenceEventType
    title: str
    summary: str
    subject: str
    previous_state: dict[str, Any] | str | None = None
    new_state: dict[str, Any] | str | None = None
    occurred_at_iso: str | None = None
    time_precision: TimePrecision = TimePrecision.UNKNOWN
    approximate_period: str | None = None
    confidence_score: float = Field(default=0.85, ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    confidence_rationale: str | None = None
    supporting_chunk_ids: list[str] = Field(min_length=1)


class LLMExtractionResponse(BaseModel):
    """Structured extraction response from LLM."""

    events: list[LLMExtractedEventItem] = Field(default_factory=list)


class LLMEvidenceInterpreter(BaseEvidenceInterpreter):
    """LLM-assisted evidence interpreter using structured Pydantic extraction.

    Invariants:
    - Strictly isolates LLM behind BaseLLMClient and Pydantic validation.
    - Every extracted event MUST cite valid retrieved chunk IDs from the provided evidence set.
    - Hallucinated or uncited candidate events are rejected with explicit rejection diagnostics.
    - On LLM failure (timeout, error, malformed output), falls back safely to fallback_interpreter.
    - Fallback is explicitly observable in extraction metadata/diagnostics ('fallback_used': True).
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        fallback_interpreter: BaseEvidenceInterpreter | None = None,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.fallback_interpreter = fallback_interpreter or RuleBasedEvidenceInterpreter()
        self.budget_tracker = budget_tracker
        self.last_diagnostics: dict[str, Any] = {
            "candidate_events_count": 0,
            "accepted_events_count": 0,
            "rejected_events_count": 0,
            "rejection_reasons": [],
            "fallback_used": False,
            "fallback_reason": None,
        }

    async def interpret_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
        context_metadata: dict[str, Any] | None = None,
    ) -> list[RawEventPayload]:
        if not evidence:
            self.last_diagnostics = {
                "candidate_events_count": 0,
                "accepted_events_count": 0,
                "rejected_events_count": 0,
                "rejection_reasons": ["Empty evidence set provided."],
                "fallback_used": False,
                "fallback_reason": None,
            }
            return []

        valid_chunk_map: dict[str, RetrievalResult] = {item.chunk_id: item for item in evidence}

        # Format prompt
        evidence_snippets_text = "\n\n".join(
            f"[Chunk ID: {r.chunk_id}]\nURL: {r.canonical_url or 'N/A'}\nCategory: {r.source_category or 'other'}\nText: {r.text}"
            for r in evidence
        )

        prompt = (
            f"You are a factual competitive intelligence extraction system.\n"
            f"Target Company: {company_id}\n"
            f"Research Query: {query}\n\n"
            f"Extract discrete, factual events observed in the following retrieved evidence chunks.\n"
            f"CRITICAL RULES:\n"
            f"1. Only extract events that are directly stated in the evidence text.\n"
            f"2. Every event MUST reference at least one valid Chunk ID from the provided chunks.\n"
            f"3. Do NOT make strategic conclusions or recommendations.\n"
            f"4. If no clear factual event occurred, return an empty events list.\n\n"
            f"Retrieved Evidence:\n{evidence_snippets_text}"
        )

        try:
            response = await self.llm_client.generate_structured(
                prompt=prompt,
                response_schema=LLMExtractionResponse,
                system_prompt="You are an evidence extraction assistant. Extract factual events with explicit chunk citations.",
                budget_tracker=self.budget_tracker,
            )

            raw_payloads: list[RawEventPayload] = []
            rejection_reasons: list[str] = []
            candidate_count = len(response.events)

            for item in response.events:
                # Validate supporting chunk IDs
                valid_snippets: list[RetrievedEvidenceSnippet] = []
                for cid in item.supporting_chunk_ids:
                    if cid in valid_chunk_map:
                        res = valid_chunk_map[cid]
                        persp = SourcePerspective.UNKNOWN
                        if isinstance(res.metadata.get("source_perspective"), SourcePerspective):
                            persp = res.metadata["source_perspective"]
                        elif isinstance(res.metadata.get("source_perspective"), str):
                            try:
                                persp = SourcePerspective(res.metadata["source_perspective"].lower())
                            except ValueError:
                                pass

                        valid_snippets.append(
                            RetrievedEvidenceSnippet(
                                chunk_id=res.chunk_id,
                                document_id=res.document_id,
                                version_id=res.version_id,
                                source_id=res.source_id or "unknown_source",
                                source_url=res.canonical_url or "https://example.com",
                                source_category=res.source_category or "other",
                                source_perspective=persp,
                                extracted_text=res.text,
                                retrieval_score=res.score,
                                metadata=dict(res.metadata),
                            )
                        )

                if not valid_snippets:
                    reason = f"Candidate event '{item.title}' cited no valid chunk IDs from evidence (cited: {item.supporting_chunk_ids})."
                    rejection_reasons.append(reason)
                    continue

                occurred_at: datetime | None = None
                if item.occurred_at_iso:
                    try:
                        occurred_at = datetime.fromisoformat(item.occurred_at_iso)
                        if occurred_at.tzinfo is None:
                            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
                    except ValueError:
                        occurred_at = None

                payload = RawEventPayload(
                    event_type=item.event_type,
                    title=item.title,
                    summary=item.summary,
                    subject=item.subject,
                    previous_state=item.previous_state,
                    new_state=item.new_state,
                    occurred_at=occurred_at,
                    time_precision=item.time_precision,
                    approximate_period=item.approximate_period,
                    confidence_score=item.confidence_score,
                    confidence_level=item.confidence_level,
                    confidence_rationale=item.confidence_rationale,
                    evidence_snippets=valid_snippets,
                    metadata={
                        "extraction_source": "llm",
                        "fallback_used": False,
                        "cited_chunk_ids": [s.chunk_id for s in valid_snippets],
                    },
                )
                raw_payloads.append(payload)

            self.last_diagnostics = {
                "candidate_events_count": candidate_count,
                "accepted_events_count": len(raw_payloads),
                "rejected_events_count": candidate_count - len(raw_payloads),
                "rejection_reasons": rejection_reasons,
                "fallback_used": False,
                "fallback_reason": None,
            }
            return raw_payloads

        except Exception as exc:
            # Fallback path on LLM failure
            fallback_payloads = await self.fallback_interpreter.interpret_evidence(
                query=query,
                company_id=company_id,
                workspace_id=workspace_id,
                evidence=evidence,
                context_metadata=context_metadata,
            )

            # Mark all fallback payloads with explicit observable fallback metadata
            for p in fallback_payloads:
                p.metadata["extraction_source"] = "rule_based_fallback"
                p.metadata["fallback_used"] = True
                p.metadata["fallback_reason"] = f"{type(exc).__name__}: {str(exc)}"
                p.confidence_rationale = f"[Fallback: {type(exc).__name__}] {p.confidence_rationale or ''}"

            self.last_diagnostics = {
                "candidate_events_count": len(fallback_payloads),
                "accepted_events_count": len(fallback_payloads),
                "rejected_events_count": 0,
                "rejection_reasons": [f"LLM extraction failed ({type(exc).__name__}: {str(exc)}); used rule-based fallback."],
                "fallback_used": True,
                "fallback_reason": f"{type(exc).__name__}: {str(exc)}",
            }
            return fallback_payloads


# ==============================================================================
# Retrieved Evidence Analyzer Component
# ==============================================================================


class RetrievedEvidenceAnalyzer(BaseIntelligenceAnalyzer):
    """Production component analyzing retrieved search evidence into grounded IntelligenceEvents.

    Performs deterministic clustering and evidence role aggregation:
    - Avoids duplicate events for overlapping chunks.
    - Preserves multi-source evidence references without premature credibility ranking.
    - Strictly preserves: No evidence -> no event.
    """

    def __init__(
        self,
        evidence_interpreter: BaseEvidenceInterpreter | None = None,
        signature_strategy: BaseEventSignatureStrategy | None = None,
    ) -> None:
        self.evidence_interpreter = evidence_interpreter or RuleBasedEvidenceInterpreter()
        self.signature_strategy = signature_strategy or DeterministicEventSignatureStrategy()
        self.last_diagnostics: dict[str, Any] = {
            "candidate_events_count": 0,
            "accepted_events_count": 0,
            "rejected_events_count": 0,
            "rejection_reasons": [],
            "fallback_used": False,
            "fallback_reason": None,
        }

    async def analyze_retrieved_evidence(
        self,
        query: str,
        company_id: str,
        workspace_id: str,
        evidence: list[RetrievalResult],
    ) -> list[IntelligenceEvent]:
        """Analyze retrieved evidence chunks and synthesize verified IntelligenceEvents."""
        if not evidence:
            self.last_diagnostics = {
                "candidate_events_count": 0,
                "accepted_events_count": 0,
                "rejected_events_count": 0,
                "rejection_reasons": ["Empty evidence list provided."],
                "fallback_used": False,
                "fallback_reason": None,
            }
            return []

        # 1. Interpret Evidence Chunks into Candidate RawEventPayloads
        candidate_payloads = await self.evidence_interpreter.interpret_evidence(
            query=query,
            company_id=company_id,
            workspace_id=workspace_id,
            evidence=evidence,
        )

        if not candidate_payloads:
            interp_diag = getattr(self.evidence_interpreter, "last_diagnostics", {})
            self.last_diagnostics = {
                "candidate_events_count": interp_diag.get("candidate_events_count", 0),
                "accepted_events_count": 0,
                "rejected_events_count": interp_diag.get("rejected_events_count", 0),
                "rejection_reasons": interp_diag.get("rejection_reasons", ["No candidate events extracted from evidence."]),
                "fallback_used": interp_diag.get("fallback_used", False),
                "fallback_reason": interp_diag.get("fallback_reason", None),
            }
            return []

        # 2. Cluster / Deduplicate by Deterministic Signature
        clusters: dict[str, list[RawEventPayload]] = defaultdict(list)
        interp_diag = getattr(self.evidence_interpreter, "last_diagnostics", {})
        rejection_reasons: list[str] = list(interp_diag.get("rejection_reasons", []))
        rejected_count = 0

        for payload in candidate_payloads:
            # Enforce Grounding Invariant: No Evidence -> No Event
            if not payload.evidence_snippets:
                rejected_count += 1
                rejection_reasons.append(f"Rejected ungrounded candidate event: '{payload.title}' (0 evidence snippets).")
                continue
            sig = self.signature_strategy.compute_signature(payload, company_id)
            clusters[sig].append(payload)

        # 3. Assemble Unified Intelligence Events
        intelligence_events: list[IntelligenceEvent] = []

        for sig, payload_list in clusters.items():
            primary_payload = payload_list[0]

            # Merge all evidence snippets across duplicate/overlapping payloads
            seen_snippets: set[tuple[str, str]] = set()
            all_evidence_refs: list[EvidenceReference] = []

            for p in payload_list:
                for snippet in p.evidence_snippets:
                    snip_key = (snippet.chunk_id, snippet.extracted_text)
                    if snip_key in seen_snippets:
                        continue
                    seen_snippets.add(snip_key)

                    merged_meta = dict(snippet.metadata)
                    if snippet.locator:
                        merged_meta["locator"] = snippet.locator

                    all_evidence_refs.append(
                        EvidenceReference(
                            source_id=snippet.source_id,
                            document_id=snippet.document_id,
                            version_id=snippet.version_id,
                            chunk_id=snippet.chunk_id,
                            source_url=snippet.source_url,
                            source_category=snippet.source_category,
                            source_perspective=snippet.source_perspective,
                            extracted_text=snippet.extracted_text,
                            acquired_at=snippet.acquired_at,
                            retrieval_score=snippet.retrieval_score,
                            role=snippet.role,
                            metadata=merged_meta,
                        )
                    )

            if not all_evidence_refs:
                rejected_count += 1
                rejection_reasons.append(f"Cluster '{sig}' yielded 0 merged evidence references.")
                continue

            # Determine primary perspective from the primary direct snippet
            primary_perspective = all_evidence_refs[0].source_perspective

            # Build EventTime (preferring explicit occurred_at if present in any clustered payload)
            best_occurred_at = next(
                (p.occurred_at for p in payload_list if p.occurred_at is not None), None
            )
            best_precision = (
                primary_payload.time_precision
                if best_occurred_at is None
                else TimePrecision.DATE_ONLY
            )
            best_approx = next(
                (p.approximate_period for p in payload_list if p.approximate_period), None
            )

            event_time = EventTime(
                occurred_at=best_occurred_at,
                precision=best_precision,
                approximate_period=best_approx,
                acquired_at=all_evidence_refs[0].acquired_at,
                detected_at=datetime.now(timezone.utc),
            )

            # Confidence Assessment (is_corroborated remains False until a dedicated source-independence layer evaluates it)
            confidence = ConfidenceAssessment(
                score=primary_payload.confidence_score,
                level=primary_payload.confidence_level,
                rationale=primary_payload.confidence_rationale,
                is_corroborated=False,
            )

            # Significance Assessment remains UNASSESSED
            significance = SignificanceAssessment(
                status=SignificanceStatus.UNASSESSED,
                level=SignificanceLevel.UNKNOWN,
            )

            best_new_state = next((p.new_state for p in payload_list if p.new_state is not None), None)
            best_prev_state = next((p.previous_state for p in payload_list if p.previous_state is not None), None)

            # Combined metadata preserving extraction provenance
            combined_meta = dict(primary_payload.metadata)
            combined_meta["cluster_signature"] = sig
            combined_meta["merged_payload_count"] = len(payload_list)

            event = IntelligenceEvent(
                workspace_id=workspace_id,
                company_id=company_id,
                event_type=primary_payload.event_type,
                title=primary_payload.title,
                summary=primary_payload.summary,
                subject=primary_payload.subject,
                previous_state=best_prev_state,
                new_state=best_new_state,
                event_time=event_time,
                confidence=confidence,
                significance=significance,
                evidence=all_evidence_refs,
                primary_perspective=primary_perspective,
                metadata=combined_meta,
            )
            intelligence_events.append(event)

        total_candidates = max(len(candidate_payloads), interp_diag.get("candidate_events_count", 0))
        total_rejected = rejected_count + interp_diag.get("rejected_events_count", 0)

        self.last_diagnostics = {
            "candidate_events_count": total_candidates,
            "accepted_events_count": len(intelligence_events),
            "rejected_events_count": total_rejected,
            "candidate_events": total_candidates,
            "accepted_events": len(intelligence_events),
            "rejected_events": total_rejected,
            "rejection_reasons": rejection_reasons,
            "fallback_used": interp_diag.get("fallback_used", False),
            "fallback_reason": interp_diag.get("fallback_reason", None),
        }


        return intelligence_events

    async def analyze_document_change(
        self,
        change: ChangeResult,
        current_version: DocumentVersion,
        previous_version: DocumentVersion | None,
        context_evidence: list[RetrievalResult] | None = None,
    ) -> list[IntelligenceEvent]:
        """Subsystem 5 Phase 1 capability (delegated to DocumentChangeAnalyzer)."""
        return []

