from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from competitive_intelligence.application.interfaces import BaseIntelligenceRepository
from competitive_intelligence.infrastructure.database.models import (
    EventEvidenceLinkOrm,
    EvidenceReferenceOrm,
    InferredSignalOrm,
    InsightEventLinkOrm,
    InsightOrm,
    InsightSignalLinkOrm,
    IntelligenceEventOrm,
    SignalEventLinkOrm,
)
from competitive_intelligence.infrastructure.database.session import (
    DatabaseSessionManager,
    get_db_manager,
)
from competitive_intelligence.intelligence.models import (
    ConfidenceAssessment,
    ConfidenceLevel,
    EventTime,
    EvidenceReference,
    EvidenceRole,
    InferredSignal,
    Insight,
    InsightStatus,
    InsightType,
    IntelligenceEvent,
    IntelligenceEventType,
    InterpretiveConfidenceAssessment,
    SignalConfidenceAssessment,
    SignalStatus,
    SignalType,
    SignificanceAssessment,
    SignificanceLevel,
    SignificanceStatus,
    SourcePerspective,
    TemporalScope,
    TimePrecision,
)


def _ensure_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _evidence_to_orm(ev: EvidenceReference) -> EvidenceReferenceOrm:
    return EvidenceReferenceOrm(
        evidence_id=ev.evidence_id,
        source_id=ev.source_id,
        document_id=ev.document_id,
        version_id=ev.version_id,
        chunk_id=ev.chunk_id,
        source_url=ev.source_url,
        source_category=ev.source_category,
        source_perspective=ev.source_perspective.value,
        evidence_role=ev.role.value,
        extracted_text=ev.extracted_text,
        retrieval_score=ev.retrieval_score,
        acquired_at=ev.acquired_at,
        metadata_=ev.metadata,
    )


def _orm_to_evidence(orm: EvidenceReferenceOrm) -> EvidenceReference:
    acq_at = _ensure_utc(orm.acquired_at)
    assert acq_at is not None
    return EvidenceReference(
        evidence_id=orm.evidence_id,
        source_id=orm.source_id,
        document_id=orm.document_id,
        version_id=orm.version_id,
        chunk_id=orm.chunk_id,
        source_url=orm.source_url,
        source_category=orm.source_category,
        source_perspective=SourcePerspective(orm.source_perspective),
        extracted_text=orm.extracted_text,
        acquired_at=acq_at,
        retrieval_score=orm.retrieval_score,
        role=EvidenceRole(orm.evidence_role),
        metadata=orm.metadata_,
    )


def _parse_event_time(data: dict[str, Any]) -> EventTime:
    occurred_str = data.get("occurred_at")
    occurred_at = (
        datetime.fromisoformat(occurred_str)
        if isinstance(occurred_str, str)
        else occurred_str
    )
    acquired_str = data.get("acquired_at")
    acquired_at = (
        datetime.fromisoformat(acquired_str)
        if isinstance(acquired_str, str)
        else (acquired_str or datetime.now(timezone.utc))
    )
    detected_str = data.get("detected_at")
    detected_at = (
        datetime.fromisoformat(detected_str)
        if isinstance(detected_str, str)
        else (detected_str or datetime.now(timezone.utc))
    )
    acq_utc = _ensure_utc(acquired_at)
    assert acq_utc is not None
    det_utc = _ensure_utc(detected_at)
    assert det_utc is not None
    return EventTime(
        occurred_at=_ensure_utc(occurred_at),
        precision=TimePrecision(data.get("precision", "unknown")),
        approximate_period=data.get("approximate_period"),
        acquired_at=acq_utc,
        detected_at=det_utc,
    )


def _parse_confidence(data: dict[str, Any]) -> ConfidenceAssessment:
    return ConfidenceAssessment(
        score=data.get("score", 1.0),
        level=ConfidenceLevel(data.get("level", "high")),
        rationale=data.get("rationale"),
        is_corroborated=data.get("is_corroborated", False),
    )


def _parse_significance(data: dict[str, Any]) -> SignificanceAssessment:
    eval_str = data.get("evaluated_at")
    evaluated_at = (
        datetime.fromisoformat(eval_str) if isinstance(eval_str, str) else eval_str
    )
    return SignificanceAssessment(
        status=SignificanceStatus(data.get("status", "unassessed")),
        level=SignificanceLevel(data.get("level", "unknown")),
        score=data.get("score"),
        rationale=data.get("rationale"),
        evaluated_at=_ensure_utc(evaluated_at),
    )


class SqlIntelligenceRepository(BaseIntelligenceRepository):
    """Repository managing grounded Intelligence events, signals, insights, and evidence lineage."""

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
            raise RuntimeError("SqlIntelligenceRepository has no active session or sessionmaker")

    # --------------------------------------------------------------------------
    # EvidenceReference
    # --------------------------------------------------------------------------

    async def create_evidence(self, evidence: EvidenceReference) -> EvidenceReference:
        """Persist a single EvidenceReference."""
        existing = await self.get_evidence_by_id(evidence.evidence_id)
        if existing is not None:
            return existing

        async with self._get_session() as session:
            orm_ev = _evidence_to_orm(evidence)
            session.add(orm_ev)
            await session.flush()
        return evidence

    async def get_evidence_by_id(self, evidence_id: str) -> EvidenceReference | None:
        """Fetch an EvidenceReference by primary evidence_id."""
        async with self._get_session() as session:
            stmt = select(EvidenceReferenceOrm).where(EvidenceReferenceOrm.evidence_id == evidence_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _orm_to_evidence(row)

    # --------------------------------------------------------------------------
    # IntelligenceEvent
    # --------------------------------------------------------------------------

    async def create_event(self, event: IntelligenceEvent) -> IntelligenceEvent:
        """Persist an IntelligenceEvent and link its supporting EvidenceReferences."""
        existing = await self.get_event_by_id(event.event_id)
        if existing is not None:
            return existing

        # Ensure evidence references exist
        for ev in event.evidence:
            existing_ev = await self.get_evidence_by_id(ev.evidence_id)
            if existing_ev is None:
                await self.create_evidence(ev)

        event_time_dict = {
            "occurred_at": event.event_time.occurred_at.isoformat()
            if event.event_time.occurred_at
            else None,
            "precision": event.event_time.precision.value,
            "approximate_period": event.event_time.approximate_period,
            "acquired_at": event.event_time.acquired_at.isoformat(),
            "detected_at": event.event_time.detected_at.isoformat(),
        }

        confidence_dict = {
            "score": event.confidence.score,
            "level": event.confidence.level.value,
            "rationale": event.confidence.rationale,
            "is_corroborated": event.confidence.is_corroborated,
        }

        significance_dict = {
            "status": event.significance.status.value,
            "level": event.significance.level.value,
            "score": event.significance.score,
            "rationale": event.significance.rationale,
            "evaluated_at": event.significance.evaluated_at.isoformat()
            if event.significance.evaluated_at
            else None,
        }

        orm_event = IntelligenceEventOrm(
            event_id=event.event_id,
            workspace_id=event.workspace_id,
            company_id=event.company_id,
            event_type=event.event_type.value,
            title=event.title,
            summary=event.summary,
            subject=event.subject,
            previous_state=event.previous_state if isinstance(event.previous_state, dict) else ({"state": event.previous_state} if event.previous_state else None),
            new_state=event.new_state if isinstance(event.new_state, dict) else ({"state": event.new_state} if event.new_state else None),
            event_time=event_time_dict,
            detected_at=event.detected_at,
            confidence=confidence_dict,
            significance=significance_dict,
            primary_perspective=event.primary_perspective.value,
            sequence_id=event.sequence_id,
            preceding_event_id=event.preceding_event_id,
            metadata_=event.metadata,
        )

        async with self._get_session() as session:
            session.add(orm_event)
            await session.flush()

            # Link event -> evidence M:N
            for ev in event.evidence:
                link = EventEvidenceLinkOrm(event_id=event.event_id, evidence_id=ev.evidence_id)
                session.add(link)
            await session.flush()

        return event

    async def get_event_by_id(self, event_id: str) -> IntelligenceEvent | None:
        """Fetch an IntelligenceEvent with all joined EvidenceReferences."""
        async with self._get_session() as session:
            stmt = select(IntelligenceEventOrm).where(IntelligenceEventOrm.event_id == event_id)
            result = await session.execute(stmt)
            orm_event = result.scalar_one_or_none()
            if orm_event is None:
                return None

            # Fetch linked evidence
            ev_stmt = (
                select(EvidenceReferenceOrm)
                .join(EventEvidenceLinkOrm, EvidenceReferenceOrm.evidence_id == EventEvidenceLinkOrm.evidence_id)
                .where(EventEvidenceLinkOrm.event_id == event_id)
            )
            ev_result = await session.execute(ev_stmt)
            ev_rows = ev_result.scalars().all()
            evidence_list = [_orm_to_evidence(r) for r in ev_rows]

            return self._orm_to_event(orm_event, evidence_list)

    async def list_evidence_for_document(self, document_id: str) -> list[EvidenceReference]:
        """List all evidence references derived from a document."""
        async with self._get_session() as session:
            stmt = (
                select(EvidenceReferenceOrm)
                .where(EvidenceReferenceOrm.document_id == document_id)
                .order_by(EvidenceReferenceOrm.acquired_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [_orm_to_evidence(r) for r in rows]

    async def list_events(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[IntelligenceEvent]:
        """List intelligence events in a workspace with optional competitor filter."""
        async with self._get_session() as session:
            stmt = select(IntelligenceEventOrm).where(IntelligenceEventOrm.workspace_id == workspace_id)
            if company_id:
                stmt = stmt.where(IntelligenceEventOrm.company_id == company_id)
            stmt = stmt.order_by(IntelligenceEventOrm.detected_at.desc()).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()

            events: list[IntelligenceEvent] = []
            for r in rows:
                ev_stmt = (
                    select(EvidenceReferenceOrm)
                    .join(EventEvidenceLinkOrm, EvidenceReferenceOrm.evidence_id == EventEvidenceLinkOrm.evidence_id)
                    .where(EventEvidenceLinkOrm.event_id == r.event_id)
                )
                ev_res = await session.execute(ev_stmt)
                ev_rows = ev_res.scalars().all()
                events.append(self._orm_to_event(r, [_orm_to_evidence(e) for e in ev_rows]))
            return events

    async def list_events_by_company(self, company_id: str) -> list[IntelligenceEvent]:
        """List all events detected for a company."""
        async with self._get_session() as session:
            stmt = (
                select(IntelligenceEventOrm)
                .where(IntelligenceEventOrm.company_id == company_id)
                .order_by(IntelligenceEventOrm.detected_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

            events: list[IntelligenceEvent] = []
            for r in rows:
                ev_stmt = (
                    select(EvidenceReferenceOrm)
                    .join(EventEvidenceLinkOrm, EvidenceReferenceOrm.evidence_id == EventEvidenceLinkOrm.evidence_id)
                    .where(EventEvidenceLinkOrm.event_id == r.event_id)
                )
                ev_res = await session.execute(ev_stmt)
                ev_rows = ev_res.scalars().all()
                events.append(self._orm_to_event(r, [_orm_to_evidence(e) for e in ev_rows]))
            return events

    # --------------------------------------------------------------------------
    # InferredSignal
    # --------------------------------------------------------------------------

    async def create_signal(self, signal: InferredSignal) -> InferredSignal:
        """Persist an InferredSignal and its event links."""
        existing = await self.get_signal_by_id(signal.signal_id)
        if existing is not None:
            return existing

        confidence_dict = {
            "score": signal.confidence.score,
            "level": signal.confidence.level.value,
            "rationale": signal.confidence.rationale,
            "supporting_event_count": signal.confidence.supporting_event_count,
        }

        orm_sig = InferredSignalOrm(
            signal_id=signal.signal_id,
            workspace_id=signal.workspace_id,
            company_id=signal.company_id,
            signal_type=signal.signal_type.value,
            subject=signal.subject,
            title=signal.title,
            summary=signal.summary,
            first_observed_at=signal.first_observed_at,
            last_observed_at=signal.last_observed_at,
            inferred_at=signal.inferred_at,
            confidence=confidence_dict,
            status=signal.status.value,
            sequence_id=signal.sequence_id,
            metadata_=signal.metadata,
        )

        async with self._get_session() as session:
            session.add(orm_sig)
            await session.flush()

            for ev_id in signal.supporting_event_ids:
                link = SignalEventLinkOrm(signal_id=signal.signal_id, event_id=ev_id)
                session.add(link)
            await session.flush()
        return signal

    async def get_signal_by_id(self, signal_id: str) -> InferredSignal | None:
        """Fetch an InferredSignal by primary signal_id with linked event IDs."""
        async with self._get_session() as session:
            stmt = select(InferredSignalOrm).where(InferredSignalOrm.signal_id == signal_id)
            result = await session.execute(stmt)
            orm_sig = result.scalar_one_or_none()
            if orm_sig is None:
                return None

            # Fetch linked event IDs
            links_stmt = select(SignalEventLinkOrm.event_id).where(
                SignalEventLinkOrm.signal_id == signal_id
            )
            links_res = await session.execute(links_stmt)
            event_ids = list(links_res.scalars().all())

            return InferredSignal(
                signal_id=orm_sig.signal_id,
                workspace_id=orm_sig.workspace_id,
                company_id=orm_sig.company_id,
                company_ids=[orm_sig.company_id],
                signal_type=SignalType(orm_sig.signal_type),
                subject=orm_sig.subject,
                title=orm_sig.title,
                summary=orm_sig.summary,
                supporting_event_ids=event_ids if event_ids else ["unknown_event"],
                first_observed_at=_ensure_utc(orm_sig.first_observed_at),
                last_observed_at=_ensure_utc(orm_sig.last_observed_at),
                inferred_at=_ensure_utc(orm_sig.inferred_at) or datetime.now(timezone.utc),
                confidence=SignalConfidenceAssessment(
                    score=orm_sig.confidence.get("score", 0.8),
                    level=ConfidenceLevel(orm_sig.confidence.get("level", "high")),
                    rationale=orm_sig.confidence.get("rationale"),
                    supporting_event_count=orm_sig.confidence.get("supporting_event_count", 1),
                ),
                status=SignalStatus(orm_sig.status),
                sequence_id=orm_sig.sequence_id,
                metadata=orm_sig.metadata_,
            )

    async def list_signals(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[InferredSignal]:
        """List inferred signals in a workspace with optional competitor filter."""
        async with self._get_session() as session:
            stmt = select(InferredSignalOrm).where(InferredSignalOrm.workspace_id == workspace_id)
            if company_id:
                stmt = stmt.where(InferredSignalOrm.company_id == company_id)
            stmt = stmt.order_by(InferredSignalOrm.inferred_at.desc()).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()

            signals: list[InferredSignal] = []
            for r in rows:
                links_stmt = select(SignalEventLinkOrm.event_id).where(
                    SignalEventLinkOrm.signal_id == r.signal_id
                )
                links_res = await session.execute(links_stmt)
                event_ids = list(links_res.scalars().all())

                signals.append(
                    InferredSignal(
                        signal_id=r.signal_id,
                        workspace_id=r.workspace_id,
                        company_id=r.company_id,
                        company_ids=[r.company_id],
                        signal_type=SignalType(r.signal_type),
                        subject=r.subject,
                        title=r.title,
                        summary=r.summary,
                        supporting_event_ids=event_ids if event_ids else ["unknown_event"],
                        first_observed_at=_ensure_utc(r.first_observed_at),
                        last_observed_at=_ensure_utc(r.last_observed_at),
                        inferred_at=_ensure_utc(r.inferred_at) or datetime.now(timezone.utc),
                        confidence=SignalConfidenceAssessment(
                            score=r.confidence.get("score", 0.8),
                            level=ConfidenceLevel(r.confidence.get("level", "high")),
                            rationale=r.confidence.get("rationale"),
                            supporting_event_count=r.confidence.get("supporting_event_count", 1),
                        ),
                        status=SignalStatus(r.status),
                        sequence_id=r.sequence_id,
                        metadata=r.metadata_,
                    )
                )
            return signals

    # --------------------------------------------------------------------------
    # Insight
    # --------------------------------------------------------------------------

    async def create_insight(self, insight: Insight) -> Insight:
        """Persist a strategic Insight and its M:N signal and event links."""
        existing = await self.get_insight_by_id(insight.insight_id)
        if existing is not None:
            return existing

        factual_conf_dict = {
            "score": insight.factual_confidence.score,
            "level": insight.factual_confidence.level.value,
            "rationale": insight.factual_confidence.rationale,
            "is_corroborated": insight.factual_confidence.is_corroborated,
        }

        interp_conf_dict = {
            "score": insight.interpretive_confidence.score,
            "level": insight.interpretive_confidence.level.value,
            "rationale": insight.interpretive_confidence.rationale,
            "factual_basis_summary": insight.interpretive_confidence.factual_basis_summary,
        }

        significance_dict = {
            "status": insight.significance.status.value,
            "level": insight.significance.level.value,
            "score": insight.significance.score,
            "rationale": insight.significance.rationale,
            "evaluated_at": insight.significance.evaluated_at.isoformat()
            if insight.significance.evaluated_at
            else None,
        }

        temporal_dict = {
            "start_date": insight.temporal_scope.start_date.isoformat()
            if insight.temporal_scope.start_date
            else None,
            "end_date": insight.temporal_scope.end_date.isoformat()
            if insight.temporal_scope.end_date
            else None,
            "period_description": insight.temporal_scope.period_description,
        }

        orm_ins = InsightOrm(
            insight_id=insight.insight_id,
            workspace_id=insight.workspace_id,
            company_id=insight.company_id,
            insight_type=insight.insight_type.value,
            title=insight.title,
            summary=insight.summary,
            statement=insight.statement,
            factual_confidence=factual_conf_dict,
            interpretive_confidence=interp_conf_dict,
            significance=significance_dict,
            temporal_scope=temporal_dict,
            generated_at=insight.generated_at,
            status=insight.status.value,
            supersedes_insight_id=insight.supersedes_insight_id,
            metadata_=insight.metadata,
        )

        async with self._get_session() as session:
            session.add(orm_ins)
            await session.flush()

            # Link signal M:N
            for sig_id in insight.supporting_signal_ids:
                sig_link = InsightSignalLinkOrm(insight_id=insight.insight_id, signal_id=sig_id)
                session.add(sig_link)

            # Link event M:N
            for ev_id in insight.supporting_event_ids:
                ev_link = InsightEventLinkOrm(insight_id=insight.insight_id, event_id=ev_id)
                session.add(ev_link)

            await session.flush()
        return insight

    async def list_insights(
        self,
        workspace_id: str,
        company_id: str | None = None,
        limit: int = 100,
    ) -> list[Insight]:
        """List strategic insights in a workspace with optional competitor filter."""
        async with self._get_session() as session:
            stmt = select(InsightOrm).where(InsightOrm.workspace_id == workspace_id)
            if company_id:
                stmt = stmt.where(InsightOrm.company_id == company_id)
            stmt = stmt.order_by(InsightOrm.generated_at.desc()).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()

            insights: list[Insight] = []
            for r in rows:
                sig_stmt = select(InsightSignalLinkOrm.signal_id).where(
                    InsightSignalLinkOrm.insight_id == r.insight_id
                )
                sig_res = await session.execute(sig_stmt)
                signal_ids = list(sig_res.scalars().all())

                ev_stmt = select(InsightEventLinkOrm.event_id).where(
                    InsightEventLinkOrm.insight_id == r.insight_id
                )
                ev_res = await session.execute(ev_stmt)
                event_ids = list(ev_res.scalars().all())

                t_data = r.temporal_scope
                start_date = (
                    datetime.fromisoformat(t_data["start_date"])
                    if t_data.get("start_date")
                    else None
                )
                end_date = (
                    datetime.fromisoformat(t_data["end_date"])
                    if t_data.get("end_date")
                    else None
                )
                gen_at = _ensure_utc(r.generated_at)
                assert gen_at is not None

                insights.append(
                    Insight(
                        insight_id=r.insight_id,
                        workspace_id=r.workspace_id,
                        company_id=r.company_id,
                        insight_type=InsightType(r.insight_type),
                        title=r.title,
                        summary=r.summary,
                        statement=r.statement,
                        supporting_signal_ids=signal_ids,
                        supporting_event_ids=event_ids if event_ids else ["unknown_event"],
                        factual_confidence=_parse_confidence(r.factual_confidence),
                        interpretive_confidence=InterpretiveConfidenceAssessment(
                            score=r.interpretive_confidence.get("score", 0.8),
                            level=ConfidenceLevel(r.interpretive_confidence.get("level", "high")),
                            rationale=r.interpretive_confidence.get("rationale"),
                            factual_basis_summary=r.interpretive_confidence.get("factual_basis_summary"),
                        ),
                        significance=_parse_significance(r.significance),
                        temporal_scope=TemporalScope(
                            start_date=_ensure_utc(start_date),
                            end_date=_ensure_utc(end_date),
                            period_description=t_data.get("period_description"),
                        ),
                        generated_at=gen_at,
                        status=InsightStatus(r.status),
                        supersedes_insight_id=r.supersedes_insight_id,
                        metadata=r.metadata_,
                    )
                )
            return insights

    async def get_insight_by_id(self, insight_id: str) -> Insight | None:
        """Fetch an Insight with joined signal and event associations."""
        async with self._get_session() as session:
            stmt = select(InsightOrm).where(InsightOrm.insight_id == insight_id)
            result = await session.execute(stmt)
            orm_ins = result.scalar_one_or_none()
            if orm_ins is None:
                return None

            # Fetch signal links
            sig_stmt = select(InsightSignalLinkOrm.signal_id).where(
                InsightSignalLinkOrm.insight_id == insight_id
            )
            sig_res = await session.execute(sig_stmt)
            signal_ids = list(sig_res.scalars().all())

            # Fetch event links
            ev_stmt = select(InsightEventLinkOrm.event_id).where(
                InsightEventLinkOrm.insight_id == insight_id
            )
            ev_res = await session.execute(ev_stmt)
            event_ids = list(ev_res.scalars().all())

            t_data = orm_ins.temporal_scope
            start_date = (
                datetime.fromisoformat(t_data["start_date"])
                if t_data.get("start_date")
                else None
            )
            end_date = (
                datetime.fromisoformat(t_data["end_date"])
                if t_data.get("end_date")
                else None
            )

            gen_at = _ensure_utc(orm_ins.generated_at)
            assert gen_at is not None

            return Insight(
                insight_id=orm_ins.insight_id,
                workspace_id=orm_ins.workspace_id,
                company_id=orm_ins.company_id,
                insight_type=InsightType(orm_ins.insight_type),
                title=orm_ins.title,
                summary=orm_ins.summary,
                statement=orm_ins.statement,
                supporting_signal_ids=signal_ids,
                supporting_event_ids=event_ids if event_ids else ["unknown_event"],
                factual_confidence=_parse_confidence(orm_ins.factual_confidence),
                interpretive_confidence=InterpretiveConfidenceAssessment(
                    score=orm_ins.interpretive_confidence.get("score", 0.8),
                    level=ConfidenceLevel(orm_ins.interpretive_confidence.get("level", "high")),
                    rationale=orm_ins.interpretive_confidence.get("rationale"),
                    factual_basis_summary=orm_ins.interpretive_confidence.get("factual_basis_summary"),
                ),
                significance=_parse_significance(orm_ins.significance),
                temporal_scope=TemporalScope(
                    start_date=_ensure_utc(start_date),
                    end_date=_ensure_utc(end_date),
                    period_description=t_data.get("period_description"),
                ),
                generated_at=gen_at,
                status=InsightStatus(orm_ins.status),
                supersedes_insight_id=orm_ins.supersedes_insight_id,
                metadata=orm_ins.metadata_,
            )

    @staticmethod
    def _orm_to_event(
        orm_event: IntelligenceEventOrm, evidence_list: list[EvidenceReference]
    ) -> IntelligenceEvent:
        prev_state: dict[str, Any] | str | None = None
        if isinstance(orm_event.previous_state, dict):
            prev_state = orm_event.previous_state.get("state", orm_event.previous_state)
        elif isinstance(orm_event.previous_state, str):
            prev_state = orm_event.previous_state

        new_state: dict[str, Any] | str | None = None
        if isinstance(orm_event.new_state, dict):
            new_state = orm_event.new_state.get("state", orm_event.new_state)
        elif isinstance(orm_event.new_state, str):
            new_state = orm_event.new_state

        det_at = _ensure_utc(orm_event.detected_at)
        assert det_at is not None

        return IntelligenceEvent(
            event_id=orm_event.event_id,
            workspace_id=orm_event.workspace_id,
            company_id=orm_event.company_id,
            event_type=IntelligenceEventType(orm_event.event_type),
            title=orm_event.title,
            summary=orm_event.summary,
            subject=orm_event.subject,
            previous_state=prev_state,
            new_state=new_state,
            event_time=_parse_event_time(orm_event.event_time),
            detected_at=det_at,
            confidence=_parse_confidence(orm_event.confidence),
            significance=_parse_significance(orm_event.significance),
            evidence=evidence_list,
            primary_perspective=SourcePerspective(orm_event.primary_perspective),
            sequence_id=orm_event.sequence_id,
            preceding_event_id=orm_event.preceding_event_id,
            metadata=orm_event.metadata_,
        )
