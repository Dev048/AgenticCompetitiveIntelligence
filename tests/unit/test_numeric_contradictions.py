"""Deterministic cross-document numeric contradiction check, and its use in verification."""

from pydantic import BaseModel

from competitive_intelligence.agentic_rag.contradictions import (
    extract_numeric_claims,
    find_numeric_conflicts,
)
from competitive_intelligence.agentic_rag.llm.interfaces import BaseLLMClient, T
from competitive_intelligence.agentic_rag.models import SufficiencyDecision, VerificationAssessment
from competitive_intelligence.agentic_rag.nodes.verify_evidence_node import VerifyEvidenceNode
from competitive_intelligence.core.budgets import LLMBudgetTracker
from competitive_intelligence.retrieval_engine.models import RetrievalResult


def _chunk(chunk_id: str, document_id: str, text: str, company: str = "acme") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        document_id=document_id,
        version_id=f"{document_id}-v1",
        company_id=company,
    )


def test_extracts_scoped_quantities_and_skips_noise() -> None:
    chunk = _chunk(
        "c1",
        "d1",
        "Team plan: $99 per month. Includes 25 users and 1,000 projects.\n"
        "Release 7.2 ships 4 out of 5 dashboards; a 14-day trial; $40 million raised.",
    )
    claims = {(c.subject, c.unit, c.value) for c in extract_numeric_claims(chunk)}
    assert claims == {("team plan", "user", 25), ("team plan", "project", 1000)}


def test_numbers_without_a_subject_are_ignored() -> None:
    assert extract_numeric_claims(_chunk("c1", "d1", "We have 40 engineers and 3 offices.")) == []


def test_negated_numbers_are_not_claims() -> None:
    chunk = _chunk("c1", "d1", "On the Pro tier we got 2 workspaces, not the 5 workspaces advertised.")
    assert [c.value for c in extract_numeric_claims(chunk)] == [2]


def test_conflict_across_documents_for_same_entity_subject_unit() -> None:
    conflicts = find_numeric_conflicts(
        [
            _chunk("c1", "docs", "Pro tier: includes 10 workspaces."),
            _chunk("c2", "forum", "Upgraded to the Pro tier, which gives 6 workspaces in practice."),
        ]
    )
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert (conflict.entity, conflict.subject, conflict.unit) == ("acme", "pro tier", "workspace")
    assert conflict.values_by_document == {"docs": [10], "forum": [6]}
    assert conflict.chunk_ids == ["c1", "c2"]


def test_versions_of_one_document_are_a_change_not_a_conflict() -> None:
    assert find_numeric_conflicts(
        [
            _chunk("c1", "pricing", "Starter plan: includes 2 seats."),
            _chunk("c2", "pricing", "Starter plan: now includes 5 seats."),
        ]
    ) == []


def test_agreeing_documents_and_different_scopes_do_not_conflict() -> None:
    results = [
        _chunk("c1", "docs", "Pro tier: includes 10 workspaces. Basic tier: includes 2 workspaces."),
        _chunk("c2", "blog", "The Pro tier gives 10 workspaces."),
        _chunk("c3", "other", "Pro tier: includes 3 workspaces.", company="globex"),
    ]
    assert find_numeric_conflicts(results) == []


class _SilentVerifier(BaseLLMClient):
    """An LLM verifier that notices nothing (the failure mode the check guards against)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> T:
        self.prompts.append(prompt)
        result: BaseModel = VerificationAssessment(
            decision=SufficiencyDecision.SUFFICIENT,
            is_sufficient=True,
            reasoning="Evidence answers the question.",
        )
        return result  # type: ignore[return-value]

    async def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        budget_tracker: LLMBudgetTracker | None = None,
    ) -> str:
        return ""


async def test_verifier_receives_and_keeps_deterministic_conflicts() -> None:
    llm = _SilentVerifier()
    node = VerifyEvidenceNode(llm_client=llm)
    state = {
        "original_query": "How many workspaces does the Pro tier include?",
        "hop_count": 1,
        "accumulated_results": [
            _chunk("c1", "docs", "Pro tier: includes 10 workspaces."),
            _chunk("c2", "forum", "Upgraded to the Pro tier, which gives 6 workspaces in practice."),
        ],
    }

    update = await node(state)  # type: ignore[arg-type]

    assert "Deterministic consistency check" in llm.prompts[0]
    contradictions = update["latest_verification"].detected_contradictions
    assert len(contradictions) == 1
    assert contradictions[0].conflicting_chunk_ids == ["c1", "c2"]
    assert contradictions[0].rationale.startswith("Detected by the deterministic")


async def test_verifier_prompt_unchanged_when_no_conflicts() -> None:
    llm = _SilentVerifier()
    node = VerifyEvidenceNode(llm_client=llm)
    await node(
        {
            "original_query": "q",
            "hop_count": 1,
            "accumulated_results": [_chunk("c1", "docs", "Pro tier: includes 10 workspaces.")],
        }  # type: ignore[arg-type]
    )
    assert "Deterministic consistency check" not in llm.prompts[0]
