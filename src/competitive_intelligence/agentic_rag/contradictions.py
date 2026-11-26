"""Deterministic cross-document check for conflicting numeric claims.

Finds quantity claims of the form "<number> <unit>" (e.g. "3 seats", "10,000 tracked
events") and scopes each to an entity (the chunk's company) and a subject (the nearest
preceding "<Name> plan|tier|edition" mention in the same chunk). Two documents conflict
when they state different values for the same (entity, subject, unit).

Deliberate limits, to stay precise rather than clever:
- Only claims with an explicit subject are considered; free-floating numbers are skipped.
- Chunks from the same document are never compared: differing values across versions of a
  document are a change over time, which change detection handles, not a contradiction.
- A number directly negated ("not the 3 seats") or used as a ratio ("4 out of 5") is not
  counted as a claim.
- Currency amounts are not compared (prices are routinely quoted with "previously" or
  "from ... to ..." wording that this pattern cannot disambiguate).

The output only flags differing observations with their chunk IDs; it never decides which
source is right. Interpretation is left to the LLM verifier.
"""

import re
from collections import defaultdict

from pydantic import BaseModel

from competitive_intelligence.retrieval_engine.models import RetrievalResult

_SUBJECT = re.compile(r"\b([A-Z][A-Za-z0-9-]*)\s+(plan|tier|edition)\b", re.IGNORECASE)
_QUANTITY = re.compile(r"(?<![$\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?![\w.,]*\d)\s+([A-Za-z]+)")
_NEGATION_BEFORE = re.compile(r"\b(?:not|no|never)\b(?:\s+\w+)?\s*$", re.IGNORECASE)
_RATIO_BEFORE = re.compile(r"\bout\s+of\s*$", re.IGNORECASE)  # "4 out of 5 dashboards"
_MAX_SUBJECT_DISTANCE = 250
_UNIT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "of", "or", "out", "per", "the", "to", "for", "in", "on", "at",
        "million", "billion", "thousand", "hundred", "percent", "times", "x",
    }
)
_SUBJECT_STOPWORDS = frozenset({"the", "a", "an", "this", "that", "our", "their", "new", "each", "every", "any", "paid", "all"})


class NumericClaim(BaseModel):
    entity: str
    subject: str
    unit: str
    value: int
    chunk_id: str
    document_id: str
    excerpt: str


class NumericConflict(BaseModel):
    entity: str
    subject: str
    unit: str
    values_by_document: dict[str, list[int]]
    chunk_ids: list[str]
    excerpts: list[str]

    def describe(self) -> str:
        observed = " vs ".join(
            "/".join(str(v) for v in values) for values in self.values_by_document.values()
        )
        return (
            f"Sources disagree on the {self.unit} count for the {self.subject} "
            f"({self.entity}): {observed}."
        )


def _singular(word: str) -> str:
    word = word.lower()
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def extract_numeric_claims(result: RetrievalResult) -> list[NumericClaim]:
    """Numeric quantity claims in one chunk, scoped to entity and subject."""
    if not result.company_id:
        return []
    text = result.text
    subjects = [
        (m.start(), f"{m.group(1).lower()} {m.group(2).lower()}")
        for m in _SUBJECT.finditer(text)
        if m.group(1).lower() not in _SUBJECT_STOPWORDS
    ]
    claims: list[NumericClaim] = []
    for match in _QUANTITY.finditer(text):
        unit = _singular(match.group(2))
        if unit in _UNIT_STOPWORDS:
            continue
        start = match.start()
        before = text[max(0, start - 20) : start]
        if _NEGATION_BEFORE.search(before) or _RATIO_BEFORE.search(before):
            continue
        preceding = [s for s in subjects if s[0] < start and start - s[0] <= _MAX_SUBJECT_DISTANCE]
        if not preceding:
            continue
        subject = preceding[-1][1]
        claims.append(
            NumericClaim(
                entity=result.company_id,
                subject=subject,
                unit=unit,
                value=int(match.group(1).replace(",", "")),
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                excerpt=" ".join(text[max(0, start - 60) : match.end() + 20].split()),
            )
        )
    return claims


def find_numeric_conflicts(results: list[RetrievalResult]) -> list[NumericConflict]:
    """Conflicts between documents stating different values for the same claim key."""
    by_key: dict[tuple[str, str, str], list[NumericClaim]] = defaultdict(list)
    for result in results:
        for claim in extract_numeric_claims(result):
            by_key[(claim.entity, claim.subject, claim.unit)].append(claim)

    conflicts: list[NumericConflict] = []
    for (entity, subject, unit), claims in sorted(by_key.items()):
        values_by_doc: dict[str, set[int]] = defaultdict(set)
        for claim in claims:
            values_by_doc[claim.document_id].add(claim.value)
        if len(values_by_doc) < 2:
            continue
        distinct_sets = {frozenset(v) for v in values_by_doc.values()}
        if len(distinct_sets) < 2:
            continue
        conflicts.append(
            NumericConflict(
                entity=entity,
                subject=subject,
                unit=unit,
                values_by_document={doc: sorted(vals) for doc, vals in sorted(values_by_doc.items())},
                chunk_ids=list(dict.fromkeys(c.chunk_id for c in claims)),
                excerpts=list(dict.fromkeys(c.excerpt for c in claims)),
            )
        )
    return conflicts
