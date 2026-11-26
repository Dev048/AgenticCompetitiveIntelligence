"""Deterministic sanitation of LLM-proposed retrieval filters.

The query-analysis LLM may propose per-sub-query company and source-category filters.
It writes what it sees in the question ("Northwind", "Northwind Metrics", a domain), not
the stored competitor IDs, and may name categories that do not exist. Applied verbatim,
such filters silently match nothing and waste a retrieval hop. These helpers map proposed
values onto known identifiers and drop anything that cannot be resolved unambiguously.
"""

import re
from abc import ABC, abstractmethod

from pydantic import BaseModel

from competitive_intelligence.source_acquisition.models import SourceCategory

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MIN_PREFIX_CHARS = 4
_VALID_CATEGORIES = frozenset(c.value for c in SourceCategory)


class CompanyRef(BaseModel):
    """A competitor known to a workspace, as the resolver sees it."""

    company_id: str
    name: str
    domain: str | None = None


class BaseCompanyResolver(ABC):
    """Supplies the competitors of a workspace for filter resolution."""

    @abstractmethod
    async def known_companies(self, workspace_id: str) -> list[CompanyRef]:
        """Return the competitors registered in ``workspace_id``."""


def _norm(value: str) -> str:
    return _NON_ALNUM.sub("", value.lower())


def _aliases(company: CompanyRef) -> set[str]:
    aliases = {_norm(company.company_id), _norm(company.name)}
    if company.domain:
        domain = company.domain.lower().removeprefix("www.")
        aliases.add(_norm(domain))
        aliases.add(_norm(domain.split(".")[0]))
    aliases.discard("")
    return aliases


def resolve_company_refs(refs: list[str], known: list[CompanyRef]) -> list[str]:
    """Map LLM-written company references to stored competitor IDs.

    A reference resolves when, after lower-casing and stripping punctuation, it equals a
    competitor's ID, name or domain (with or without TLD), or is an unambiguous prefix
    (at least 4 characters) of exactly one competitor's name. Unresolved or ambiguous
    references are dropped. Order is preserved and duplicates removed.
    """
    resolved: list[str] = []
    for ref in refs:
        cleaned = ref.strip().lower().removeprefix("https://").removeprefix("http://")
        key = _norm(cleaned.removeprefix("www.").rstrip("/"))
        if not key:
            continue
        exact = [c.company_id for c in known if key in _aliases(c)]
        matches = exact or (
            [c.company_id for c in known if _norm(c.name).startswith(key)]
            if len(key) >= _MIN_PREFIX_CHARS
            else []
        )
        if len(set(matches)) == 1 and matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def sanitize_source_categories(categories: list[str]) -> list[str]:
    """Keep only categories that exist in the source taxonomy (case-insensitive)."""
    kept: list[str] = []
    for category in categories:
        value = category.strip().lower()
        if value in _VALID_CATEGORIES and value not in kept:
            kept.append(value)
    return kept
