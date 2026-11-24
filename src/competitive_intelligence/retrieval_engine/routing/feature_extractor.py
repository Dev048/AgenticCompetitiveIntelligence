"""Deterministic query feature extraction for retrieval routing.

Extracts lexical, structural, numeric, entity, and conceptual signals
from raw queries using pure deterministic regexes and rule sets.
"""

from collections.abc import Sequence
import re

from competitive_intelligence.retrieval_engine.routing.models import QueryFeatures

# Pre-compiled regex patterns for deterministic extraction
VERSION_REGEX = re.compile(
    r"\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?\b",
    re.IGNORECASE,
)
IDENTIFIER_REGEX = re.compile(
    r"\b[A-Z0-9]{2,}(?:[-_][A-Z0-9]+)+\b",
)
MODEL_NAME_REGEX = re.compile(
    r"\b(?:GPT|Claude|Llama|Gemini|BGE|Mistral|BERT)[-_A-Za-z0-9.]*\b",
    re.IGNORECASE,
)
QUOTE_REGEX = re.compile(r'"([^"]+)"')
CURRENCY_REGEX = re.compile(r"(?:[\$€£¥])\s*\d+(?:\.\d+)?(?:[kmbKMB])?\b")
PERCENTAGE_REGEX = re.compile(r"\b\d+(?:\.\d+)?%(?!\w)")
FINANCIAL_METRIC_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|ARR|MRR|users|tokens|seats|nodes)\b",
    re.IGNORECASE,
)
TEMPORAL_REGEX = re.compile(
    r"\b(?:Q[1-4]|20\d\d|19\d\d|annual|quarterly|fiscal|yesterday|today|recent|last\s+(?:year|month|quarter)|roadmap)\b",
    re.IGNORECASE,
)
COMPARISON_REGEX = re.compile(
    r"\b(?:vs|versus|compared?\s+to|difference\s+between|differences\s+between|alternative\s+to|better\s+than)\b",
    re.IGNORECASE,
)
CONCEPTUAL_REGEX = re.compile(
    r"\b(?:breaking\s+changes?|architecture|strategy|roadmap|sentiment|migrat(?:e|ion|ing)|security|scalability|implications|trends?|philosophy|overview|mechanisms?|capabilities)\b",
    re.IGNORECASE,
)
INTERROGATIVE_START_REGEX = re.compile(
    r"^\s*(?:what|why|how|when|where|who|which)\b",
    re.IGNORECASE,
)

# Common stopwords to exclude from heuristic capitalized entity detection
COMMON_STOPWORDS = {
    "A", "An", "The", "In", "On", "At", "To", "For", "With", "By", "About", "Against",
    "Between", "Into", "Through", "During", "Before", "After", "Above", "Below", "From",
    "Up", "Down", "In", "Out", "Off", "Over", "Under", "Again", "Further", "Then", "Once",
    "Here", "There", "When", "Where", "Why", "How", "All", "Any", "Both", "Each", "Few",
    "More", "Most", "Other", "Some", "Such", "No", "Nor", "Not", "Only", "Own", "Same",
    "So", "Than", "Too", "Very", "Can", "Will", "Just", "Don", "Should", "Now", "Is", "Are",
    "Was", "Were", "What", "Which", "Who", "Whom", "This", "That", "These", "Those", "Explain",
}


class QueryFeatureExtractor:
    """Extracts deterministic query characteristics and signals without external dependencies or LLMs."""

    def __init__(self, known_entities: Sequence[str] | None = None) -> None:
        """Initialize extractor with optional known entity dictionary.

        Args:
            known_entities: Sequence of canonical entity names (e.g. competitors, products).
        """
        self._known_entities = list(known_entities) if known_entities else []

    def extract(
        self,
        query: str,
        known_entities_override: Sequence[str] | None = None,
    ) -> QueryFeatures:
        """Extract structural, lexical, entity, and semantic features from query text.

        Args:
            query: Raw query string.
            known_entities_override: Optional override for known entity list.

        Returns:
            QueryFeatures model populated with all deterministic signals.
        """
        clean_query = query.strip()
        tokens = clean_query.split() if clean_query else []
        token_count = len(tokens)
        char_count = len(query)

        # 1. Identifier & Version signals
        version_matches = VERSION_REGEX.findall(clean_query)
        id_matches = IDENTIFIER_REGEX.findall(clean_query)
        model_matches = MODEL_NAME_REGEX.findall(clean_query)
        # Deduplicate while preserving order
        all_identifiers: list[str] = []
        for match in id_matches + model_matches:
            if match not in all_identifiers:
                all_identifiers.append(match)

        quoted_phrases = QUOTE_REGEX.findall(query)
        has_quotes = len(quoted_phrases) > 0

        # 2. Numeric & Financial signals
        currency_matches = CURRENCY_REGEX.findall(clean_query)
        percentage_matches = PERCENTAGE_REGEX.findall(clean_query)
        metric_matches = FINANCIAL_METRIC_REGEX.findall(clean_query)
        numeric_patterns: list[str] = []
        for match in currency_matches + percentage_matches + metric_matches:
            if match not in numeric_patterns:
                numeric_patterns.append(match)

        # 3. Entity signals
        active_known_entities = (
            list(known_entities_override)
            if known_entities_override is not None
            else self._known_entities
        )
        detected_known_entities: list[str] = []
        for entity in active_known_entities:
            entity_pattern = rf"\b{re.escape(entity)}\b"
            if re.search(entity_pattern, clean_query, re.IGNORECASE):
                if entity not in detected_known_entities:
                    detected_known_entities.append(entity)

        heuristic_entities = self._extract_heuristic_entities(tokens)

        # 4. Semantic & Analytical signals
        comparison_matches = [m.lower() for m in COMPARISON_REGEX.findall(clean_query)]
        comparison_terms = list(dict.fromkeys(comparison_matches))

        temporal_matches = [m.lower() for m in TEMPORAL_REGEX.findall(clean_query)]
        temporal_terms = list(dict.fromkeys(temporal_matches))

        conceptual_matches = [m.lower() for m in CONCEPTUAL_REGEX.findall(clean_query)]
        conceptual_terms = list(dict.fromkeys(conceptual_matches))

        is_interrogative = bool(INTERROGATIVE_START_REGEX.search(clean_query))

        return QueryFeatures(
            raw_query=query,
            token_count=token_count,
            char_count=char_count,
            version_patterns=version_matches,
            identifier_patterns=all_identifiers,
            has_quotes=has_quotes,
            quoted_phrases=quoted_phrases,
            numeric_patterns=numeric_patterns,
            known_entities=detected_known_entities,
            heuristic_entities=heuristic_entities,
            comparison_terms=comparison_terms,
            temporal_terms=temporal_terms,
            conceptual_terms=conceptual_terms,
            is_interrogative=is_interrogative,
        )

    def _extract_heuristic_entities(self, tokens: list[str]) -> list[str]:
        """Extract capitalized words as weak heuristic entity candidates (excluding 1st word & stopwords)."""
        if len(tokens) <= 1:
            return []

        candidates: list[str] = []
        # Skip token at index 0 because sentence capitalization is not a reliable entity signal
        for token in tokens[1:]:
            # Strip punctuation
            clean_token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
            if clean_token and clean_token[0].isupper() and clean_token not in COMMON_STOPWORDS:
                if clean_token not in candidates:
                    candidates.append(clean_token)
        return candidates
