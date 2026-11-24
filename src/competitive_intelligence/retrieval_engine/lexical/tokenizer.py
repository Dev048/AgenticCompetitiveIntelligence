"""Deterministic lexical tokenizer for Retrieval Engine subsystem.

Designed specifically for competitive intelligence text:
- Preserves pricing & currency expressions ($49.99, €99, £25)
- Preserves version strings (v2.1.0, 1.0.0-rc1)
- Preserves model names and hyphenated identifiers (gpt-4o, claude-3.5-sonnet, bge-m3)
- Preserves percentages (50%) and snake_case/dotted identifiers (api_key, workspace_id)
- Performs explicit lowercase normalization
- Avoids aggressive stemming/stopword removal to retain exact entity searchability
"""

import re


class LexicalTokenizer:
    """Deterministic tokenizer tailored for lexical retrieval over competitive intelligence entities."""

    # Matches currency/percentages, version numbers, hyphenated/dotted identifiers, or alphanumeric words
    TOKEN_PATTERN = re.compile(
        r"[\$€£]?\d+(?:\.\d+)?%?|[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*|\w+"
    )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize input text into normalized lexical terms.

        Args:
            text: Raw document content or query string.

        Returns:
            A list of lowercased lexical tokens.
        """
        if not text:
            return []

        raw_tokens = self.TOKEN_PATTERN.findall(text)
        tokens: list[str] = []

        for token in raw_tokens:
            cleaned = token.strip().lower()
            # Trim boundary punctuation that may attach to words (e.g. trailing period in sentence)
            cleaned = cleaned.strip(".-_")
            if cleaned:
                tokens.append(cleaned)

        return tokens
