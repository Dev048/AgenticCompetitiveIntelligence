"""State comparators for evaluating domain transitions and directionality."""

import re
from typing import Any

from competitive_intelligence.intelligence.signals.interfaces import BaseStateComparator


class NumericStateComparator(BaseStateComparator):
    """Comparator for numeric states and structured numeric dictionaries (e.g. price_usd, headcount)."""

    def is_reversal(self, initial_state: Any, current_state: Any) -> bool:
        """Return True if current_state restores initial_state numerically or structurally."""
        if initial_state is None or current_state is None:
            return False

        v_init = self._extract_numeric(initial_state)
        v_curr = self._extract_numeric(current_state)
        if v_init is not None and v_curr is not None:
            return abs(v_init - v_curr) < 1e-6

        return bool(initial_state == current_state)

    def compare_direction(self, state_a: Any, state_b: Any) -> int | None:
        """Compare two states: returns 1 if increasing, -1 if decreasing, 0 if equal, None if unorderable."""
        v_a = self._extract_numeric(state_a)
        v_b = self._extract_numeric(state_b)
        if v_a is not None and v_b is not None:
            if v_b > v_a:
                return 1
            if v_b < v_a:
                return -1
            return 0
        return None

    def _extract_numeric(self, state: Any) -> float | None:
        """Extract float value from a numeric literal, string, or state dict."""
        if state is None:
            return None
        if isinstance(state, (int, float)):
            return float(state)
        if isinstance(state, dict):
            for k in ("price_usd", "price", "amount", "value", "count", "headcount", "rate"):
                if k in state and state[k] is not None:
                    val = self._extract_numeric(state[k])
                    if val is not None:
                        return val
        if isinstance(state, str):
            clean = state.strip()
            # Match $20, $30.50, 50M, 100
            num_match = re.search(r"[\$]?([\d\.]+)\s*([MBKmbk])?", clean)
            if num_match:
                try:
                    base = float(num_match.group(1))
                    unit = num_match.group(2)
                    if unit:
                        u_low = unit.lower()
                        if u_low == "k":
                            base *= 1_000
                        elif u_low == "m":
                            base *= 1_000_000
                        elif u_low == "b":
                            base *= 1_000_000_000
                    return base
                except ValueError:
                    pass
        return None


class ExactMatchStateComparator(BaseStateComparator):
    """Comparator using strict structural equality without ordering."""

    def is_reversal(self, initial_state: Any, current_state: Any) -> bool:
        if initial_state is None or current_state is None:
            return False
        return bool(initial_state == current_state)

    def compare_direction(self, state_a: Any, state_b: Any) -> int | None:
        if state_a == state_b:
            return 0
        return None
