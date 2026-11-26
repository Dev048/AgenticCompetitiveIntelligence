"""PreToolUse hook: block Claude's Edit/Write on .env files and the demo LLM replay fixtures.

Exit 2 blocks the tool call and shows the message to Claude. This guards against accidents,
not a determined bypass: shell commands are not covered (scripts/seed_demo.py --record writes
the fixtures through Python, which is the intended path).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "src" / "competitive_intelligence" / "demo" / "fixtures" / "llm_replay.json"
FIXTURE_OVERRIDE = ROOT / ".claude" / "allow-fixture-edit"  # gitignored; the user creates it


def main() -> int:
    payload = json.load(sys.stdin)
    raw = (payload.get("tool_input") or {}).get("file_path") or ""
    if not raw:
        return 0
    path = Path(raw)
    name = path.name

    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        print(
            f"Blocked: {name} holds secrets (API keys) and is never edited by Claude. "
            "Edit it yourself if a value must change.",
            file=sys.stderr,
        )
        return 2

    if path.resolve() == FIXTURES.resolve() and not FIXTURE_OVERRIDE.exists():
        print(
            "Blocked: llm_replay.json holds recorded LLM responses; a hand edit silently turns "
            "the demo's 15/15 exact replays into fallbacks or misses.\n"
            "To change it deliberately:\n"
            "  1. Preferred: re-record with `python scripts/seed_demo.py --record` (see its "
            "docstring), then check a fresh replay reports 15/15 exact hits.\n"
            "  2. Hand edit on purpose: the user creates the empty file "
            ".claude/allow-fixture-edit, the edit is retried, then the user deletes that file.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
