"""PreToolUse hook: block Claude's Edit/Write on .env files.

Exit 2 blocks the tool call and shows the message to Claude. This guards against accidents,
not a determined bypass: shell commands are not covered.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    payload = json.load(sys.stdin)
    raw = (payload.get("tool_input") or {}).get("file_path") or ""
    if not raw:
        return 0
    name = Path(raw).name

    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        print(
            f"Blocked: {name} holds secrets (API keys) and is never edited by Claude. "
            "Edit it yourself if a value must change.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
