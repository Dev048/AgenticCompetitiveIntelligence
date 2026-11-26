"""PostToolUse hook: ruff the edited .py file, reporting only issues on changed lines.

The repo has many pre-existing ruff findings, so auto-fixing the whole file would rewrite
code nobody touched. Instead this reports (exit 2, shown to Claude) only findings on lines
that differ from HEAD; an untracked file counts as entirely changed. It never edits files.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def changed_lines(path: Path) -> set[int] | None:
    """Line numbers changed vs HEAD; None means the whole file is new."""
    if run("git", "ls-files", "--error-unmatch", str(path)).returncode != 0:
        return None
    diff = run("git", "diff", "-U0", "--no-color", "HEAD", "--", str(path)).stdout
    lines: set[int] = set()
    for start, count in HUNK.findall(diff):
        lines.update(range(int(start), int(start) + int(count or 1)))
    return lines


def main() -> int:
    payload = json.load(sys.stdin)
    raw = (payload.get("tool_input") or {}).get("file_path") or ""
    path = Path(raw)
    if path.suffix != ".py" or not path.is_file():
        return 0
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return 0  # outside the repo (scratch files etc.)

    changed = changed_lines(path)
    if changed is not None and not changed:
        return 0
    result = run(sys.executable, "-m", "ruff", "check", "--output-format", "json", "--quiet",
                 str(path))
    if not result.stdout.strip().startswith("["):
        print(f"ruff hook could not run: {result.stderr.strip()[:300]}", file=sys.stderr)
        return 1  # non-blocking
    hits = [
        f
        for f in json.loads(result.stdout)
        if changed is None
        or changed.intersection(range(f["location"]["row"], f["end_location"]["row"] + 1))
    ]
    if not hits:
        return 0
    print(f"ruff found {len(hits)} issue(s) on lines you changed in {path.name}:", file=sys.stderr)
    for f in hits:
        print(f"  line {f['location']['row']}: {f['code']} {f['message']}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
