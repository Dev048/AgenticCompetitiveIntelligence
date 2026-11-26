"""The Space bundle script must never delete a folder that is not its own output."""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_space_bundle.py"
_spec = importlib.util.spec_from_file_location("build_space_bundle", _SCRIPT)
assert _spec and _spec.loader
bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bundle)


def test_refuses_non_empty_folder_without_marker(tmp_path: Path) -> None:
    keep = tmp_path / "important.txt"
    keep.write_text("do not delete", encoding="utf-8")

    with pytest.raises(SystemExit, match="not a previous bundle"):
        bundle.build(tmp_path)

    assert keep.read_text(encoding="utf-8") == "do not delete"


def test_refuses_a_file(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a directory"):
        bundle.ensure_safe_to_replace(target)


def test_allows_missing_empty_and_previous_bundle(tmp_path: Path) -> None:
    bundle.ensure_safe_to_replace(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    bundle.ensure_safe_to_replace(empty)
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / bundle.MARKER).write_text("", encoding="utf-8")
    (previous / "Dockerfile").write_text("", encoding="utf-8")
    bundle.ensure_safe_to_replace(previous)


def test_rebuild_over_own_output(tmp_path: Path) -> None:
    out = tmp_path / "space"
    bundle.build(out)
    (out / "stale.txt").write_text("old", encoding="utf-8")
    bundle.build(out)
    assert (out / bundle.MARKER).is_file()
    assert (out / "Dockerfile").is_file()
    assert not (out / "stale.txt").exists()
