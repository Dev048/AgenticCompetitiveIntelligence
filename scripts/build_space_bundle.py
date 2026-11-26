"""Assemble the Hugging Face Space repository contents into dist/space/.

Spaces build from a root ``Dockerfile`` and read YAML front-matter from the root
``README.md``. Neither belongs at the root of the main repository (the compose stack has its
own Dockerfile, and front-matter would render as a table on GitHub), so this script copies
exactly what the Space needs:

    dist/space/
      Dockerfile      <- Dockerfile.space
      README.md       <- deploy/huggingface/README.md (front-matter: sdk docker, port 7860)
      .dockerignore
      pyproject.toml
      src/            (without caches)

Usage:
    python scripts/build_space_bundle.py [--out dist/space]
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    shutil.copyfile(ROOT / "Dockerfile.space", out / "Dockerfile")
    shutil.copyfile(ROOT / "deploy" / "huggingface" / "README.md", out / "README.md")
    shutil.copyfile(ROOT / ".dockerignore", out / ".dockerignore")
    shutil.copyfile(ROOT / "pyproject.toml", out / "pyproject.toml")
    shutil.copytree(
        ROOT / "src",
        out / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )

    forbidden = [p for p in out.rglob("*") if p.name == ".env" or p.suffix in {".db", ".whl"}]
    if forbidden:
        raise SystemExit(f"Refusing to build: unexpected files in bundle: {forbidden}")

    files = sum(1 for p in out.rglob("*") if p.is_file())
    size_kb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) // 1024
    print(f"Space bundle written to {out} ({files} files, {size_kb} KiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the Hugging Face Space bundle.")
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "space")
    build(parser.parse_args().out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
