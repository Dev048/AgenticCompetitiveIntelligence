"""Seed the public demo workspace, or record its LLM fixtures once against a real provider.

Replay (default; no LLM calls, uses the committed fixtures):
    python scripts/seed_demo.py --database-url sqlite+aiosqlite:///./demo.db \
        --qdrant-path ./qdrant

Record (calls the LLM configured in the environment, e.g. Groq through OPENAI_BASE_URL,
and rewrites src/competitive_intelligence/demo/fixtures/llm_replay.json). Recording always
starts from an empty database and vector store so the call sequence matches a fresh boot:
    python scripts/seed_demo.py --record --database-url sqlite+aiosqlite:///./rec.db \
        --qdrant-path ./rec-qdrant

Embeddings default to fastembed (bge-small), the demo's provider: prompts depend on what
retrieval returns, so record and replay must use the same embedder.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

_SQLITE_PREFIX = "sqlite+aiosqlite:///"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed the demo or record its LLM fixtures.")
    parser.add_argument("--record", action="store_true", help="record fixtures from the real LLM")
    parser.add_argument("--database-url", required=True, help="sqlite+aiosqlite:///<file>")
    parser.add_argument("--qdrant-path", required=True, help="embedded Qdrant folder")
    parser.add_argument("--fixtures", type=Path, default=None, help="fixture file path")
    parser.add_argument(
        "--embedding-provider",
        choices=["fastembed", "bge_m3", "mock"],
        default="fastembed",
        help="must match the demo when recording (default: fastembed)",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="replay only: fail unless every LLM call is an exact fixture hit (image builds)",
    )
    return parser.parse_args()


def _sqlite_file(database_url: str) -> Path:
    if not database_url.startswith(_SQLITE_PREFIX) or ":memory:" in database_url:
        raise SystemExit("Only file-backed sqlite+aiosqlite URLs are supported.")
    return Path(database_url[len(_SQLITE_PREFIX) :])


async def _main(args: argparse.Namespace) -> int:
    _sqlite_file(args.database_url)
    # Configure before importing settings-dependent modules. ENVIRONMENT is forced: with a
    # local .env saying "production", the vector store would ignore QDRANT_PATH and write
    # to whatever Qdrant server QDRANT_HOST/QDRANT_URL point at.
    os.environ["ENVIRONMENT"] = "development"
    os.environ["DATABASE_URL"] = args.database_url
    os.environ["QDRANT_PATH"] = args.qdrant_path
    os.environ["QDRANT_URL"] = ""
    os.environ["QDRANT_LOCATION"] = ""
    os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    os.environ.setdefault("MONITORING_MODE", "disabled")

    from competitive_intelligence.agentic_rag.llm import create_llm_client
    from competitive_intelligence.application.api.deps import get_vector_store
    from competitive_intelligence.core.config import get_settings
    from competitive_intelligence.core.logger import configure_logging
    from competitive_intelligence.demo.replay_llm import LLMFixtureStore, ReplayLLMClient
    from competitive_intelligence.demo.seed import DEFAULT_FIXTURES_PATH, seed_demo_workspace

    settings = get_settings()
    configure_logging(log_level="INFO", log_format="console")
    fixtures: Path = args.fixtures or DEFAULT_FIXTURES_PATH

    if args.record:
        # Fresh stores, removed before the embedded Qdrant folder is opened (and locked).
        _sqlite_file(args.database_url).unlink(missing_ok=True)
        shutil.rmtree(args.qdrant_path, ignore_errors=True)

    store_path = getattr(getattr(get_vector_store(), "config", None), "path", None)
    if store_path != args.qdrant_path:
        raise SystemExit(f"Refusing to seed: vector store is not the embedded one ({store_path!r}).")

    if args.record:
        if settings.llm_provider == "mock":
            raise SystemExit("Recording needs a real LLM provider (LLM_PROVIDER=openai|anthropic).")
        inner = create_llm_client()
        client = ReplayLLMClient(LLMFixtureStore(fixtures), mode="record", inner=inner)
        print(f"Recording with {inner.provider_name}:{inner.model_name} -> {fixtures}")
    else:
        client = ReplayLLMClient(LLMFixtureStore.load(fixtures))

    summary = await seed_demo_workspace(llm_client=client)
    if args.record:
        client.store.save()
    print(json.dumps(summary.model_dump(), indent=2))
    if summary.steps_failed:
        return 1
    stats = summary.llm
    if args.require_exact and not args.record and (stats.misses or stats.ordinal_hits):
        print(f"Replay was not exact: {stats.model_dump()}. Re-record the fixtures.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(_parse_args())))
