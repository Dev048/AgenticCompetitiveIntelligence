"""The demo process must never load torch or sentence-transformers (512 MB free-tier host)."""

import json
import os
import subprocess
import sys
from pathlib import Path

_PROBE = r"""
import json, sys, types

class _FakeTextEmbedding:  # stands in for fastembed so the test needs no model download
    def __init__(self, model_name, threads=None): pass
    @staticmethod
    def list_supported_models(): return [{"model": "BAAI/bge-small-en-v1.5", "dim": 384}]
    def passage_embed(self, texts, batch_size=256):
        for _ in texts: yield [0.1] * 384
    def query_embed(self, query): yield [0.1] * 384

fake = types.ModuleType("fastembed"); fake.TextEmbedding = _FakeTextEmbedding
sys.modules["fastembed"] = fake

from fastapi.testclient import TestClient
from competitive_intelligence.application.api import deps
from competitive_intelligence.application.api.app import create_app

with TestClient(create_app()) as client:
    assert client.get("/health/live").status_code == 200
    deps.get_evidence_query_service()
    deps.get_embedding_service().provider.embed_query("Northwind pricing")
print(json.dumps({m: m in sys.modules for m in ("torch", "sentence_transformers", "crawl4ai")}))
"""


def test_demo_startup_and_query_path_never_import_torch(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("DEMO_", "QDRANT_"))},
        "PYTHONPATH": str(src),
        "ENVIRONMENT": "demo",
        "DEMO_MODE": "true",
        "DEMO_SEED_ON_STARTUP": "false",
        "DEBUG": "false",
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-not-a-real-key-123456",
        "EMBEDDING_PROVIDER": "fastembed",
        "DATABASE_URL": f"sqlite+aiosqlite:///{tmp_path / 'demo.db'}",
        "QDRANT_PATH": str(tmp_path / "qdrant"),
        "MONITORING_MODE": "disabled",
    }
    # cwd=tmp_path: no .env file is found, so local settings cannot leak in.
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=tmp_path, env=env, capture_output=True,
        text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {"torch": False, "sentence_transformers": False, "crawl4ai": False}
