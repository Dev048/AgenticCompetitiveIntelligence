"""GET .../recommendations/{id}/provenance returns every layer's details, not just IDs."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from competitive_intelligence.application.api import deps
from competitive_intelligence.application.api.app import create_app
from competitive_intelligence.core.config import get_settings

BASE = "/api/v1/workspaces/test-workspace"  # seeded by the seeded_workspace fixture


@pytest.fixture
def client(seeded_workspace: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    for key, value in seeded_workspace.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client
    deps.reset_resource_caches()
    get_settings.cache_clear()


def test_provenance_exposes_all_ten_layers_with_details(client: TestClient) -> None:
    recommendations = client.get(f"{BASE}/recommendations").json()
    assert recommendations, "seeded workspace should have recommendations"

    for rec in recommendations:
        body = client.get(f"{BASE}/recommendations/{rec['recommendation_id']}/provenance").json()
        layers = body["layers"]

        assert layers["recommendation"]["recommendation_id"] == rec["recommendation_id"]
        assert layers["recommendation"]["status"] == "proposed"
        assert layers["decision_context"]["decision_id"] == rec["decision_id"]
        assert body["selected_option"] is not None
        assert layers["insights"] and layers["events"]
        assert layers["evidence"] and all(e["quote"] for e in layers["evidence"])
        assert layers["chunks"] and all(c["content_hash"] for c in layers["chunks"])
        assert layers["versions"] and all(len(v["content_hash"]) == 64 for v in layers["versions"])
        assert layers["documents"] and layers["sources"]
        assert all(s["url"].startswith("https://") for s in layers["sources"])

        # Details and the legacy count/ID fields describe the same lineage.
        assert len(layers["events"]) == body["event_count"]
        assert len(layers["evidence"]) == body["evidence_count"]
        assert {c["chunk_id"] for c in layers["chunks"]} == set(body["chunk_ids"])
        assert {s["source_id"] for s in layers["sources"]} == set(body["source_ids"])

        # Every evidence item points at a chunk/version/source present in the lower layers.
        chunk_ids = {c["chunk_id"] for c in layers["chunks"]}
        version_ids = {v["version_id"] for v in layers["versions"]}
        for ev in layers["evidence"]:
            assert ev["chunk_id"] in chunk_ids
            assert ev["version_id"] in version_ids


def test_provenance_for_other_workspace_is_not_found(client: TestClient) -> None:
    rec_id = client.get(f"{BASE}/recommendations").json()[0]["recommendation_id"]
    response = client.get(f"/api/v1/workspaces/not-a-workspace/recommendations/{rec_id}/provenance")
    assert response.status_code == 404
