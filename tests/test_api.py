from fastapi.testclient import TestClient

from governance_rag.api import app


def test_health_check():
    # Not using the lifespan: models are not loaded, /health must still answer.
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "models_loaded" in r.json()


def test_corpus_endpoint_lists_manifest():
    r = TestClient(app).get("/corpus")
    assert r.status_code == 200
    body = r.json()
    assert body["as_of"]
    ids = {s["id"] for s in body["sources"]}
    assert {"eu_ai_act", "arxiv_papers"} <= ids
    assert all("papers" not in s for s in body["sources"])


def test_retrieve_without_index_is_503():
    r = TestClient(app).post("/retrieve", json={"question": "x"})
    assert r.status_code == 503
