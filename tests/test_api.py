from fastapi.testclient import TestClient

from cortex import config, engine
from cortex.api import app


def _reset_storage(tmp_path):
    config.settings.graph_snapshot_path = str(tmp_path / "graph.json")
    config.settings.chroma_persist_dir = str(tmp_path / "chroma")
    config.settings.chroma_collection = "api_test_chunks"
    engine._instance = None


def test_health():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_graph_and_history_endpoints_start_empty_then_populate(tmp_path):
    _reset_storage(tmp_path)
    client = TestClient(app)

    empty = client.get("/api/graph").json()
    assert empty["fact_count"] == 0

    with client.stream(
        "GET",
        "/api/research/stream",
        params={"query": "Who leads AgentBench?", "epoch": 1},
    ) as response:
        events = [line for line in response.iter_lines() if line.startswith("data:")]
    assert any('"type": "done"' in e or '"type":"done"' in e for e in events)

    populated = client.get("/api/graph").json()
    assert populated["fact_count"] > 0

    history = client.get("/api/history", params={"subject": "AgentBench", "relation": "leader"}).json()
    assert history["facts"][0]["object"] == "Arcline AI"

    engine._instance = None
