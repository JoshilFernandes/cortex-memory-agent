import pytest

from cortex import config, engine


@pytest.fixture
def cortex_instance(tmp_path, monkeypatch):
    """A Cortex instance with isolated, per-test storage (no shared state)."""
    monkeypatch.setattr(config.settings, "graph_snapshot_path", str(tmp_path / "graph.json"))
    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(config.settings, "chroma_collection", "test_chunks")
    engine._instance = None
    instance = engine.Cortex()
    yield instance
    instance.close()
    engine._instance = None
