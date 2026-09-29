from cortex.memory.graph_store import InMemoryGraphStore
from cortex.memory.hybrid_retriever import HybridRetriever
from cortex.memory.vector_store import VectorStore
from cortex.models import Fact


def test_hybrid_retrieval_combines_graph_and_vector_signal(tmp_path):
    graph = InMemoryGraphStore(snapshot_path=str(tmp_path / "graph.json"))
    vectors = VectorStore(str(tmp_path / "chroma"), "test")

    fact = Fact(
        subject="AgentBench",
        relation="leader",
        object="Arcline AI",
        confidence=0.9,
        source_url="https://example.test/1",
        source_title="AgentBench update",
        observed_at="2026-03-10",
    )
    graph.add_fact(fact)
    vectors.add(
        "chunk-1",
        "AgentBench is led by Arcline AI with a 91 percent completion rate.",
        {"source_title": "AgentBench update", "source_url": "https://example.test/1", "entities": "AgentBench|Arcline AI"},
    )
    vectors.add(
        "chunk-2",
        "Unrelated passage about quarterly gardening trends in Berlin.",
        {"source_title": "Unrelated", "source_url": "https://example.test/2", "entities": ""},
    )

    retriever = HybridRetriever(graph, vectors)
    result = retriever.retrieve("Who leads AgentBench?")

    assert any(f.subject == "AgentBench" for f in result.facts)
    assert any("Arcline AI" in c.text for c in result.chunks)
    assert "AgentBench" in result.seed_entities

    context = result.as_context_text()
    assert "AgentBench" in context
    assert "leader" in context


def test_neighborhood_expands_via_graph_edges(tmp_path):
    graph = InMemoryGraphStore(snapshot_path=str(tmp_path / "graph.json"))
    graph.add_fact(
        Fact(
            subject="Meridian Systems",
            relation="acquired",
            object="Fenwick Labs",
            confidence=0.9,
            source_url="u",
            source_title="t",
            observed_at="2026-09-20",
        )
    )
    graph.add_fact(
        Fact(
            subject="Meridian Systems",
            relation="raised",
            object="$75M Series C",
            confidence=0.9,
            source_url="u",
            source_title="t",
            observed_at="2026-09-20",
        )
    )

    neighbors = graph.neighborhood(["Fenwick Labs"], hops=1)
    subjects_and_objects = {f.subject for f in neighbors} | {f.object for f in neighbors}
    assert "Meridian Systems" in subjects_and_objects
    # 1-hop from Fenwick Labs should surface the acquisition fact itself, and
    # walking back out from Meridian Systems should also reach the funding fact.
    assert any(f.relation == "acquired" for f in neighbors)
