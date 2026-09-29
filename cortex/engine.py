"""
Cortex: the top-level object wiring together the LLM, search tool, graph
memory and vector memory, and exposing a simple `run()` generator that both
the CLI and the FastAPI SSE endpoint stream from.

Memory (graph_store / vector_store) is constructed ONCE and reused across
every call to `run()` — that persistence, not any single request/response, is
the point of the whole project. `epoch` only controls which offline fixture
corpus a given research pass sees (simulating "time passing" between two
research sessions); it never resets memory.
"""
from __future__ import annotations

from cortex.config import settings
from cortex.llm import get_llm_client
from cortex.memory.graph_store import GraphStore, get_graph_store
from cortex.memory.vector_store import VectorStore
from cortex.models import Fact
from cortex.pipeline import PipelineState, build_pipeline
from cortex.search import get_search_tool


class Cortex:
    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.graph_store: GraphStore = get_graph_store(
            settings.graph_backend,
            snapshot_path=settings.graph_snapshot_path,
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        self.vector_store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)

    def run(self, query: str, epoch: int = 1):
        """Yields {'type': 'step', ...} events live, then one {'type': 'done', ...}."""
        search_tool = get_search_tool(settings.search_backend, epoch=epoch)
        graph = build_pipeline(self.llm, search_tool, self.graph_store, self.vector_store)

        initial: PipelineState = {"query": query, "epoch": epoch, "revise_count": 0}
        seen = 0
        final_state: PipelineState = {}
        for state in graph.stream(initial, stream_mode="values"):
            events = state.get("events", [])
            for event in events[seen:]:
                yield {"type": "step", **event}
            seen = len(events)
            final_state = state

        yield {
            "type": "done",
            "final_answer": final_state.get("final_answer", ""),
            "revise_count": final_state.get("revise_count", 0),
            "contradiction_count": len(final_state.get("contradictions", [])),
        }

    def snapshot(self) -> dict:
        return self.graph_store.snapshot()

    def history(self, subject: str, relation: str | None = None) -> list[Fact]:
        return self.graph_store.get_history(subject, relation)

    def close(self) -> None:
        self.graph_store.close()


_instance: Cortex | None = None


def get_cortex() -> Cortex:
    global _instance
    if _instance is None:
        _instance = Cortex()
    return _instance
