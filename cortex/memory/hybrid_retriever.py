"""
Graph-RAG: hybrid retrieval that combines vector similarity over raw text
chunks with graph traversal over structured facts.

The two signals cover different failure modes of pure-vector RAG:
  * vector search finds passages that are *semantically* similar to the
    query, even if they don't share exact entity names;
  * graph traversal finds facts that are *structurally* connected to the
    entities those passages (or the query itself) mention, including facts
    that came from a totally different document than the one vector search
    surfaced.

Combining both means Cortex can answer "what do we currently believe about
AgentBench's leader, and how did that answer change over time" even when the
literal string "AgentBench" doesn't appear in the highest-scoring chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from cortex.memory.graph_store import GraphStore
from cortex.memory.vector_store import Chunk, VectorStore
from cortex.models import Fact

_CAPITALIZED_RUN = re.compile(r"\b([A-Z][\w&]*(?:\s+[A-Z][\w&]*){0,3})\b")


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    facts: list[Fact]
    seed_entities: list[str] = field(default_factory=list)

    def as_context_text(self) -> str:
        parts = []
        if self.facts:
            parts.append("Known graph facts (subject | relation | object | status):")
            for f in self.facts:
                parts.append(f"- {f.subject} | {f.relation} | {f.object} | {f.status}")
        if self.chunks:
            parts.append("\nRelevant source passages:")
            for c in self.chunks:
                parts.append(f"- ({c.metadata.get('source_title', 'unknown')}) {c.text}")
        return "\n".join(parts)


class HybridRetriever:
    def __init__(self, graph_store: GraphStore, vector_store: VectorStore) -> None:
        self.graph = graph_store
        self.vectors = vector_store

    def retrieve(self, query: str, *, k_chunks: int = 5, hops: int = 1) -> RetrievalResult:
        chunks = self.vectors.query(query, k=k_chunks)

        seed_entities: set[str] = set(_extract_candidate_entities(query))
        for chunk in chunks:
            seed_entities.update(chunk.metadata.get("entities", "").split("|"))
        seed_entities.discard("")

        facts = self.graph.neighborhood(list(seed_entities), hops=hops) if seed_entities else []
        if not facts:
            # Fall back to the whole active graph when nothing matched yet
            # (e.g. the very first query before any entities are known) —
            # keeps the demo usable from a cold start.
            facts = self.graph.get_active_facts()

        return RetrievalResult(chunks=chunks, facts=facts, seed_entities=sorted(seed_entities))


def _extract_candidate_entities(text: str) -> list[str]:
    return [m.strip() for m in _CAPITALIZED_RUN.findall(text)]
