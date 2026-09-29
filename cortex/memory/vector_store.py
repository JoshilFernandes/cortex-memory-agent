"""
Vector memory (the "RAG" half of Graph-RAG).

Chunks of raw research text are embedded and stored so that retrieval isn't
limited to whatever made it into the graph as a clean triple — the full
source passage is still searchable semantically. Chroma is the vector
engine; the *embedding function* is pluggable:

- HashingEmbeddingFunction (default): a deterministic, offline, dependency-
  free bag-of-words hashing embedding. It has no idea what a synonym is, but
  it needs no model download and gives 100%-reproducible results, which is
  exactly what the demo, the tests and CI need.
- Sentence-Transformers ("all-MiniLM-L6-v2"): real semantic embeddings, one
  extra dependency + a one-off model download. Set EMBEDDING_BACKEND=st.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from cortex.config import settings

HASH_DIMS = 256


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic, offline embedding. Subclasses chromadb's EmbeddingFunction
    protocol to inherit its default embed_query/is_legacy/space plumbing."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - chroma's API
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:  # required by newer chromadb versions
        return "cortex-hashing-embedding-v1"

    def get_config(self) -> dict:
        return {"dims": HASH_DIMS}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * HASH_DIMS
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % HASH_DIMS
            sign = 1.0 if int(digest, 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def get_embedding_function():
    if settings.embedding_backend == "st":
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return HashingEmbeddingFunction()


@dataclass
class Chunk:
    id: str
    text: str
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        client = chromadb.PersistentClient(path=persist_dir)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

    def add(self, chunk_id: str, text: str, metadata: dict) -> None:
        self._collection.upsert(ids=[chunk_id], documents=[text], metadatas=[metadata])

    def query(self, text: str, k: int = 5) -> list[Chunk]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(query_texts=[text], n_results=min(k, self._collection.count()))
        chunks: list[Chunk] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            chunks.append(Chunk(id=cid, text=doc, score=1.0 - dist, metadata=meta or {}))
        return chunks
