"""
Central configuration for Cortex.

Everything here has a working default that needs zero external services or
API keys, so `git clone && pip install -r requirements.txt && uvicorn cortex.api:app`
works out of the box. Every knob can be swapped to a "real" backend via
environment variables for a production-grade deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is an optional convenience
    pass


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM backend -------------------------------------------------
    # "mock"  -> deterministic templated responses, zero dependencies, used
    #            by default and by the whole test suite / CI.
    # "groq"  -> real LLM calls via the Groq API (needs GROQ_API_KEY).
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "mock"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # --- Search backend ------------------------------------------------
    # "offline" -> a small, deliberately time-versioned fixture corpus used
    #              for deterministic demos and tests (see cortex/demo_corpus.py)
    # "live"    -> real web search via DuckDuckGo (ddgs), no API key needed.
    search_backend: str = field(default_factory=lambda: os.getenv("SEARCH_BACKEND", "offline"))

    # --- Graph memory backend ------------------------------------------
    # "memory" -> networkx graph persisted to a local JSON file. No server.
    # "neo4j"  -> real Neo4j via bolt driver (see docker-compose.yml).
    graph_backend: str = field(default_factory=lambda: os.getenv("GRAPH_BACKEND", "memory"))
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "cortex-dev-password"))
    graph_snapshot_path: str = field(default_factory=lambda: os.getenv("GRAPH_SNAPSHOT_PATH", "./data/graph_memory.json"))

    # --- Vector memory backend ------------------------------------------
    # "hash"  -> deterministic offline hashing embedding, zero downloads.
    # "st"    -> sentence-transformers ("all-MiniLM-L6-v2"), real semantic
    #            embeddings, needs a one-off model download.
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "hash"))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "cortex_chunks"))

    # --- Agent pipeline ---------------------------------------------------
    max_revise_loops: int = field(default_factory=lambda: int(os.getenv("MAX_REVISE_LOOPS", "2")))
    researcher_fanout: int = field(default_factory=lambda: int(os.getenv("RESEARCHER_FANOUT", "3")))

    # --- API ---------------------------------------------------------------
    cors_allow_all: bool = field(default_factory=lambda: _bool("CORS_ALLOW_ALL", True))


settings = Settings()
