"""FastAPI app: kicks off research runs and streams their progress over SSE."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from cortex.config import settings
from cortex.engine import get_cortex

app = FastAPI(title="Cortex", description="A multi-agent research system with persistent Graph-RAG memory.")

if settings.cors_allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/graph")
def graph() -> dict:
    return get_cortex().snapshot()


@app.get("/api/history")
def history(subject: str, relation: str | None = None) -> dict:
    facts = get_cortex().history(subject, relation)
    return {"subject": subject, "relation": relation, "facts": [f.to_dict() for f in facts]}


@app.get("/api/research/stream")
def research_stream(query: str = Query(...), epoch: int = Query(1)):
    cortex = get_cortex()

    def event_source():
        for event in cortex.run(query, epoch=epoch):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


if os.path.isdir(_FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
