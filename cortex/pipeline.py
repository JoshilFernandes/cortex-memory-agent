"""
The LangGraph pipeline: plan -> parallel research -> reconcile memory ->
propose -> critique -> (revise -> critique)* -> finalize.

Every node appends a small structured event describing what it just did.
Events are how the FastAPI layer streams live progress over SSE, and how the
frontend animates the knowledge graph growing in real time.
"""
from __future__ import annotations

import json
import operator
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from cortex.config import settings
from cortex.contradiction import ContradictionEvent, reconcile_facts
from cortex.extraction import extract_facts
from cortex.llm import LLMClient, stable_hash
from cortex.memory.graph_store import GraphStore
from cortex.memory.hybrid_retriever import HybridRetriever
from cortex.memory.vector_store import VectorStore
from cortex.models import Fact, normalize_entity
from cortex.prompts import CRITIQUE_SYSTEM, PLAN_SYSTEM, PROPOSE_SYSTEM, REVISE_SYSTEM
from cortex.search import SearchTool


class PipelineState(TypedDict, total=False):
    query: str
    epoch: int
    subqueries: list[str]
    research_records: list[dict]
    new_facts: list[Fact]
    contradictions: list[ContradictionEvent]
    context_text: str
    seed_entities: list[str]
    draft: str
    verdict: str
    critique_reason: str
    revise_count: int
    final_answer: str
    events: Annotated[list[dict], operator.add]


def build_pipeline(
    llm: LLMClient,
    search_tool: SearchTool,
    graph_store: GraphStore,
    vector_store: VectorStore,
):
    retriever = HybridRetriever(graph_store, vector_store)

    def _event(node: str, message: str, **extra) -> dict:
        return {"node": node, "message": message, **extra}

    def plan_node(state: PipelineState) -> dict:
        raw = llm.complete(PLAN_SYSTEM, state["query"], json_mode=True)
        try:
            subqueries = json.loads(raw).get("subqueries") or [state["query"]]
        except json.JSONDecodeError:
            subqueries = [state["query"]]
        return {
            "subqueries": subqueries,
            "events": [_event("plan", f"Planned {len(subqueries)} sub-queries", subqueries=subqueries)],
        }

    def research_node(state: PipelineState) -> dict:
        def work(subquery: str) -> list[dict]:
            docs = search_tool.search(subquery, k=2)
            records = []
            for doc in docs:
                facts = extract_facts(
                    llm,
                    text=doc.snippet,
                    source_url=doc.url,
                    source_title=doc.title,
                    observed_at=doc.published,
                )
                records.append({"subquery": subquery, "doc": doc, "facts": facts})
            return records

        with ThreadPoolExecutor(max_workers=max(settings.researcher_fanout, 1)) as pool:
            grouped = list(pool.map(work, state["subqueries"]))
        research_records = [r for group in grouped for r in group]
        new_facts = [f for r in research_records for f in r["facts"]]
        return {
            "research_records": research_records,
            "new_facts": new_facts,
            "events": [
                _event(
                    "research",
                    f"Researched {len(state['subqueries'])} sub-queries in parallel — "
                    f"{len(research_records)} documents, {len(new_facts)} candidate facts extracted",
                )
            ],
        }

    def reconcile_node(state: PipelineState) -> dict:
        existing_active = graph_store.get_active_facts()
        result = reconcile_facts(state.get("new_facts", []), existing_active)

        for old_fact, new_fact in result.facts_to_supersede:
            graph_store.mark_superseded(old_fact.id, new_fact.id)
        for fact in result.facts_to_add:
            graph_store.add_fact(fact)

        seen_urls: set[str] = set()
        for record in state.get("research_records", []):
            doc = record["doc"]
            if doc.url in seen_urls:
                continue
            seen_urls.add(doc.url)
            entities = sorted(
                {normalize_entity(f.subject) for f in record["facts"]}
                | {normalize_entity(f.object) for f in record["facts"]}
            )
            vector_store.add(
                chunk_id=stable_hash(doc.url),
                text=doc.snippet,
                metadata={
                    "source_title": doc.title,
                    "source_url": doc.url,
                    "entities": "|".join(entities),
                },
            )

        message = (
            f"Reconciled memory — {len(result.facts_to_add)} facts added "
            f"({result.duplicates_skipped} duplicates skipped), "
            f"{len(result.contradictions)} contradiction(s) detected"
        )
        return {
            "contradictions": result.contradictions,
            "events": [
                _event(
                    "reconcile_memory",
                    message,
                    contradictions=[c.reason for c in result.contradictions],
                )
            ],
        }

    def propose_node(state: PipelineState) -> dict:
        retrieval = retriever.retrieve(state["query"])
        context = retrieval.as_context_text()
        if state.get("contradictions"):
            context += (
                "\n\nNOTE: one or more facts above were just updated this run — "
                "treat any 'superseded' status as historical, not current."
            )
        draft = llm.complete(PROPOSE_SYSTEM, f"Question: {state['query']}\n\n{context}")
        return {
            "context_text": context,
            "draft": draft,
            "seed_entities": retrieval.seed_entities,
            "events": [_event("propose", "Drafted an answer from hybrid graph+vector retrieval")],
        }

    def critique_node(state: PipelineState) -> dict:
        revise_count = state.get("revise_count", 0)
        notice = ""
        if state.get("contradictions") and revise_count == 0:
            reasons = "; ".join(c.reason for c in state["contradictions"])
            notice = f"CONTRADICTION NOTICE: {reasons}"
        prompt = f"Draft answer:\n{state['draft']}\n\n{notice or 'No open issues.'}"
        raw = llm.complete(CRITIQUE_SYSTEM, prompt, json_mode=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"verdict": "ACCEPT", "reason": "could not parse critique"}
        verdict = parsed.get("verdict", "ACCEPT")
        reason = parsed.get("reason", "")
        return {
            "verdict": verdict,
            "critique_reason": reason,
            "events": [_event("critique", f"Critique verdict: {verdict} — {reason}")],
        }

    def revise_node(state: PipelineState) -> dict:
        prompt = (
            f"Draft:\n{state['draft']}\n\nCritique: {state['critique_reason']}\n\n{state['context_text']}"
        )
        new_draft = llm.complete(REVISE_SYSTEM, prompt)
        return {
            "draft": new_draft,
            "revise_count": state.get("revise_count", 0) + 1,
            "events": [_event("revise", "Revised the draft to address the critique")],
        }

    def finalize_node(state: PipelineState) -> dict:
        return {
            "final_answer": state["draft"],
            "events": [_event("finalize", "Finalized answer and persisted the memory graph")],
        }

    def route_after_critique(state: PipelineState) -> str:
        if state.get("verdict") == "REVISE" and state.get("revise_count", 0) < settings.max_revise_loops:
            return "revise"
        return "finalize"

    graph = StateGraph(PipelineState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("reconcile_memory", reconcile_node)
    graph.add_node("propose", propose_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "reconcile_memory")
    graph.add_edge("reconcile_memory", "propose")
    graph.add_edge("propose", "critique")
    graph.add_conditional_edges("critique", route_after_critique, {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "critique")
    graph.add_edge("finalize", END)

    return graph.compile()
