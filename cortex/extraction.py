"""Turn raw research text into structured (subject, relation, object) Facts."""
from __future__ import annotations

import json

from cortex.llm import LLMClient
from cortex.models import Fact

EXTRACTION_SYSTEM_PROMPT = """You extract structured knowledge-graph triples.
Read the passage and extract every clear (subject, relation, object) fact you
can find. Respond ONLY as JSON: {"triples": [{"subject": ..., "relation": ...,
"object": ..., "confidence": 0-1}]}. Use short snake_case relations."""


def extract_facts(
    llm: LLMClient,
    *,
    text: str,
    source_url: str,
    source_title: str,
    observed_at: str,
) -> list[Fact]:
    raw = llm.complete(EXTRACTION_SYSTEM_PROMPT, text, json_mode=True)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    facts: list[Fact] = []
    for triple in payload.get("triples", []):
        subject = str(triple.get("subject", "")).strip()
        relation = str(triple.get("relation", "")).strip()
        obj = str(triple.get("object", "")).strip()
        if not subject or not relation or not obj:
            continue
        facts.append(
            Fact(
                subject=subject,
                relation=relation,
                object=obj,
                confidence=float(triple.get("confidence", 0.6)),
                source_url=source_url,
                source_title=source_title,
                observed_at=observed_at,
            )
        )
    return facts
