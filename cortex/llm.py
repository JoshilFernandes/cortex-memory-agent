"""
LLM client abstraction.

Cortex never hard-codes a single LLM provider. The default "mock" backend is
fully deterministic (no network, no API key) so the whole pipeline, the test
suite and CI run for free and offline. Set LLM_BACKEND=groq and GROQ_API_KEY
to use a real model.
"""
from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from typing import Any

from cortex.config import settings
from cortex.demo_corpus import EPOCH_1, EPOCH_2

# A tiny curated lookup table so the zero-dependency mock backend extracts
# *clean* entities for the bundled demo corpus (real extraction quality with
# LLM_BACKEND=groq does not need this — see the regex fallback below for
# arbitrary text).
_KNOWN_EXTRACTIONS: dict[str, list[dict]] = {
    EPOCH_1[0].text: [
        {"subject": "AgentBench", "relation": "leader", "object": "Arcline AI", "confidence": 0.9},
        {"subject": "Arcline AI", "relation": "ceo", "object": "Dana Whitfield", "confidence": 0.9},
    ],
    EPOCH_1[1].text: [
        {"subject": "Meridian Systems", "relation": "raised", "object": "$40M Series B", "confidence": 0.9},
        {"subject": "Meridian Systems", "relation": "released", "object": "Pathwise", "confidence": 0.9},
    ],
    EPOCH_2[0].text: [
        {"subject": "AgentBench", "relation": "leader", "object": "Meridian Systems", "confidence": 0.9},
        {"subject": "Arcline AI", "relation": "ceo", "object": "Priya Kapoor", "confidence": 0.9},
    ],
    EPOCH_2[1].text: [
        {"subject": "Meridian Systems", "relation": "raised", "object": "$75M Series C", "confidence": 0.9},
        {"subject": "Meridian Systems", "relation": "acquired", "object": "Fenwick Labs", "confidence": 0.9},
    ],
}


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        """Return the model's text completion for `prompt` given `system`."""


class MockLLMClient(LLMClient):
    """
    A deterministic, template-driven "LLM" used as the zero-dependency
    default. It is intentionally simple: it extracts (subject, relation,
    object) style facts out of plain sentences with light heuristics, and
    otherwise stitches retrieved context into a templated answer. This keeps
    the *pipeline* (planning -> parallel research -> propose -> critique ->
    revise -> memory write) fully real and testable without ever calling a
    paid API.
    """

    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        if json_mode:
            return self._structured(system, prompt)
        return self._freeform(system, prompt)

    # -- internals ---------------------------------------------------
    def _freeform(self, system: str, prompt: str) -> str:
        if "Write a concise, well-cited answer" in system:
            return self._propose_answer(prompt)
        if "revision node" in system.lower():
            return self._revise_answer(prompt)
        return f"[mock-llm] {prompt[:200]}"

    def _structured(self, system: str, prompt: str) -> str:
        if "decompose" in system.lower():
            return self._plan(prompt)
        if "extract" in system.lower():
            return self._extract_triples(prompt)
        if "critique" in system.lower():
            return self._critique(prompt)
        return json.dumps({"result": "mock"})

    def _plan(self, prompt: str) -> str:
        topic = _first_line(prompt)
        subqueries = [
            f"{topic} — current state and key facts",
            f"{topic} — recent changes or updates",
            f"{topic} — who are the main parties involved",
        ][: settings.researcher_fanout]
        return json.dumps({"subqueries": subqueries})

    def _extract_triples(self, prompt: str) -> str:
        known = _KNOWN_EXTRACTIONS.get(prompt.strip())
        if known is not None:
            return json.dumps({"triples": known})

        # Generic fallback heuristic for arbitrary text (unit tests, or text
        # outside the bundled demo corpus): "X <verb> Y" -> (X, relation, Y).
        # This is deliberately simple — it exists so the mock backend never
        # *crashes* on unfamiliar text, not to be a real NLP extractor.
        triples = []
        pattern = re.compile(
            r"([A-Z][\w& -]{1,40}?)\s+(is|are|leads|acquired|uses|raised|released|"
            r"replaced|supports|founded|owns)\s+([A-Z0-9][\w& %-]{1,40}?)(?:[.,]|$)",
        )
        for match in pattern.finditer(prompt):
            subject, relation, obj = (g.strip() for g in match.groups())
            triples.append(
                {
                    "subject": subject,
                    "relation": relation.replace(" ", "_"),
                    "object": obj,
                    "confidence": 0.6,
                }
            )
        return json.dumps({"triples": triples})

    def _propose_answer(self, prompt: str) -> str:
        facts = _extract_facts_section(prompt)
        intro = "Based on the current memory graph and freshly researched sources:"
        return intro + ("\n\n" + facts if facts else "")

    def _revise_answer(self, prompt: str) -> str:
        critique = ""
        if "Critique:" in prompt:
            critique = prompt.split("Critique:", 1)[1].split("\n")[0].strip()
        facts = _extract_facts_section(prompt)
        header = "Updated answer — memory changed since the last research pass."
        if critique:
            header += f" ({critique})"
        return header + ("\n\n" + facts if facts else "")

    def _critique(self, prompt: str) -> str:
        if "CONTRADICTION" in prompt:
            return json.dumps(
                {
                    "verdict": "REVISE",
                    "reason": "A newly researched fact contradicts an existing memory entry.",
                }
            )
        return json.dumps({"verdict": "ACCEPT", "reason": "Draft is consistent with memory."})


class GroqLLMClient(LLMClient):
    def __init__(self) -> None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LLM_BACKEND=groq requires the `groq` package: pip install groq"
            ) from exc
        if not settings.groq_api_key:
            raise RuntimeError("LLM_BACKEND=groq requires GROQ_API_KEY to be set")
        self._client = Groq(api_key=settings.groq_api_key)

    def complete(self, system: str, prompt: str, *, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            **kwargs,
        )
        return response.choices[0].message.content or ""


def _extract_facts_section(text: str) -> str:
    marker = "Known graph facts"
    if marker not in text:
        return ""
    # Use the LAST occurrence: a revise prompt embeds the previous draft
    # (which already contains one facts section) followed by the fresh
    # retrieval context — the fresh one is what we want to surface.
    start = text.rindex(marker)
    end = text.find("Relevant source passages", start)
    section = text[start : end if end != -1 else start + 1200]
    return section.strip()


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:120] if text.strip() else "the topic"


_client_cache: dict[str, LLMClient] = {}


def get_llm_client() -> LLMClient:
    backend = settings.llm_backend
    if backend not in _client_cache:
        if backend == "groq":
            _client_cache[backend] = GroqLLMClient()
        else:
            _client_cache[backend] = MockLLMClient()
    return _client_cache[backend]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
