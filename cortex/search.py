"""
Web search abstraction.

- OfflineFixtureSearchTool: returns the deterministic corpus in
  cortex/demo_corpus.py, keyed by an explicit "epoch" (simulating research
  done at two different points in time). Zero network, zero flakiness — used
  by the test suite, CI, and the offline demo.
- LiveDuckDuckGoSearchTool: real web search via the `ddgs` package, no API
  key required. Used for the live, "wow, it actually researched the real
  web" demo recording.
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from cortex.demo_corpus import Doc, corpus_for_epoch


@dataclass
class SearchResult:
    title: str
    url: str
    published: str
    snippet: str


class SearchTool(ABC):
    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        ...


class OfflineFixtureSearchTool(SearchTool):
    """Deterministic, offline. `epoch` simulates "when" the research ran."""

    def __init__(self, epoch: int = 1) -> None:
        self.epoch = epoch

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        docs: list[Doc] = corpus_for_epoch(self.epoch)
        if not docs:
            return []
        # Deterministically rotate the start point by the query so that
        # parallel researchers working different sub-queries tend to land on
        # different documents first, instead of all three fetching doc #0.
        offset = zlib.crc32(query.encode("utf-8")) % len(docs)
        rotated = docs[offset:] + docs[:offset]
        return [
            SearchResult(title=d.title, url=d.url, published=d.published, snippet=d.text)
            for d in rotated[:k]
        ]


class LiveDuckDuckGoSearchTool(SearchTool):
    def __init__(self) -> None:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SEARCH_BACKEND=live requires the `ddgs` package: pip install ddgs"
            ) from exc
        self._ddgs_cls = DDGS

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        with self._ddgs_cls() as ddgs:
            hits = list(ddgs.text(query, max_results=k))
        return [
            SearchResult(
                title=h.get("title", ""),
                url=h.get("href", h.get("link", "")),
                published="",
                snippet=h.get("body", h.get("snippet", "")),
            )
            for h in hits
        ]


def get_search_tool(backend: str, epoch: int = 1) -> SearchTool:
    if backend == "live":
        return LiveDuckDuckGoSearchTool()
    return OfflineFixtureSearchTool(epoch=epoch)
