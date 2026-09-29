"""
A small, hand-written, time-versioned fixture corpus.

This exists so the whole system — pipeline, contradiction detection,
temporal memory, the test suite and CI — works deterministically with zero
network access and zero API keys. It also makes the "memory that persists
and catches its own contradictions" story reproducible for anyone cloning
the repo, instead of depending on whatever a live web search happens to
return on a given day.

The scenario: two research passes ("epochs") over the same beat — the
competitive landscape of (fictional) AI agent-orchestration startups. Epoch 2
updates, corrects and extends what epoch 1 found, which is exactly the
situation a long-lived research agent has to handle gracefully:

  * a *functional* fact changes value       -> contradiction, old fact superseded
  * a *cumulative* fact gets a new instance  -> both kept, no contradiction
  * a brand new fact appears                -> the graph simply grows

All company/people names below are fictional.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Doc:
    title: str
    url: str
    published: str
    text: str


EPOCH_1: list[Doc] = [
    Doc(
        title="AgentBench Q1 leaderboard update",
        url="https://example-news.test/agentbench-q1",
        published="2026-03-10",
        text=(
            "AgentBench is led by Arcline AI with a 91 percent completion rate "
            "across the long-horizon tool-use suite. Arcline AI is led by CEO "
            "Dana Whitfield."
        ),
    ),
    Doc(
        title="Meridian Systems closes new funding round",
        url="https://example-news.test/meridian-series-b",
        published="2026-02-18",
        text=(
            "Meridian Systems raised 40 million dollars in its Series B round. "
            "Meridian Systems released Pathwise, an orchestration layer for "
            "long-running agents."
        ),
    ),
]

EPOCH_2: list[Doc] = [
    Doc(
        title="AgentBench Q3 leaderboard update",
        url="https://example-news.test/agentbench-q3",
        published="2026-09-14",
        text=(
            "AgentBench is led by Meridian Systems with a 94 percent completion "
            "rate, overtaking the previous leader. Arcline AI is led by CEO "
            "Priya Kapoor after a leadership change earlier this year."
        ),
    ),
    Doc(
        title="Meridian Systems raises Series C, acquires Fenwick Labs",
        url="https://example-news.test/meridian-series-c",
        published="2026-09-20",
        text=(
            "Meridian Systems raised 75 million dollars in its Series C round. "
            "Meridian Systems acquired Fenwick Labs to expand its evaluation "
            "tooling."
        ),
    ),
]


def corpus_for_epoch(epoch: int) -> list[Doc]:
    return EPOCH_1 if epoch <= 1 else EPOCH_2


DEMO_TOPIC = "the competitive landscape of AI agent-orchestration startups"
