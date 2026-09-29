"""Shared data model: the atomic unit of memory is a Fact (a graph triple)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

FactStatus = Literal["active", "superseded"]

# Relations that can only hold ONE value at a time for a given subject
# (a company has exactly one current CEO, a benchmark has exactly one current
# leader). A new observation for one of these relations *contradicts* the
# existing active fact and supersedes it.
#
# Everything not listed here is treated as CUMULATIVE: new observations are
# additional history (a company can raise several funding rounds, acquire
# several companies, release several products) and do not contradict prior
# facts. This mirrors the "functional vs. non-functional property" distinction
# from knowledge-graph ontology design (e.g. OWL's owl:FunctionalProperty).
SINGULAR_RELATIONS: set[str] = {"is", "leads", "leader", "ceo", "owns"}


def relation_is_singular(relation: str) -> bool:
    return relation in SINGULAR_RELATIONS


def normalize_entity(name: str) -> str:
    return " ".join(name.strip().split())


@dataclass
class Fact:
    subject: str
    relation: str
    object: str
    confidence: float
    source_url: str
    source_title: str
    observed_at: str  # publication date of the source, e.g. "2026-09-14"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: FactStatus = "active"
    superseded_by: str | None = None
    created_at: float = field(default_factory=time.time)

    def key(self) -> tuple[str, str]:
        return (normalize_entity(self.subject), self.relation)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "observed_at": self.observed_at,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            subject=d["subject"],
            relation=d["relation"],
            object=d["object"],
            confidence=d["confidence"],
            source_url=d["source_url"],
            source_title=d["source_title"],
            observed_at=d["observed_at"],
            id=d.get("id", uuid.uuid4().hex[:12]),
            status=d.get("status", "active"),
            superseded_by=d.get("superseded_by"),
            created_at=d.get("created_at", time.time()),
        )
