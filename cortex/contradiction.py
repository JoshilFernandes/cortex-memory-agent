"""
Temporal contradiction detection & resolution.

This is the heart of Cortex's "never silently forgets, never silently
contradicts itself" behaviour. Given freshly extracted facts and the set of
currently-active facts already in memory:

  * SINGULAR relation + same (subject, relation) + a DIFFERENT object
    -> contradiction. The old fact is marked `superseded` (never deleted —
       its full history stays queryable) and the new fact becomes active.
  * CUMULATIVE relation -> the new fact is simply added; prior facts for the
    same (subject, relation) stay active (a company can raise many rounds).
  * An identical fact seen again -> treated as reinforcement, not a new row.
"""
from __future__ import annotations

from dataclasses import dataclass

from cortex.models import Fact, normalize_entity, relation_is_singular


@dataclass
class ContradictionEvent:
    old_fact: Fact
    new_fact: Fact
    reason: str


@dataclass
class ReconciliationResult:
    facts_to_add: list[Fact]
    facts_to_supersede: list[tuple[Fact, Fact]]  # (old, superseded_by=new)
    contradictions: list[ContradictionEvent]
    duplicates_skipped: int


def reconcile_facts(
    new_facts: list[Fact],
    existing_active_facts: list[Fact],
) -> ReconciliationResult:
    by_key: dict[tuple[str, str], list[Fact]] = {}
    for f in existing_active_facts:
        by_key.setdefault(f.key(), []).append(f)

    facts_to_add: list[Fact] = []
    facts_to_supersede: list[tuple[Fact, Fact]] = []
    contradictions: list[ContradictionEvent] = []
    duplicates_skipped = 0

    for new_fact in new_facts:
        key = new_fact.key()
        current = by_key.get(key, [])

        exact_dupe = next(
            (f for f in current if _norm(f.object) == _norm(new_fact.object)),
            None,
        )
        if exact_dupe is not None:
            duplicates_skipped += 1
            continue

        if relation_is_singular(new_fact.relation) and current:
            for old_fact in current:
                facts_to_supersede.append((old_fact, new_fact))
                contradictions.append(
                    ContradictionEvent(
                        old_fact=old_fact,
                        new_fact=new_fact,
                        reason=(
                            f"'{normalize_entity(new_fact.subject)}' {new_fact.relation} "
                            f"changed from '{old_fact.object}' to '{new_fact.object}'"
                        ),
                    )
                )
            # the new value becomes the sole active fact for this key
            by_key[key] = [new_fact]
        else:
            by_key.setdefault(key, []).append(new_fact)

        facts_to_add.append(new_fact)

    return ReconciliationResult(
        facts_to_add=facts_to_add,
        facts_to_supersede=facts_to_supersede,
        contradictions=contradictions,
        duplicates_skipped=duplicates_skipped,
    )


def _norm(text: str) -> str:
    return " ".join(text.lower().strip().rstrip(".").split())
