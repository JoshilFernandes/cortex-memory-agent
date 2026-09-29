"""Pure unit tests for the contradiction/reconciliation logic — no LLM, no I/O."""
from cortex.contradiction import reconcile_facts
from cortex.models import Fact


def make_fact(subject, relation, obj, **kw) -> Fact:
    return Fact(
        subject=subject,
        relation=relation,
        object=obj,
        confidence=kw.get("confidence", 0.9),
        source_url=kw.get("source_url", "https://example.test/a"),
        source_title=kw.get("source_title", "Test source"),
        observed_at=kw.get("observed_at", "2026-01-01"),
    )


def test_singular_relation_contradiction_supersedes_old_fact():
    old = make_fact("Arcline AI", "ceo", "Dana Whitfield")
    new = make_fact("Arcline AI", "ceo", "Priya Kapoor")

    result = reconcile_facts([new], existing_active_facts=[old])

    assert len(result.contradictions) == 1
    assert result.contradictions[0].old_fact is old
    assert result.contradictions[0].new_fact is new
    assert result.facts_to_supersede == [(old, new)]
    assert result.facts_to_add == [new]


def test_cumulative_relation_does_not_contradict():
    old = make_fact("Meridian Systems", "raised", "$40M Series B")
    new = make_fact("Meridian Systems", "raised", "$75M Series C")

    result = reconcile_facts([new], existing_active_facts=[old])

    assert result.contradictions == []
    assert result.facts_to_supersede == []
    assert result.facts_to_add == [new]


def test_exact_duplicate_is_skipped_not_added_twice():
    old = make_fact("Meridian Systems", "raised", "$40M Series B")
    duplicate = make_fact("Meridian Systems", "raised", "$40M series b.")  # case/punct only

    result = reconcile_facts([duplicate], existing_active_facts=[old])

    assert result.duplicates_skipped == 1
    assert result.facts_to_add == []
    assert result.contradictions == []


def test_multiple_new_facts_can_mix_contradiction_and_addition():
    old_ceo = make_fact("Arcline AI", "ceo", "Dana Whitfield")
    old_funding = make_fact("Meridian Systems", "raised", "$40M Series B")
    new_ceo = make_fact("Arcline AI", "ceo", "Priya Kapoor")
    new_funding = make_fact("Meridian Systems", "raised", "$75M Series C")
    brand_new = make_fact("Meridian Systems", "acquired", "Fenwick Labs")

    result = reconcile_facts(
        [new_ceo, new_funding, brand_new],
        existing_active_facts=[old_ceo, old_funding],
    )

    assert len(result.contradictions) == 1
    assert result.contradictions[0].reason.startswith("'Arcline AI' ceo changed")
    assert (old_ceo, new_ceo) in result.facts_to_supersede
    assert brand_new in result.facts_to_add
    assert new_funding in result.facts_to_add
