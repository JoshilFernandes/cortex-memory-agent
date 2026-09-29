"""
A small scripted eval suite, in the same spirit as the eval harnesses in the
other repos in this portfolio: a handful of named scenarios with explicit
pass/fail criteria, runnable standalone for a human-readable report, and
gated in CI as an ordinary pytest test.

    python -m tests.test_eval_scenarios
"""
from __future__ import annotations

from dataclasses import dataclass

from cortex import config, engine

QUERY = "Who leads AgentBench and who runs Arcline AI?"


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str


def _fresh_cortex(tmp_path) -> engine.Cortex:
    config.settings.graph_snapshot_path = str(tmp_path / "graph.json")
    config.settings.chroma_persist_dir = str(tmp_path / "chroma")
    config.settings.chroma_collection = "eval_chunks"
    engine._instance = None
    return engine.Cortex()


def run_all_scenarios(tmp_path) -> list[ScenarioResult]:
    cortex = _fresh_cortex(tmp_path)
    results: list[ScenarioResult] = []
    try:
        epoch1_events = list(cortex.run(QUERY, epoch=1))
        epoch1_done = next(e for e in epoch1_events if e["type"] == "done")
        results.append(
            ScenarioResult(
                "cold_start_has_no_contradictions",
                epoch1_done["contradiction_count"] == 0,
                f"contradiction_count={epoch1_done['contradiction_count']}",
            )
        )

        epoch2_events = list(cortex.run(QUERY, epoch=2))
        epoch2_done = next(e for e in epoch2_events if e["type"] == "done")
        results.append(
            ScenarioResult(
                "re_research_detects_both_known_contradictions",
                epoch2_done["contradiction_count"] == 2,
                f"contradiction_count={epoch2_done['contradiction_count']}",
            )
        )
        results.append(
            ScenarioResult(
                "critique_loop_forces_at_least_one_revision",
                epoch2_done["revise_count"] >= 1,
                f"revise_count={epoch2_done['revise_count']}",
            )
        )

        leader_history = cortex.history("AgentBench", "leader")
        results.append(
            ScenarioResult(
                "superseded_fact_keeps_full_temporal_history",
                [f.status for f in leader_history] == ["superseded", "active"],
                f"history={[(f.object, f.status) for f in leader_history]}",
            )
        )

        funding_history = cortex.history("Meridian Systems", "raised")
        results.append(
            ScenarioResult(
                "cumulative_relation_keeps_all_facts_active",
                len(funding_history) == 2 and all(f.status == "active" for f in funding_history),
                f"history={[(f.object, f.status) for f in funding_history]}",
            )
        )

        snapshot = cortex.snapshot()
        node_labels = {n["label"] for n in snapshot["nodes"]}
        results.append(
            ScenarioResult(
                "graph_grows_with_genuinely_new_entities",
                "Fenwick Labs" in node_labels,
                f"node_count={len(node_labels)}",
            )
        )
    finally:
        cortex.close()
        engine._instance = None

    return results


def test_eval_suite_passes(tmp_path):
    results = run_all_scenarios(tmp_path)
    failures = [r for r in results if not r.passed]
    assert not failures, f"eval scenarios failed: {failures}"


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        outcomes = run_all_scenarios(Path(tmp))

    width = max(len(o.name) for o in outcomes)
    print(f"\nCortex eval suite — {len(outcomes)} scenarios\n" + "-" * 60)
    all_passed = True
    for o in outcomes:
        mark = "PASS" if o.passed else "FAIL"
        all_passed &= o.passed
        print(f"[{mark}] {o.name.ljust(width)}  {o.detail}")
    print("-" * 60)
    print("ALL SCENARIOS PASSED" if all_passed else "SOME SCENARIOS FAILED")
    sys.exit(0 if all_passed else 1)
