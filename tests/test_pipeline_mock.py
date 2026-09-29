"""
End-to-end pipeline tests using the zero-dependency mock LLM + offline
fixture corpus (see cortex/demo_corpus.py) — no network, no API keys.
"""
from cortex import config, engine

QUERY = "Who leads AgentBench and who runs Arcline AI?"


def _run_to_done(cortex, query, epoch):
    events = list(cortex.run(query, epoch=epoch))
    steps = [e for e in events if e["type"] == "step"]
    done = next(e for e in events if e["type"] == "done")
    return steps, done


def test_first_pass_finds_no_contradictions(cortex_instance):
    steps, done = _run_to_done(cortex_instance, QUERY, epoch=1)

    node_order = [s["node"] for s in steps]
    assert node_order == ["plan", "research", "reconcile_memory", "propose", "critique", "finalize"]
    assert done["contradiction_count"] == 0
    assert done["revise_count"] == 0
    assert "AgentBench" in done["final_answer"]


def test_second_pass_detects_contradictions_and_revises(cortex_instance):
    list(cortex_instance.run(QUERY, epoch=1))
    steps, done = _run_to_done(cortex_instance, QUERY, epoch=2)

    assert done["contradiction_count"] == 2  # benchmark leader + CEO both changed
    assert done["revise_count"] >= 1

    reconcile_step = next(s for s in steps if s["node"] == "reconcile_memory")
    assert any("AgentBench" in reason for reason in reconcile_step["contradictions"])
    assert any("Arcline AI" in reason for reason in reconcile_step["contradictions"])


def test_history_retains_superseded_facts_with_full_temporal_trail(cortex_instance):
    list(cortex_instance.run(QUERY, epoch=1))
    list(cortex_instance.run(QUERY, epoch=2))

    history = cortex_instance.history("AgentBench", "leader")
    assert [f.status for f in history] == ["superseded", "active"]
    assert history[0].object == "Arcline AI"
    assert history[1].object == "Meridian Systems"
    assert history[0].superseded_by == history[1].id


def test_cumulative_facts_do_not_get_superseded(cortex_instance):
    list(cortex_instance.run(QUERY, epoch=1))
    list(cortex_instance.run(QUERY, epoch=2))

    funding_history = cortex_instance.history("Meridian Systems", "raised")
    assert len(funding_history) == 2
    assert all(f.status == "active" for f in funding_history)


def test_memory_persists_across_a_process_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "graph_snapshot_path", str(tmp_path / "graph.json"))
    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(config.settings, "chroma_collection", "restart_test")

    engine._instance = None
    first = engine.Cortex()
    list(first.run(QUERY, epoch=1))
    first.close()
    engine._instance = None

    # Simulate a fresh process: brand new Cortex instance, same on-disk paths.
    second = engine.Cortex()
    try:
        active = second.snapshot()
        assert active["fact_count"] == 4
        history = second.history("AgentBench", "leader")
        assert len(history) == 1
        assert history[0].object == "Arcline AI"
    finally:
        second.close()
        engine._instance = None
