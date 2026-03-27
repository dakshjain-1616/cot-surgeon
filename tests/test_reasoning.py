"""
tests/test_reasoning.py — pytest suite for CoT Surgeon.

Covers:
  - ReasoningNode creation, serialization, deserialization
  - ReasoningGraph operations (CRUD, Mermaid, serialization)
  - ReasoningEngine: generate, parse, edit, recalculate
  - Demo prompts JSON file validity
  - Integration flows matching the test spec
"""

import os
import json
import pytest
import sys

# Ensure the project root is on sys.path when running from tests/ subdir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cot_surgeon import (
    ReasoningEngine,
    ReasoningGraph,
    ReasoningNode,
    NodeType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sky_graph():
    """Pre-built 3-node graph for 'why is the sky blue'."""
    nodes = [
        ReasoningNode(
            id="node_1",
            label="Fact",
            content="Sunlight contains all visible wavelengths (400–700 nm).",
            node_type=NodeType.FACT,
        ),
        ReasoningNode(
            id="node_2",
            label="Physics",
            content="Rayleigh scattering causes blue light (~450 nm) to scatter ~10× more than red.",
            node_type=NodeType.REASONING,
            dependencies=["node_1"],
        ),
        ReasoningNode(
            id="node_3",
            label="Conclusion",
            content="The sky appears blue because blue light scatters in all directions.",
            node_type=NodeType.CONCLUSION,
            dependencies=["node_2"],
        ),
    ]
    return ReasoningGraph(nodes=nodes, prompt="Explain why the sky is blue")


@pytest.fixture
def mock_engine():
    return ReasoningEngine(mode="mock")


# ── ReasoningNode ─────────────────────────────────────────────────────────────

class TestReasoningNode:
    def test_creation_defaults(self):
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT)
        assert n.id == "n1"
        assert n.label == "Fact"
        assert n.edited is False
        assert n.dependencies == []

    def test_serialization_round_trip(self):
        n = ReasoningNode(
            id="n1", label="Physics", content="Rayleigh", node_type=NodeType.REASONING,
            dependencies=["n0"], edited=True,
        )
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["node_type"] == "reasoning"
        assert d["edited"] is True
        assert d["dependencies"] == ["n0"]

        restored = ReasoningNode.from_dict(d)
        assert restored.id == n.id
        assert restored.node_type == NodeType.REASONING
        assert restored.edited is True

    def test_node_type_enum_values(self):
        assert NodeType.FACT.value == "fact"
        assert NodeType.REASONING.value == "reasoning"
        assert NodeType.CONCLUSION.value == "conclusion"


# ── ReasoningGraph ────────────────────────────────────────────────────────────

class TestReasoningGraph:
    def test_has_three_nodes(self, sky_graph):
        assert len(sky_graph.nodes) == 3

    def test_get_node_by_id(self, sky_graph):
        node = sky_graph.get_node("node_2")
        assert node is not None
        assert node.label == "Physics"

    def test_get_node_missing(self, sky_graph):
        assert sky_graph.get_node("nonexistent") is None

    def test_get_node_index(self, sky_graph):
        assert sky_graph.get_node_index("node_1") == 0
        assert sky_graph.get_node_index("node_3") == 2
        assert sky_graph.get_node_index("nope") == -1

    def test_update_node_marks_edited(self, sky_graph):
        sky_graph.update_node("node_2", "Because of clouds")
        node = sky_graph.get_node("node_2")
        assert node.content == "Because of clouds"
        assert node.edited is True

    def test_get_conclusion_returns_last_conclusion(self, sky_graph):
        c = sky_graph.get_conclusion()
        assert "blue" in c.lower() or "scatter" in c.lower()

    def test_get_conclusion_on_empty_gives_none(self):
        g = ReasoningGraph(nodes=[], prompt="test")
        assert g.get_conclusion() is None

    def test_mermaid_contains_graph_td(self, sky_graph):
        m = sky_graph.to_mermaid()
        assert "graph TD" in m

    def test_mermaid_contains_all_node_ids(self, sky_graph):
        m = sky_graph.to_mermaid()
        assert "node_1" in m
        assert "node_2" in m
        assert "node_3" in m

    def test_mermaid_contains_edges(self, sky_graph):
        m = sky_graph.to_mermaid()
        assert "-->" in m

    def test_mermaid_selected_node_highlight(self, sky_graph):
        m = sky_graph.to_mermaid(selected_node_id="node_2")
        assert "FF5722" in m   # selected colour

    def test_mermaid_edited_node_colour(self, sky_graph):
        sky_graph.update_node("node_2", "edited content")
        m = sky_graph.to_mermaid()
        assert "9C27B0" in m   # edited colour

    def test_serialization_round_trip(self, sky_graph):
        d = sky_graph.to_dict()
        assert d["prompt"] == "Explain why the sky is blue"
        assert len(d["nodes"]) == 3
        assert d["version"] == 1

        restored = ReasoningGraph.from_dict(d)
        assert len(restored.nodes) == 3
        assert restored.prompt == sky_graph.prompt
        assert restored.nodes[1].label == "Physics"

    def test_version_defaults_to_one(self, sky_graph):
        assert sky_graph.version == 1


# ── ReasoningEngine — mode detection ─────────────────────────────────────────

class TestEngineModeDetection:
    def test_explicit_mock_mode(self):
        e = ReasoningEngine(mode="mock")
        assert e.mode == "mock"

    def test_auto_falls_to_mock_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        e = ReasoningEngine(mode="auto")
        assert e.mode == "mock"

    def test_local_mode_env_var(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        e = ReasoningEngine(mode="auto")
        assert e.mode == "local"
        monkeypatch.delenv("LOCAL_MODE")

    def test_openrouter_mode_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        e = ReasoningEngine(mode="openrouter")
        assert e.mode == "openrouter"


# ── ReasoningEngine — generate_cot (Test Spec #1) ────────────────────────────

class TestGenerateCOT:
    def test_sky_generates_exactly_three_nodes(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert len(graph.nodes) == 3

    def test_sky_has_fact_node(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        labels = [n.label for n in graph.nodes]
        assert "Fact" in labels

    def test_sky_has_physics_node(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        labels = [n.label for n in graph.nodes]
        assert "Physics" in labels

    def test_sky_has_conclusion_node(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        labels = [n.label for n in graph.nodes]
        assert "Conclusion" in labels

    def test_sky_node_types_correct(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        types = [n.node_type for n in graph.nodes]
        assert NodeType.FACT in types
        assert NodeType.REASONING in types
        assert NodeType.CONCLUSION in types

    def test_prompt_preserved_on_graph(self, mock_engine):
        prompt = "Explain why the sky is blue"
        graph = mock_engine.generate_cot(prompt)
        assert graph.prompt == prompt

    def test_node_chain_dependencies(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert graph.nodes[1].dependencies == ["node_1"]
        assert graph.nodes[2].dependencies == ["node_2"]

    def test_math_prompt_generates_three_nodes(self, mock_engine):
        graph = mock_engine.generate_cot("What is 15% of 240?")
        assert len(graph.nodes) == 3

    def test_math_nodes_have_correct_types(self, mock_engine):
        graph = mock_engine.generate_cot("What is 15% of 240?")
        assert graph.nodes[0].node_type == NodeType.FACT
        assert graph.nodes[-1].node_type == NodeType.CONCLUSION

    def test_generic_prompt_generates_nodes(self, mock_engine):
        graph = mock_engine.generate_cot("What is artificial intelligence?")
        assert len(graph.nodes) >= 1
        assert graph.get_conclusion() is not None


# ── ReasoningEngine — recalculate (Test Spec #2) ─────────────────────────────

class TestRecalculate:
    def test_edit_physics_node_updates_conclusion(self, mock_engine):
        """Test Spec #2: Edit [Physics] → 'Because of clouds' → conclusion updates."""
        graph = mock_engine.generate_cot("Explain why the sky is blue")

        # Find Physics node
        physics = next((n for n in graph.nodes if n.label == "Physics"), None)
        assert physics is not None, "Physics node must exist"

        original_conclusion = graph.get_conclusion()
        graph.update_node(physics.id, "Because of clouds")
        assert graph.get_node(physics.id).content == "Because of clouds"

        graph = mock_engine.recalculate_from_node(graph, physics.id)
        new_conclusion = graph.get_conclusion()

        # Conclusion should reference clouds or be demonstrably different
        assert "cloud" in new_conclusion.lower() or new_conclusion != original_conclusion

    def test_recalculate_increments_version(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert graph.version == 1
        graph.update_node("node_2", "edited")
        graph = mock_engine.recalculate_from_node(graph, "node_2")
        assert graph.version == 2

    def test_recalculate_preserves_node_count(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_2", "new reasoning")
        graph = mock_engine.recalculate_from_node(graph, "node_2")
        assert len(graph.nodes) == 3

    def test_recalculate_preserves_kept_nodes(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        original_fact = graph.get_node("node_1").content
        graph.update_node("node_2", "because of clouds")
        graph = mock_engine.recalculate_from_node(graph, "node_2")
        # node_1 (Fact) must be unchanged
        assert graph.get_node("node_1").content == original_fact

    def test_recalculate_preserves_edited_flag(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_2", "because of clouds")
        graph = mock_engine.recalculate_from_node(graph, "node_2")
        assert graph.get_node("node_2").edited is True

    def test_recalculate_on_last_node_only_bumps_version(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_3", "custom conclusion")
        graph = mock_engine.recalculate_from_node(graph, "node_3")
        # Nothing downstream — version still increments
        assert graph.version == 2
        assert graph.get_node("node_3").content == "custom conclusion"

    def test_recalculate_invalid_id_is_noop(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        v_before = graph.version
        graph = mock_engine.recalculate_from_node(graph, "does_not_exist")
        assert graph.version == v_before


# ── Demo prompts JSON ─────────────────────────────────────────────────────────

class TestDemoPromptsFile:
    def test_file_exists(self):
        assert os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_prompts.json")
        )

    def test_valid_json_list(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_prompts.json")
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) >= 5

    def test_each_entry_has_label_and_prompt(self):
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_prompts.json")
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            assert "label" in entry, f"Missing 'label' in {entry}"
            assert "prompt" in entry, f"Missing 'prompt' in {entry}"
            assert isinstance(entry["label"], str)
            assert isinstance(entry["prompt"], str)
            assert len(entry["prompt"]) > 0


# ── Integration: local flag (Test Spec #3) ───────────────────────────────────

class TestLocalModeFlag:
    def test_local_mode_initialises_without_crash(self, monkeypatch):
        """Test Spec #3: --local flag sets mode to 'local' without needing an API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("LLAMA_MODEL_PATH", raising=False)
        # Local mode without a model path falls back to mock for generation
        e = ReasoningEngine(mode="local")
        assert e.mode == "local"
        # Without LLAMA_MODEL_PATH it degrades gracefully to mock output
        graph = e.generate_cot("Why is the sky blue?")
        assert len(graph.nodes) >= 1

    def test_local_mode_env_var_true(self, monkeypatch):
        monkeypatch.setenv("LOCAL_MODE", "true")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        e = ReasoningEngine(mode="auto")
        assert e.mode == "local"


# ── Node confidence scores ────────────────────────────────────────────────────

class TestNodeConfidence:
    def test_default_confidence_is_one(self):
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT)
        assert n.confidence == 1.0

    def test_confidence_in_serialization(self):
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT, confidence=0.85)
        d = n.to_dict()
        assert d["confidence"] == 0.85
        restored = ReasoningNode.from_dict(d)
        assert restored.confidence == 0.85

    def test_confidence_backward_compat(self):
        """Nodes deserialized without confidence field default to 1.0."""
        d = {"id": "n1", "label": "Fact", "content": "hi", "node_type": "fact"}
        n = ReasoningNode.from_dict(d)
        assert n.confidence == 1.0

    def test_mock_sky_confidence_in_range(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        for node in graph.nodes:
            assert 0.0 <= node.confidence <= 1.0

    def test_mock_sky_has_realistic_confidence(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        # Sky mock template has high confidence (>0.9)
        assert all(n.confidence > 0.9 for n in graph.nodes)

    def test_low_confidence_nodes_detected(self, sky_graph):
        sky_graph.nodes[0].confidence = 0.5
        low = sky_graph.low_confidence_nodes(threshold=0.7)
        assert len(low) == 1
        assert low[0].id == "node_1"

    def test_low_confidence_none_below_threshold(self, sky_graph):
        for n in sky_graph.nodes:
            n.confidence = 0.95
        low = sky_graph.low_confidence_nodes(threshold=0.7)
        assert len(low) == 0

    def test_low_confidence_threshold_env_var(self, sky_graph, monkeypatch):
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.99")
        sky_graph.nodes[0].confidence = 0.95
        low = sky_graph.low_confidence_nodes()
        assert sky_graph.nodes[0] in low
        monkeypatch.delenv("CONFIDENCE_THRESHOLD")


# ── Graph undo history ────────────────────────────────────────────────────────

class TestGraphHistory:
    def test_no_history_on_fresh_graph(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert not graph.can_undo()

    def test_can_undo_after_edit(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_2", "changed")
        assert graph.can_undo()

    def test_undo_restores_content(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        original = graph.get_node("node_2").content
        graph.update_node("node_2", "changed content")
        graph.undo()
        assert graph.get_node("node_2").content == original

    def test_undo_returns_true_on_success(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_1", "edit")
        assert graph.undo() is True

    def test_undo_returns_false_on_empty_history(self, sky_graph):
        assert sky_graph.undo() is False

    def test_multiple_undos(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        original = graph.get_node("node_1").content
        graph.update_node("node_1", "first edit")
        graph.update_node("node_1", "second edit")
        graph.undo()
        assert graph.get_node("node_1").content == "first edit"
        graph.undo()
        assert graph.get_node("node_1").content == original

    def test_undo_after_recalculate(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        graph.update_node("node_2", "because of clouds")
        graph = mock_engine.recalculate_from_node(graph, "node_2")
        assert graph.version == 2
        graph.undo()  # undo the recalculation
        assert graph.version == 1

    def test_undo_restores_edit_count(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert graph.edit_count == 0
        graph.update_node("node_1", "edit")
        assert graph.edit_count == 1
        graph.undo()
        assert graph.edit_count == 0


# ── Node alternative branches ─────────────────────────────────────────────────

class TestNodeAlternatives:
    def test_add_alternative(self, sky_graph):
        sky_graph.add_alternative("node_1", "Alternative content")
        node = sky_graph.get_node("node_1")
        assert "Alternative content" in node.alternatives

    def test_add_duplicate_alternative_ignored(self, sky_graph):
        sky_graph.add_alternative("node_1", "Alt A")
        sky_graph.add_alternative("node_1", "Alt A")
        node = sky_graph.get_node("node_1")
        assert node.alternatives.count("Alt A") == 1

    def test_add_same_as_current_content_ignored(self, sky_graph):
        original = sky_graph.get_node("node_1").content
        sky_graph.add_alternative("node_1", original)
        node = sky_graph.get_node("node_1")
        assert original not in node.alternatives

    def test_add_empty_alternative_ignored(self, sky_graph):
        sky_graph.add_alternative("node_1", "")
        assert sky_graph.get_node("node_1").alternatives == []

    def test_alternatives_serialized(self, sky_graph):
        sky_graph.add_alternative("node_1", "Alt content")
        d = sky_graph.to_dict()
        assert "Alt content" in d["nodes"][0]["alternatives"]

    def test_alternatives_deserialized(self, sky_graph):
        sky_graph.add_alternative("node_1", "Alt content")
        d = sky_graph.to_dict()
        restored = ReasoningGraph.from_dict(d)
        assert "Alt content" in restored.get_node("node_1").alternatives

    def test_alternatives_backward_compat(self):
        """Nodes deserialized without alternatives field default to empty list."""
        d = {"id": "n1", "label": "Fact", "content": "hi", "node_type": "fact"}
        n = ReasoningNode.from_dict(d)
        assert n.alternatives == []

    def test_add_multiple_alternatives(self, sky_graph):
        sky_graph.add_alternative("node_1", "Alt A")
        sky_graph.add_alternative("node_1", "Alt B")
        sky_graph.add_alternative("node_1", "Alt C")
        assert len(sky_graph.get_node("node_1").alternatives) == 3


# ── Batch analysis ────────────────────────────────────────────────────────────

class TestBatchAnalyze:
    def test_batch_returns_correct_count(self, mock_engine):
        prompts = ["Why is the sky blue?", "What is 15% of 240?", "Why do things fall?"]
        graphs = mock_engine.batch_analyze(prompts)
        assert len(graphs) == 3

    def test_batch_each_has_nodes(self, mock_engine):
        prompts = ["Sky is blue", "Gravity", "Math: 5+5"]
        for g in mock_engine.batch_analyze(prompts):
            assert len(g.nodes) >= 1

    def test_batch_preserves_prompts(self, mock_engine):
        prompts = ["prompt alpha", "prompt beta"]
        graphs = mock_engine.batch_analyze(prompts)
        assert graphs[0].prompt == "prompt alpha"
        assert graphs[1].prompt == "prompt beta"

    def test_batch_empty_list(self, mock_engine):
        assert mock_engine.batch_analyze([]) == []

    def test_batch_single_prompt(self, mock_engine):
        graphs = mock_engine.batch_analyze(["Why is the sky blue?"])
        assert len(graphs) == 1
        assert graphs[0].get_conclusion() is not None

    def test_batch_each_graph_has_generation_time(self, mock_engine):
        graphs = mock_engine.batch_analyze(["Sky?", "Gravity?"])
        for g in graphs:
            assert g.generation_time_ms is not None
            assert g.generation_time_ms >= 0


# ── Graph statistics ──────────────────────────────────────────────────────────

class TestGraphStats:
    def test_stats_has_required_keys(self, sky_graph):
        s = sky_graph.stats()
        for key in ("version", "node_count", "edit_count", "edited_nodes",
                    "generation_time_ms", "avg_confidence", "low_confidence_count",
                    "has_alternatives"):
            assert key in s, f"Missing key: {key}"

    def test_stats_node_count(self, sky_graph):
        assert sky_graph.stats()["node_count"] == 3

    def test_stats_edit_count_starts_zero(self, sky_graph):
        assert sky_graph.stats()["edit_count"] == 0

    def test_stats_edit_count_increments(self, sky_graph):
        sky_graph.update_node("node_1", "new content")
        assert sky_graph.stats()["edit_count"] == 1

    def test_stats_edited_nodes(self, sky_graph):
        assert sky_graph.stats()["edited_nodes"] == 0
        sky_graph.update_node("node_1", "new content")
        assert sky_graph.stats()["edited_nodes"] == 1

    def test_stats_avg_confidence(self, sky_graph):
        for n in sky_graph.nodes:
            n.confidence = 0.8
        assert sky_graph.stats()["avg_confidence"] == pytest.approx(0.8, rel=1e-3)

    def test_stats_empty_graph_avg_confidence_is_none(self):
        g = ReasoningGraph(nodes=[], prompt="test")
        assert g.stats()["avg_confidence"] is None

    def test_stats_in_to_dict(self, sky_graph):
        d = sky_graph.to_dict()
        assert "stats" in d
        assert d["stats"]["node_count"] == 3

    def test_stats_has_alternatives_false_by_default(self, sky_graph):
        assert sky_graph.stats()["has_alternatives"] is False

    def test_stats_has_alternatives_true_after_branch(self, sky_graph):
        sky_graph.add_alternative("node_1", "alt")
        assert sky_graph.stats()["has_alternatives"] is True


# ── COT steps configuration ───────────────────────────────────────────────────

class TestCOTSteps:
    def test_default_cot_steps_is_three(self, mock_engine):
        assert mock_engine.cot_steps == 3

    def test_cot_steps_from_env(self, monkeypatch):
        monkeypatch.setenv("COT_STEPS", "5")
        e = ReasoningEngine(mode="mock")
        assert e.cot_steps == 5
        monkeypatch.delenv("COT_STEPS")

    def test_cot_steps_clamped_to_two_minimum(self, monkeypatch):
        monkeypatch.setenv("COT_STEPS", "1")
        e = ReasoningEngine(mode="mock")
        assert e.cot_steps == 2
        monkeypatch.delenv("COT_STEPS")

    def test_cot_steps_clamped_to_seven_maximum(self, monkeypatch):
        monkeypatch.setenv("COT_STEPS", "99")
        e = ReasoningEngine(mode="mock")
        assert e.cot_steps == 7
        monkeypatch.delenv("COT_STEPS")

    def test_system_prompt_includes_step_count(self, mock_engine):
        prompt = mock_engine._build_system_generate()
        assert str(mock_engine.cot_steps) in prompt

    def test_generation_time_tracked(self, mock_engine):
        graph = mock_engine.generate_cot("Explain why the sky is blue")
        assert graph.generation_time_ms is not None
        assert graph.generation_time_ms >= 0


# ── Node timestamps ───────────────────────────────────────────────────────────

class TestNodeTimestamps:
    def test_created_at_set_on_new_node(self):
        from datetime import datetime
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT)
        assert n.created_at
        # Should be a parseable ISO-8601 timestamp
        datetime.fromisoformat(n.created_at)

    def test_edited_at_none_by_default(self):
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT)
        assert n.edited_at is None

    def test_edited_at_set_after_update(self, sky_graph):
        from datetime import datetime
        sky_graph.update_node("node_1", "new content")
        node = sky_graph.get_node("node_1")
        assert node.edited_at is not None
        datetime.fromisoformat(node.edited_at)

    def test_timestamps_serialized(self):
        n = ReasoningNode(id="n1", label="Fact", content="hello", node_type=NodeType.FACT)
        d = n.to_dict()
        assert "created_at" in d
        assert "edited_at" in d

    def test_timestamps_round_trip(self):
        n = ReasoningNode(
            id="n1", label="Fact", content="hello", node_type=NodeType.FACT,
            created_at="2026-01-01T00:00:00+00:00",
            edited_at="2026-01-02T12:30:00+00:00",
        )
        restored = ReasoningNode.from_dict(n.to_dict())
        assert restored.created_at == "2026-01-01T00:00:00+00:00"
        assert restored.edited_at == "2026-01-02T12:30:00+00:00"


# ── Mock template coverage ────────────────────────────────────────────────────

class TestMockTemplates:
    @pytest.mark.parametrize("prompt,expected_label", [
        ("Explain why the sky is blue", "Physics"),
        ("What is 15% of 240?", "Setup"),
        ("Why do objects fall towards the Earth?", "Physics"),
        ("Why does water boil at 100 celsius?", "Chemistry"),
        ("Explain how plants make energy from sunlight", "Biology"),
        ("How is a rainbow formed?", "Optics"),
        ("Why can't light escape from a black hole?", "Physics"),
        ("Why do we have seasons on Earth?", "Astronomy"),
        ("How does DNA replication work?", "Biology"),
    ])
    def test_mock_template_routing(self, mock_engine, prompt, expected_label):
        graph = mock_engine.generate_cot(prompt)
        labels = [n.label for n in graph.nodes]
        assert expected_label in labels, (
            f"Expected label '{expected_label}' in {labels} for prompt: {prompt!r}"
        )

    def test_default_template_used_for_unknown_prompt(self, mock_engine):
        graph = mock_engine.generate_cot("Tell me about the history of chess")
        assert len(graph.nodes) >= 1
        assert graph.get_conclusion() is not None
