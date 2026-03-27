"""
02_advanced_usage.py — Node editing, recalculation, undo, and graph statistics.

Demonstrates:
  - Generating a CoT graph
  - Editing a reasoning node in-place
  - Recalculating only the downstream nodes
  - Undoing a recalculation
  - Inspecting graph statistics

Run:
    python examples/02_advanced_usage.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cot_surgeon import ReasoningEngine

engine = ReasoningEngine(mode="mock")

# ── Generate ──────────────────────────────────────────────────────────────────
graph = engine.generate_cot("Explain why the sky is blue")
print(f"Generated graph v{graph.version} with {len(graph.nodes)} nodes")
print(f"Stats: {graph.stats()}\n")

# ── Edit a node ───────────────────────────────────────────────────────────────
physics = next(n for n in graph.nodes if n.label == "Physics")
print(f"Original [{physics.label}]: {physics.content[:80]}")

graph.update_node(physics.id, "Because of clouds blocking and scattering light differently")
print(f"Edited  [{physics.label}]: {graph.get_node(physics.id).content}")
print(f"Can undo: {graph.can_undo()}\n")

# ── Recalculate downstream ────────────────────────────────────────────────────
graph = engine.recalculate_from_node(graph, physics.id)
print(f"After recalculate — graph v{graph.version}")
print(f"New conclusion: {graph.get_conclusion()[:120]}\n")

# ── Undo ──────────────────────────────────────────────────────────────────────
graph.undo()
print(f"After undo — graph v{graph.version}")
print(f"Conclusion restored: {graph.get_conclusion()[:120]}\n")

# ── Stats ─────────────────────────────────────────────────────────────────────
s = graph.stats()
print("Graph statistics:")
for k, v in s.items():
    print(f"  {k}: {v}")
