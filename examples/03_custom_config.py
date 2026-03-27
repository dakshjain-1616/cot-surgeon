"""
03_custom_config.py — Customising behaviour via environment variables.

Demonstrates:
  - Setting COT_STEPS to control reasoning chain length
  - CONFIDENCE_THRESHOLD to change the low-confidence detection level
  - MAX_HISTORY to control undo depth
  - Alternative branches on a node

Run:
    python examples/03_custom_config.py

Or override settings at the shell level:
    COT_STEPS=5 CONFIDENCE_THRESHOLD=0.9 python examples/03_custom_config.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Configure via env vars before importing the engine ────────────────────────
os.environ.setdefault("COT_STEPS", "4")
os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.9")
os.environ.setdefault("MAX_HISTORY", "5")

from cot_surgeon import ReasoningEngine

engine = ReasoningEngine(mode="mock")
print(f"CoT steps : {engine.cot_steps}")
print(f"Confidence threshold (env): {os.environ['CONFIDENCE_THRESHOLD']}\n")

# ── Generate with configured step count ───────────────────────────────────────
# Note: mock templates always return 3 nodes; the step count affects real LLM calls.
graph = engine.generate_cot("How does DNA replication work?")
print(f"Nodes: {len(graph.nodes)}")
for n in graph.nodes:
    print(f"  [{n.label}] conf={n.confidence:.0%} — {n.content[:80]}")

# ── Low-confidence detection ──────────────────────────────────────────────────
low = graph.low_confidence_nodes()
print(f"\nLow-confidence nodes (threshold={os.environ['CONFIDENCE_THRESHOLD']}): {len(low)}")
for n in low:
    print(f"  [{n.label}] conf={n.confidence:.0%}")

# ── Alternative branches ──────────────────────────────────────────────────────
print("\n── Alternative branches ──")
node = graph.nodes[1]
graph.add_alternative(node.id, "Alternative reasoning path A")
graph.add_alternative(node.id, "Alternative reasoning path B")
print(f"Node [{node.label}] has {len(node.alternatives)} alternatives:")
for i, alt in enumerate(node.alternatives, 1):
    print(f"  {i}. {alt}")
