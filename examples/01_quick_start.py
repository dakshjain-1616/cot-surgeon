"""
01_quick_start.py — Minimal CoT Surgeon example.

Generates a 3-step chain-of-thought graph in mock mode (no API key needed)
and prints each reasoning node.

Run:
    python examples/01_quick_start.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cot_surgeon import ReasoningEngine

engine = ReasoningEngine(mode="mock")
graph = engine.generate_cot("Explain why the sky is blue")

print(f"Prompt : {graph.prompt}")
print(f"Nodes  : {len(graph.nodes)}")
print()

for node in graph.nodes:
    print(f"[{node.label}] ({node.node_type.value}) conf={node.confidence:.0%}")
    print(f"  {node.content[:120]}")
    print()

print(f"Conclusion: {graph.get_conclusion()[:100]}")
