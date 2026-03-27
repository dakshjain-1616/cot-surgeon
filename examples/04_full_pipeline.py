"""
04_full_pipeline.py — End-to-end CoT Surgeon workflow.

Demonstrates the complete pipeline:
  1. Generate a reasoning graph
  2. Inspect and compare nodes
  3. Edit a node + add an alternative branch
  4. Recalculate downstream nodes
  5. Batch-analyze multiple prompts
  6. Export results to JSON and Mermaid

Outputs are written to outputs/pipeline_*.json and outputs/pipeline_*.mmd

Run:
    python examples/04_full_pipeline.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cot_surgeon import ReasoningEngine

OUTPUTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

engine = ReasoningEngine(mode="mock")

# ── Step 1: Generate ──────────────────────────────────────────────────────────
print("=== Step 1: Generate reasoning graph ===")
graph = engine.generate_cot("Explain why the sky is blue")
print(f"Generated {len(graph.nodes)} nodes in {graph.generation_time_ms:.0f} ms")
for n in graph.nodes:
    print(f"  [{n.label:<12}] {n.node_type.value:<12} conf={n.confidence:.0%}  {n.content[:70]}")

# ── Step 2: Inspect statistics ────────────────────────────────────────────────
print("\n=== Step 2: Graph statistics ===")
s = graph.stats()
print(f"  version={s['version']}  nodes={s['node_count']}  avg_confidence={s['avg_confidence']:.0%}")
low = graph.low_confidence_nodes(threshold=0.7)
print(f"  Low-confidence nodes: {len(low)}")

# ── Step 3: Edit + alternative branch ─────────────────────────────────────────
print("\n=== Step 3: Edit a node & save an alternative ===")
physics = next(n for n in graph.nodes if n.label == "Physics")
original = physics.content
graph.add_alternative(physics.id, original)  # save original as a branch
graph.update_node(physics.id, "Clouds scatter light differently — the sky looks grey/white")
print(f"  Original saved as branch: {original[:60]}…")
print(f"  New content: {graph.get_node(physics.id).content[:70]}")
print(f"  Branches stored: {len(graph.get_node(physics.id).alternatives)}")

# ── Step 4: Recalculate ───────────────────────────────────────────────────────
print("\n=== Step 4: Recalculate downstream nodes ===")
graph = engine.recalculate_from_node(graph, physics.id)
print(f"  Graph is now v{graph.version}")
print(f"  New conclusion: {graph.get_conclusion()[:110]}")

# ── Step 5: Batch analysis ────────────────────────────────────────────────────
print("\n=== Step 5: Batch analysis (3 prompts) ===")
prompts = [
    "Why do objects fall towards the Earth?",
    "How is a rainbow formed?",
    "Why does water boil at 100°C at sea level?",
]
graphs = engine.batch_analyze(prompts)
for prompt, g in zip(prompts, graphs):
    s = g.stats()
    print(f"  [{prompt[:50]:<50}] nodes={s['node_count']} avg_conf={s['avg_confidence']:.0%}")
    print(f"    Conclusion: {g.get_conclusion()[:90]}")

# ── Step 6: Export ────────────────────────────────────────────────────────────
print("\n=== Step 6: Export results ===")

json_path = os.path.join(OUTPUTS, "pipeline_result.json")
with open(json_path, "w") as f:
    export = {
        "main_graph": graph.to_dict(),
        "batch": [{"prompt": p, "graph": g.to_dict()} for p, g in zip(prompts, graphs)],
    }
    json.dump(export, f, indent=2)
print(f"  JSON  → {json_path}")

mmd_path = os.path.join(OUTPUTS, "pipeline_graphs.mmd")
with open(mmd_path, "w") as f:
    f.write("%% CoT Surgeon — Full Pipeline Example\n\n")
    f.write("%% Main graph (after edit + recalculate)\n")
    f.write(graph.to_mermaid())
    f.write("\n\n")
    for prompt, g in zip(prompts, graphs):
        f.write(f"%% {prompt}\n")
        f.write(g.to_mermaid())
        f.write("\n\n")
print(f"  Mermaid → {mmd_path}")

print("\nPipeline complete.")
