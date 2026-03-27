# CoT Surgeon – Inspect and surgically edit LLM chain-of-thought reasoning graphs

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-104%20passed-brightgreen.svg)]()

> Parse any LLM's chain-of-thought into a structured graph of fact → reasoning → conclusion nodes, edit any node, and watch the downstream conclusions recalculate.

## Install

```bash
git clone https://github.com/dakshjain-1616/cot-surgeon
cd cot-surgeon
pip install -r requirements.txt
```

## What problem this solves

When a reasoning model gives the wrong answer, you can't tell if it's a bad fact, a flawed inference step, or a broken conclusion — the output is just a wall of text. There's no tool that lets you isolate the exact node in the reasoning chain that went wrong, fix it, and see whether the conclusion changes. LangChain's tracing shows token streams, not reasoning structure. CoT Surgeon parses chain-of-thought output into a typed graph (`fact` → `reasoning` → `conclusion`), gives each node a confidence score, and lets you edit any node with `update_node()` — the graph tracks edit history so you can undo.

## Real world examples

```python
from cot_surgeon import ReasoningEngine

engine = ReasoningEngine()  # uses OPENROUTER_API_KEY if set, mock mode otherwise

# Parse a question into a structured reasoning graph
graph = engine.generate_cot("Why does ice float on water?")

print(f"Generated {len(graph.nodes)} nodes:")
for node in graph.nodes:
    print(f"  [{node.node_type.value:11s}] conf={node.confidence:.2f}  {node.label}")
# [fact       ] conf=0.85  Fact
# [reasoning  ] conf=0.80  Analysis
# [conclusion ] conf=0.75  Conclusion
```

```python
# Edit a fact node and track the change
graph.update_node(
    graph.nodes[0].id,
    "Water molecules form a hexagonal lattice when frozen, increasing intermolecular spacing"
)
print("edit_count:", graph.edit_count)   # 1
print("can_undo:", graph.can_undo())     # True

# Undo the edit
graph.undo()
print("edit_count after undo:", graph.edit_count)  # 0
```

```python
# Find low-confidence nodes — likely spots for errors
weak = graph.low_confidence_nodes(threshold=0.7)
for node in weak:
    print(f"  [{node.node_type.value}] conf={node.confidence:.2f} — {node.label}")
```

```python
# Batch analyze multiple prompts
results = engine.batch_analyze([
    "Why is the sky blue?",
    "How does GPS work?",
    "Why do we dream?",
])
for graph in results:
    print(f"Prompt: {graph.prompt[:40]}...")
    print(f"  Nodes: {len(graph.nodes)}, edit_count: {graph.edit_count}")
# Prompt: Why is the sky blue?...
#   Nodes: 3, edit_count: 0
# Prompt: How does GPS work?...
#   Nodes: 3, edit_count: 0
```

```python
# Export as Mermaid diagram (paste into any Mermaid renderer)
print(graph.to_mermaid())
# graph TD
#     node_0["Fact\nconf: 0.85"]
#     node_1["Analysis\nconf: 0.80"]
#     node_0 --> node_1
#     node_2["Conclusion\nconf: 0.75"]
#     node_1 --> node_2
```

## Who it's for

AI researchers and prompt engineers who debug multi-step reasoning failures in models like DeepSeek-R1, Qwen3, or o3. If you've ever stared at a 2000-token chain-of-thought trying to find where the logic broke down, CoT Surgeon gives you a structured graph you can traverse and edit programmatically.

## Key features

- Typed node graph: `fact`, `reasoning`, `conclusion` — not just raw text
- Per-node confidence scores (0.0–1.0) with `low_confidence_nodes()` filter
- Full undo history via `snapshot()` / `undo()` — edit safely, revert instantly
- Mermaid diagram export with `to_mermaid()` for visual inspection
- Batch analysis with `batch_analyze()` across multiple prompts
- Works without any API key in mock mode — full offline test coverage

## Run tests

```
$ pytest tests/ -v --tb=no -q --no-header

tests/test_reasoning.py ................................................ [ 46%]
........................................................                 [100%]

104 passed in 0.69s
```

## Project structure

```
cot-surgeon/
├── cot_surgeon/
│   ├── reasoning_engine.py  ← ReasoningEngine, ReasoningGraph, ReasoningNode
│   └── __init__.py
├── tests/
│   └── test_reasoning.py    ← 104 tests
├── scripts/demo.py          ← runnable demo with HTML output
└── requirements.txt
```
