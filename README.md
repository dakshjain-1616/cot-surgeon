# cot-surgeon – Edit LLM reasoning chains node by node

> *Made autonomously using [NEO](https://heyneo.so) · [![Install NEO Extension](https://img.shields.io/badge/VS%20Code-Install%20NEO-7B61FF?logo=visual-studio-code)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo)*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-104%20passed-brightgreen.svg)]()

> Stop re-prompting blindly; visualize and surgically fix broken reasoning steps in your LLM workflows.

## Install

```bash
git clone https://github.com/dakshjain-1616/cot-surgeon
cd cot-surgeon
pip install -r requirements.txt
```

## The Problem

Debugging LLM reasoning currently requires re-submitting prompts repeatedly, hoping the model self-corrects without visibility into specific failure points. Tools like LangSmith, LangFuse, and OpenAI Playground provide traces but treat Chain-of-Thought as an opaque string, preventing you from editing a single faulty reasoning node without regenerating the entire chain.

## Who it's for

This tool is for ML engineers and prompt engineers debugging complex reasoning tasks like math proofs, logic puzzles, or multi-step code generation. You need it when your model produces the wrong final answer due to a hallucinated intermediate step, and you want to inject the correct fact directly into the reasoning flow to verify downstream logic without rewriting the system prompt.

## Quickstart

Launch the interactive Streamlit playground:

```bash
streamlit run app.py
```

Or use the reasoning engine programmatically:

```python
from cot_surgeon.reasoning_engine import ReasoningEngine

engine = ReasoningEngine(model_path="./models/llama-7b.gguf")
graph = engine.generate_graph(prompt="Calculate the integral of x^2")
graph.edit_node(node_id=2, new_text="Integral of x^2 is x^3/3 + C")
final_answer = engine.recalculate(graph)
```

## Key features

- **CoT graph generation** — Breaks prompts into `Fact → Reasoning → Conclusion` nodes with per-node confidence scores (0.0–1.0)
- **Surgical node editing** — Edit any node in-place; only downstream nodes regenerate on **Recalculate**
- **Undo history** — Full snapshot/undo stack allows branching and comparing alternatives without losing prior work

## Run tests

```bash
pytest tests/ -q
# 104 passed
```

## Project structure

```
cot-surgeon/
├── cot_surgeon/      ← main library
├── tests/            ← test suite
├── scripts/          ← demo scripts
├── examples/         ← usage examples
├── app.py            ← Streamlit UI entry
└── requirements.txt
```