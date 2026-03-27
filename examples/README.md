# CoT Surgeon — Examples

All scripts use mock mode by default (no API key required).
Each script adds the project root to `sys.path` automatically so it can be run from any directory.

## Running an example

```bash
python examples/01_quick_start.py
```

## Scripts

| Script | What it demonstrates |
|---|---|
| [01_quick_start.py](01_quick_start.py) | Minimal usage — generate a CoT graph and print each node |
| [02_advanced_usage.py](02_advanced_usage.py) | Node editing, downstream recalculation, undo, and graph statistics |
| [03_custom_config.py](03_custom_config.py) | Configuring behaviour via env vars (`COT_STEPS`, `CONFIDENCE_THRESHOLD`, `MAX_HISTORY`) and using alternative branches |
| [04_full_pipeline.py](04_full_pipeline.py) | End-to-end workflow: generate → inspect → edit + branch → recalculate → batch analyze → export JSON + Mermaid |

## Using a real LLM

Set `OPENROUTER_API_KEY` in your `.env` file (copy `.env.example`) and the scripts will
automatically use OpenRouter instead of the mock templates:

```bash
cp .env.example .env
# edit .env and add OPENROUTER_API_KEY=sk-...
python examples/04_full_pipeline.py
```
