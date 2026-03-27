"""
demo.py — CoT Surgeon runnable demo.

Works without any API keys: auto-detects mock mode when OPENROUTER_API_KEY
is absent. Saves all outputs to outputs/ directory on every run.

Usage:
    python demo.py            # mock / auto mode
    python demo.py --local    # local llama.cpp (requires LLAMA_MODEL_PATH)
    python demo.py --quiet    # suppress verbose output
    python demo.py --batch    # run batch comparison demo (Demo 4)
    python demo.py --version  # print version and exit
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:  # pragma: no cover — rich is optional at runtime
    _RICH = False

load_dotenv()

VERSION = "1.0.0"
console = Console() if _RICH else None

OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── HTML report generator ─────────────────────────────────────────────────────

def _confidence_bar(confidence: float) -> str:
    """Return an HTML confidence badge with colour-coded background."""
    pct = int(confidence * 100)
    if confidence >= 0.85:
        color = "#4CAF50"
    elif confidence >= 0.65:
        color = "#FF9800"
    else:
        color = "#F44336"
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'padding:1px 6px;border-radius:8px;font-size:11px;font-weight:600">'
        f'{pct}%</span>'
    )


def _html_report(results: list, timestamp: str, mode: str, stats_summary: dict) -> str:
    """Render a self-contained HTML report for a demo run."""
    node_colors = {"fact": "#4CAF50", "reasoning": "#2196F3", "conclusion": "#FF9800"}

    sections_html = ""
    for r in results:
        if "graph" not in r:
            continue
        g = r["graph"]
        nodes_html = ""
        for nd in g["nodes"]:
            c = node_colors.get(nd["node_type"], "#888")
            edited_badge = " <em style='color:#9C27B0'>(edited)</em>" if nd.get("edited") else ""
            conf_html = _confidence_bar(nd.get("confidence", 1.0))
            ts_html = ""
            if nd.get("created_at"):
                ts_html += f'<span style="font-size:10px;color:#888"> · created {nd["created_at"]}</span>'
            if nd.get("edited_at"):
                ts_html += f'<span style="font-size:10px;color:#9C27B0"> · edited {nd["edited_at"]}</span>'
            alts = nd.get("alternatives", [])
            alts_html = ""
            if alts:
                alts_html = f'<div style="margin-top:4px;font-size:11px;color:#666">Branches: {len(alts)}</div>'
            nodes_html += (
                f'<div class="node" style="border-left:4px solid {c};padding:10px;'
                f'margin:8px 0;background:#f8f9fa;border-radius:4px;">'
                f'<strong style="color:{c}">[{nd["label"]}]</strong>{edited_badge}'
                f' {conf_html}{ts_html}'
                f'<p style="margin:6px 0 0 0;line-height:1.5">{nd["content"]}</p>'
                f'{alts_html}'
                f"</div>"
            )
        title = r.get("prompt") or r.get("action", "Edit &amp; Recalculate")
        conclusion_html = ""
        if r.get("conclusion"):
            conclusion_html = (
                f'<div class="conclusion"><strong>Conclusion:</strong> {r["conclusion"]}</div>'
            )
        gs = g.get("stats", {})
        stats_html = ""
        if gs:
            avg_c = f"{gs.get('avg_confidence', 0):.0%}" if gs.get("avg_confidence") else "—"
            gen_ms = f"{gs.get('generation_time_ms', 0):.0f} ms" if gs.get("generation_time_ms") else "—"
            stats_html = (
                f'<div style="font-size:11px;color:#888;margin-top:8px">'
                f'v{gs.get("version","?")} · {gs.get("node_count","?")} nodes · '
                f'avg confidence {avg_c} · generated in {gen_ms}'
                f'</div>'
            )
        sections_html += (
            f'<div class="demo-section">'
            f'<h3>Demo {r["demo"]}: {title}</h3>'
            f"{nodes_html}{conclusion_html}{stats_html}"
            f"</div>"
        )

    summary_html = ""
    if stats_summary:
        rows = "".join(
            f'<tr><td>{k}</td><td><strong>{v}</strong></td></tr>'
            for k, v in stats_summary.items()
        )
        summary_html = f"""
        <h2>Run Summary</h2>
        <table style="border-collapse:collapse;width:100%;font-size:13px">
          <tr style="background:#f0f4ff"><th style="text-align:left;padding:6px 10px">Metric</th>
          <th style="text-align:left;padding:6px 10px">Value</th></tr>
          {rows}
        </table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CoT Surgeon — Demo Report</title>
  <style>
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:860px;
         margin:0 auto;padding:24px;background:#fff;color:#222}}
    h1{{color:#1a1a2e;border-bottom:3px solid #6C63FF;padding-bottom:10px}}
    h2{{color:#444}}h3{{color:#6C63FF}}
    .meta{{background:#f0f4ff;padding:10px 16px;border-radius:6px;font-size:13px;margin:12px 0}}
    .demo-section{{margin:28px 0;padding:20px;border:1px solid #ddd;border-radius:8px}}
    .conclusion{{margin-top:14px;padding:10px 14px;background:#fff3e0;border-radius:6px}}
    table td,table th{{padding:6px 10px;border-bottom:1px solid #eee}}
    .footer{{margin-top:40px;font-size:12px;color:#888;text-align:center}}
  </style>
</head>
<body>
  <h1>🧠 CoT Surgeon — Demo Report</h1>
  <div class="meta">
    <strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
    <strong>Mode:</strong> {mode.upper()} &nbsp;|&nbsp;
    <strong>Demos:</strong> {len(results)}
  </div>
  {summary_html}
  <h2>Reasoning Graphs</h2>
  {sections_html}
  <div class="footer">
    Generated by <a href="https://github.com/dakshjain-1616/cot-surgeon">CoT Surgeon</a>
    — made autonomously with <a href="https://heyneo.so">NEO</a>
  </div>
</body>
</html>"""


# ── Rich output helpers ────────────────────────────────────────────────────────

def _print_banner(mode: str) -> None:
    """Print the startup banner — Rich panel if available, plain text otherwise."""
    if _RICH:
        content = Text()
        content.append("🧠  CoT Surgeon  ", style="bold cyan")
        content.append(f"v{VERSION}\n", style="bold white")
        content.append("Mode: ", style="dim white")
        mode_style = "bold yellow" if mode == "mock" else "bold green"
        content.append(f"{mode.upper()}", style=mode_style)
        content.append(f"  ·  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim white")
        if mode == "mock":
            content.append(
                "\n[dim]No OPENROUTER_API_KEY found — running in mock mode.\n"
                "Set OPENROUTER_API_KEY in .env for real LLM output.[/dim]"
            )
        console.print(
            Panel(
                content,
                subtitle="[dim]Made autonomously by [link=https://heyneo.so]NEO[/link] · heyneo.so[/dim]",
                border_style="cyan",
                expand=False,
            )
        )
    else:
        print("")
        print("=" * 62)
        print(f"  🧠  CoT Surgeon v{VERSION} — Demo")
        print(f"  Mode   : {mode.upper()}")
        print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if mode == "mock":
            print("  Note   : No OPENROUTER_API_KEY found — using mock mode.")
            print("           Set OPENROUTER_API_KEY in .env for real LLM output.")
        print("  Made autonomously by NEO · heyneo.so")
        print("=" * 62)


def _print_nodes_table(nodes) -> None:
    """Print reasoning nodes — Rich table if available, plain text otherwise."""
    if _RICH:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", expand=False)
        table.add_column("Label", style="cyan", min_width=12)
        table.add_column("Type", style="blue", min_width=10)
        table.add_column("Conf", justify="right", min_width=5)
        table.add_column("Content")
        for n in nodes:
            conf_pct = f"{n.confidence:.0%}"
            if n.confidence >= 0.85:
                conf_style = "bold green"
            elif n.confidence >= 0.65:
                conf_style = "bold yellow"
            else:
                conf_style = "bold red"
            table.add_row(
                n.label,
                n.node_type.value,
                Text(conf_pct, style=conf_style),
                n.content[:120] + ("…" if len(n.content) > 120 else ""),
            )
        console.print(table)
    else:
        for n in nodes:
            print(f"     [{n.label:<12}] conf={n.confidence:.0%}  {n.content[:80]}")


def _print_stats_table(stats_summary: dict) -> None:
    """Print run statistics — Rich table if available, plain text otherwise."""
    if _RICH:
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold blue", expand=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for k, v in stats_summary.items():
            table.add_row(str(k), str(v))
        console.print(table)
    else:
        for k, v in stats_summary.items():
            print(f"  {k}: {v}")


# ── Core demo logic ───────────────────────────────────────────────────────────

def run_demo(local: bool = False, verbose: bool = True, batch: bool = False) -> list:
    """Run all demo scenarios and save outputs to the outputs/ directory."""
    from cot_surgeon import ReasoningEngine

    if local:
        mode = "local"
    elif os.getenv("OPENROUTER_API_KEY"):
        mode = "openrouter"
    else:
        mode = "mock"

    def _log(msg: str, rich_msg: str = "") -> None:
        """Print a message — use rich markup version if Rich is available."""
        if not verbose:
            return
        if _RICH:
            console.print(rich_msg or msg)
        else:
            print(msg)

    if verbose:
        _print_banner(mode)

    engine = ReasoningEngine(mode=mode)
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_gen_ms = 0.0

    # ── Demo 1: Sky is blue ───────────────────────────────────────────────────
    prompt1 = os.getenv("DEMO_PROMPT_1", "Explain why the sky is blue")
    _log(f'\n[Demo 1] Prompt: "{prompt1}"',
         f'\n[bold cyan][Demo 1][/bold cyan] Prompt: [yellow]"{prompt1}"[/yellow]')

    if _RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      TimeElapsedColumn(), console=console, transient=True) as progress:
            task = progress.add_task("Generating chain-of-thought…", total=None)
            graph1 = engine.generate_cot(prompt1)
            progress.update(task, completed=True)
    else:
        graph1 = engine.generate_cot(prompt1)

    _log(
        f"  → Generated {len(graph1.nodes)} nodes in {graph1.generation_time_ms:.0f} ms:",
        f"  [green]✓[/green] Generated [bold]{len(graph1.nodes)}[/bold] nodes "
        f"in [bold]{graph1.generation_time_ms:.0f} ms[/bold]",
    )
    if verbose:
        _print_nodes_table(graph1.nodes)
    if graph1.generation_time_ms:
        total_gen_ms += graph1.generation_time_ms

    results.append({
        "demo": 1,
        "prompt": prompt1,
        "mode": mode,
        "graph": graph1.to_dict(),
        "conclusion": graph1.get_conclusion(),
    })

    # ── Demo 2: Edit [Physics] node, recalculate ──────────────────────────────
    _log(
        "\n[Demo 2] Editing middle reasoning node → recalculate conclusion",
        "\n[bold cyan][Demo 2][/bold cyan] Editing middle reasoning node → recalculate conclusion",
    )

    graph2 = engine.generate_cot(prompt1)
    edit_id = None
    for n in graph2.nodes:
        if n.label in ("Physics", "Analysis", "Computation", "Reasoning"):
            edit_id = n.id
            break
    if edit_id is None and len(graph2.nodes) > 1:
        edit_id = graph2.nodes[1].id

    original_content = ""
    new_content_edit = "Because of clouds blocking and scattering light differently"
    if edit_id:
        original_content = graph2.get_node(edit_id).content
        node_label = graph2.get_node(edit_id).label
        _log(
            f"  Original [{node_label}]: {original_content[:80]}",
            f"  [dim]Original [{node_label}]:[/dim] {original_content[:80]}",
        )
        _log(
            f'  Replacing with: "{new_content_edit}"',
            f'  [yellow]Replacing with:[/yellow] "{new_content_edit}"',
        )

        graph2.update_node(edit_id, new_content_edit)

        if _RICH:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          TimeElapsedColumn(), console=console, transient=True) as progress:
                task = progress.add_task("Recalculating downstream nodes…", total=None)
                graph2 = engine.recalculate_from_node(graph2, edit_id)
                progress.update(task, completed=True)
        else:
            graph2 = engine.recalculate_from_node(graph2, edit_id)

        new_conclusion = graph2.get_conclusion()

        _log(
            f"  → New conclusion (v{graph2.version}): {new_conclusion[:100]}",
            f"  [green]✓[/green] New conclusion [bold](v{graph2.version})[/bold]: "
            f"{new_conclusion[:100]}",
        )
        _log(
            f"  → Edit count: {graph2.edit_count}, Can undo: {graph2.can_undo()}",
            f"  [dim]Edit count: {graph2.edit_count} · Can undo: {graph2.can_undo()}[/dim]",
        )

        results.append({
            "demo": 2,
            "action": "edit_physics_and_recalculate",
            "edited_node_id": edit_id,
            "original_content": original_content,
            "new_content": new_content_edit,
            "new_conclusion": new_conclusion,
            "graph": graph2.to_dict(),
            "conclusion": new_conclusion,
        })

    # ── Demo 3: Math prompt ───────────────────────────────────────────────────
    prompt3 = os.getenv("DEMO_PROMPT_2", "What is 15% of 240?")
    _log(f'\n[Demo 3] Prompt: "{prompt3}"',
         f'\n[bold cyan][Demo 3][/bold cyan] Prompt: [yellow]"{prompt3}"[/yellow]')

    if _RICH:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      TimeElapsedColumn(), console=console, transient=True) as progress:
            task = progress.add_task("Generating chain-of-thought…", total=None)
            graph3 = engine.generate_cot(prompt3)
            progress.update(task, completed=True)
    else:
        graph3 = engine.generate_cot(prompt3)

    _log(
        f"  → Generated {len(graph3.nodes)} nodes in {graph3.generation_time_ms:.0f} ms:",
        f"  [green]✓[/green] Generated [bold]{len(graph3.nodes)}[/bold] nodes "
        f"in [bold]{graph3.generation_time_ms:.0f} ms[/bold]",
    )
    if verbose:
        _print_nodes_table(graph3.nodes)
    if graph3.generation_time_ms:
        total_gen_ms += graph3.generation_time_ms

    results.append({
        "demo": 3,
        "prompt": prompt3,
        "mode": mode,
        "graph": graph3.to_dict(),
        "conclusion": graph3.get_conclusion(),
    })

    # ── Demo 4: Batch comparison ──────────────────────────────────────────────
    if batch:
        batch_prompts = [
            "Why do objects fall towards the Earth?",
            "How is a rainbow formed?",
            "Why does water boil at 100°C at sea level?",
        ]
        _log(
            f"\n[Demo 4] Batch comparison — {len(batch_prompts)} prompts",
            f"\n[bold cyan][Demo 4][/bold cyan] Batch comparison — "
            f"[bold]{len(batch_prompts)}[/bold] prompts",
        )

        if _RICH:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                          TimeElapsedColumn(), console=console, transient=True) as progress:
                task = progress.add_task(f"Analyzing {len(batch_prompts)} prompts…", total=None)
                graphs = engine.batch_analyze(batch_prompts)
                progress.update(task, completed=True)
        else:
            graphs = engine.batch_analyze(batch_prompts)

        for bp, bg in zip(batch_prompts, graphs):
            _log(
                f'  Prompt: "{bp[:60]}"',
                f'  [dim]Prompt:[/dim] "{bp[:60]}"',
            )
            _log(
                f"  → {len(bg.nodes)} nodes, avg conf={bg.stats()['avg_confidence']:.0%}",
                f"  [green]→[/green] {len(bg.nodes)} nodes, "
                f"avg conf=[bold]{bg.stats()['avg_confidence']:.0%}[/bold]",
            )
            _log(
                f"     Conclusion: {bg.get_conclusion()[:80]}",
                f"     [dim]Conclusion:[/dim] {bg.get_conclusion()[:80]}",
            )
            if bg.generation_time_ms:
                total_gen_ms += bg.generation_time_ms

        results.append({
            "demo": 4,
            "action": "batch_comparison",
            "prompts": batch_prompts,
            "graph": graphs[0].to_dict(),
            "conclusion": graphs[0].get_conclusion(),
            "batch_conclusions": [g.get_conclusion() for g in graphs],
        })

    # ── Stats summary ─────────────────────────────────────────────────────────
    stats_summary = {
        "Mode": mode.upper(),
        "Total demos": len(results),
        "Total generation time": f"{total_gen_ms:.0f} ms",
        "Engine model": engine.model,
        "CoT steps configured": engine.cot_steps,
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    results_path = OUTPUTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "mode": mode,
            "stats_summary": stats_summary,
            "demos": results,
        }, f, indent=2)

    mmd_path = OUTPUTS_DIR / "graphs.mmd"
    with open(mmd_path, "w") as f:
        f.write("%% CoT Surgeon — Generated Reasoning Graphs\n")
        f.write(f"%% Timestamp: {timestamp}  Mode: {mode}\n\n")
        f.write(f"%% Demo 1: {prompt1}\n")
        f.write(graph1.to_mermaid())
        f.write("\n\n")
        if edit_id:
            f.write("%% Demo 2: After editing node\n")
            f.write(graph2.to_mermaid())
            f.write("\n\n")
        f.write(f"%% Demo 3: {prompt3}\n")
        f.write(graph3.to_mermaid())
        f.write("\n")

    report_path = OUTPUTS_DIR / "report.html"
    with open(report_path, "w") as f:
        f.write(_html_report(results, timestamp, mode, stats_summary))

    if verbose:
        if _RICH:
            console.print("\n[bold green]Run Summary[/bold green]")
            _print_stats_table(stats_summary)
            console.print(f"  [green]✓[/green] [dim]{results_path}[/dim]")
            console.print(f"  [green]✓[/green] [dim]{mmd_path}[/dim]")
            console.print(f"  [green]✓[/green] [dim]{report_path}[/dim]")
            console.print(
                Panel(
                    "[green]Demo complete![/green]  Files saved to [bold]outputs/[/bold]\n"
                    "Run [bold cyan]streamlit run app.py[/bold cyan] for the interactive UI.",
                    border_style="green",
                    expand=False,
                )
            )
        else:
            print(f"\n[Output] {results_path}")
            print(f"[Output] {mmd_path}")
            print(f"[Output] {report_path}")
            print("")
            print("=" * 62)
            _print_stats_table(stats_summary)
            print("  Demo complete! Files saved to outputs/")
            print("  Run 'streamlit run app.py' for the interactive UI.")
            print("=" * 62)
            print("")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoT Surgeon demo")
    parser.add_argument("--local", action="store_true", help="Use local llama.cpp model")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--batch", action="store_true", help="Run batch comparison demo (Demo 4)")
    parser.add_argument(
        "--version", action="version", version=f"CoT Surgeon v{VERSION}"
    )
    args = parser.parse_args()
    run_demo(local=args.local, verbose=not args.quiet, batch=args.batch)
