"""
app.py — CoT Surgeon Streamlit UI.

Interactive playground for visualising and surgically editing
LLM chain-of-thought reasoning graphs.
"""

import os
import json
import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv

load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CoT Surgeon",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def render_mermaid(mermaid_code: str, height: int = 460) -> None:
    """Render Mermaid.js diagram inside a Streamlit HTML component."""
    safe = mermaid_code.replace("`", "&#96;")
    html = f"""<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
  .mermaid {{ font-family: 'Segoe UI', sans-serif; }}
</style>
</head>
<body>
<div class="mermaid">
{safe}
</div>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'dark',
    flowchart: {{ curve: 'basis', useMaxWidth: true }},
    securityLevel: 'loose'
  }});
</script>
</body>
</html>"""
    st_html(html, height=height, scrolling=False)


def get_engine():
    """Return (or create) the cached ReasoningEngine in session state."""
    from cot_surgeon import ReasoningEngine
    if "engine" not in st.session_state:
        mode = st.session_state.get("llm_mode", "auto")
        st.session_state.engine = ReasoningEngine(mode=mode)
    return st.session_state.engine


def load_demo_prompts() -> list:
    try:
        with open("demo_prompts.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return [{"label": "Sky color", "prompt": "Explain why the sky is blue"}]


def _confidence_badge(confidence: float) -> str:
    pct = int(confidence * 100)
    if confidence >= 0.85:
        color = "#4CAF50"
    elif confidence >= 0.65:
        color = "#FF9800"
    else:
        color = "#F44336"
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:10px;font-size:12px;font-weight:600">{pct}%</span>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")

        mode = st.selectbox(
            "LLM Mode",
            ["auto", "openrouter", "local", "mock"],
            index=0,
            help=(
                "auto: use OpenRouter if key present, otherwise mock.\n"
                "local: use llama.cpp (set LLAMA_MODEL_PATH).\n"
                "mock: offline demo mode."
            ),
        )

        if mode != st.session_state.get("llm_mode"):
            st.session_state.llm_mode = mode
            st.session_state.pop("engine", None)

        if mode in ("auto", "openrouter"):
            api_key = st.text_input(
                "OpenRouter API Key",
                value=os.getenv("OPENROUTER_API_KEY", ""),
                type="password",
                help="Get a free key at openrouter.ai",
            )
            if api_key:
                os.environ["OPENROUTER_API_KEY"] = api_key

            model = st.selectbox(
                "Model",
                [
                    "mistralai/mistral-small-2603",
                    "openai/gpt-5.4-nano",
                    "openai/gpt-5.4-mini",
                    "z-ai/glm-5-turbo",
                    "minimax/minimax-m2.7",
                    "reka/reka-edge",
                    "openai/gpt-5.4",
                    "x-ai/grok-4.20-beta",
                ],
            )
            os.environ["OPENROUTER_MODEL"] = model

        if mode == "local":
            model_path = st.text_input(
                "GGUF Model Path",
                value=os.getenv("LLAMA_MODEL_PATH", ""),
                placeholder="/models/mistral-7b.gguf",
            )
            if model_path:
                os.environ["LLAMA_MODEL_PATH"] = model_path

        st.divider()

        cot_steps = st.slider(
            "CoT Steps",
            min_value=2,
            max_value=7,
            value=int(os.getenv("COT_STEPS", "3")),
            help="Number of reasoning steps to generate (2–7).",
        )
        os.environ["COT_STEPS"] = str(cot_steps)
        # Reset engine when steps change so it picks up the new value
        if cot_steps != st.session_state.get("cot_steps_last"):
            st.session_state.cot_steps_last = cot_steps
            st.session_state.pop("engine", None)

        st.divider()

        st.markdown("### Legend")
        st.markdown(
            "🟢 **Fact** — foundational premise  \n"
            "🔵 **Reasoning** — mechanism / analysis  \n"
            "🟠 **Conclusion** — final answer  \n"
            "🔴 **Selected** — node being edited  \n"
            "🟣 **Edited** — user-modified node"
        )

        st.divider()

        if st.button("🗑️ Clear Session", use_container_width=True):
            for k in ["graph", "engine", "selected_node", "batch_graphs"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── Single Analysis tab ───────────────────────────────────────────────────────

def tab_single():
    from cot_surgeon import ReasoningGraph, NodeType

    demo_prompts = load_demo_prompts()

    # ── Prompt input ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])
    with col_left:
        options = ["Custom prompt…"] + [p["label"] for p in demo_prompts]
        picked = st.selectbox("Quick prompts", options, key="single_pick")
        default_text = (
            next((p["prompt"] for p in demo_prompts if p["label"] == picked), "")
            if picked != "Custom prompt…"
            else ""
        )
        prompt = st.text_area(
            "Your prompt",
            value=default_text,
            height=90,
            placeholder="e.g. Explain why the sky is blue",
            key="single_prompt",
        )

    with col_right:
        st.markdown("<div style='height:54px'></div>", unsafe_allow_html=True)
        analyze = st.button("🔬 Analyze", type="primary", use_container_width=True, key="single_analyze")

    if analyze and prompt.strip():
        with st.spinner("Generating chain-of-thought…"):
            engine = get_engine()
            graph = engine.generate_cot(prompt.strip())
            st.session_state.graph = graph
            st.session_state.selected_node = None
        st.rerun()

    # ── Graph + editor ────────────────────────────────────────────────────────
    if "graph" not in st.session_state:
        st.info(
            "Enter a prompt above and click **Analyze** to generate a reasoning graph."
        )
        return

    graph: ReasoningGraph = st.session_state.graph
    engine = get_engine()

    # Low-confidence warning
    low_conf = graph.low_confidence_nodes()
    if low_conf:
        names = ", ".join(f"**{n.label}**" for n in low_conf)
        st.warning(
            f"⚠️ Low-confidence nodes detected ({names}). "
            "Consider reviewing or editing these steps."
        )

    st.divider()

    col_graph, col_editor = st.columns([3, 2])

    # ── Graph panel ───────────────────────────────────────────────────────────
    with col_graph:
        st.subheader("📊 Reasoning Graph")
        s = graph.stats()
        gen_ms = f"{s['generation_time_ms']:.0f} ms" if s["generation_time_ms"] else "—"
        st.caption(
            f"Mode: `{engine.mode.upper()}` · Version: `v{graph.version}` · "
            f"Nodes: `{len(graph.nodes)}` · Generated in: `{gen_ms}`"
        )

        selected_id = st.session_state.get("selected_node")
        mermaid_code = graph.to_mermaid(selected_id)
        render_mermaid(mermaid_code, height=460)

        with st.expander("View Mermaid source"):
            st.code(mermaid_code, language="text")

    # ── Editor panel ──────────────────────────────────────────────────────────
    with col_editor:
        st.subheader("✂️ Node Editor")

        node_map = {f"[{n.label}]  {n.id}": n.id for n in graph.nodes}
        chosen_key = st.selectbox("Select node", list(node_map.keys()), key="node_select")
        chosen_id = node_map[chosen_key]
        st.session_state.selected_node = chosen_id

        node = graph.get_node(chosen_id)
        if node:
            type_badge = {
                NodeType.FACT: "🟢 FACT",
                NodeType.REASONING: "🔵 REASONING",
                NodeType.CONCLUSION: "🟠 CONCLUSION",
            }.get(node.node_type, node.node_type.value)

            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.markdown(f"**Type:** {type_badge}")
            with meta_col2:
                st.markdown(
                    f"**Confidence:** {_confidence_badge(node.confidence)}",
                    unsafe_allow_html=True,
                )

            # Timestamps
            ts_lines = [f"🕐 Created: `{node.created_at}`"]
            if node.edited_at:
                ts_lines.append(f"✏️ Edited: `{node.edited_at}`")
            st.caption("  \n".join(ts_lines))

            if node.edited:
                st.info("✏️ This node has been manually edited")

            new_content = st.text_area(
                "Content",
                value=node.content,
                height=140,
                key=f"textarea_{chosen_id}",
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("💾 Save", use_container_width=True, key="btn_save"):
                    graph.update_node(chosen_id, new_content)
                    st.session_state.graph = graph
                    st.success("Saved!")
                    st.rerun()

            with btn_col2:
                if st.button("🔄 Recalculate", type="primary", use_container_width=True, key="btn_recalc"):
                    graph.update_node(chosen_id, new_content)
                    with st.spinner("Recalculating downstream nodes…"):
                        graph = engine.recalculate_from_node(graph, chosen_id)
                    st.session_state.graph = graph
                    st.success(f"Done — graph is now v{graph.version}")
                    st.rerun()

            # Undo button
            if graph.can_undo():
                if st.button("↩️ Undo", use_container_width=True, key="btn_undo"):
                    graph.undo()
                    st.session_state.graph = graph
                    st.session_state.selected_node = None
                    st.rerun()

            # ── Alternative branches ──────────────────────────────────────
            st.markdown("---")
            st.markdown("**🌿 Alternative Branches**")

            if node.alternatives:
                alt_options = ["(current)"] + [
                    f"Alt {i + 1}: {a[:50]}{'…' if len(a) > 50 else ''}"
                    for i, a in enumerate(node.alternatives)
                ]
                alt_pick = st.selectbox("View branch", alt_options, key=f"alt_pick_{chosen_id}")
                if alt_pick != "(current)":
                    alt_idx = int(alt_pick.split(":")[0].replace("Alt ", "")) - 1
                    alt_content = node.alternatives[alt_idx]
                    st.text_area("Branch content", value=alt_content, height=80, disabled=True,
                                 key=f"alt_view_{chosen_id}")
                    if st.button("Apply this branch", key=f"alt_apply_{chosen_id}"):
                        graph.add_alternative(chosen_id, node.content)  # save current as alt
                        graph.update_node(chosen_id, alt_content)
                        st.session_state.graph = graph
                        st.rerun()

            alt_input = st.text_area(
                "New alternative content",
                height=80,
                placeholder="Enter an alternative reasoning for this node…",
                key=f"alt_input_{chosen_id}",
            )
            if st.button("➕ Save as Branch", use_container_width=True, key=f"alt_save_{chosen_id}"):
                if alt_input.strip():
                    graph.add_alternative(chosen_id, alt_input.strip())
                    st.session_state.graph = graph
                    st.success("Branch saved!")
                    st.rerun()

    # ── Conclusion banner ─────────────────────────────────────────────────────
    st.divider()
    conclusion = graph.get_conclusion()
    if conclusion:
        st.subheader("💡 Current Conclusion")
        st.info(conclusion)

    # ── Stats ─────────────────────────────────────────────────────────────────
    with st.expander("📈 Graph Statistics"):
        s = graph.stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", f"v{s['version']}")
        c2.metric("Edit Count", s["edit_count"])
        c3.metric("Avg Confidence", f"{s['avg_confidence']:.0%}" if s["avg_confidence"] else "—")
        c4.metric("Low Confidence", s["low_confidence_count"])
        if s["generation_time_ms"]:
            st.caption(f"Generated in {s['generation_time_ms']:.0f} ms")

    # ── Export controls ───────────────────────────────────────────────────────
    ec1, ec2 = st.columns(2)
    with ec1:
        st.download_button(
            "📥 Export JSON",
            json.dumps(graph.to_dict(), indent=2),
            file_name="reasoning_graph.json",
            mime="application/json",
        )
    with ec2:
        st.download_button(
            "📥 Export Mermaid (.mmd)",
            graph.to_mermaid(),
            file_name="reasoning_graph.mmd",
            mime="text/plain",
        )


# ── Batch Compare tab ─────────────────────────────────────────────────────────

def tab_batch():
    from cot_surgeon import ReasoningGraph

    st.markdown("### Compare reasoning chains for multiple prompts side-by-side.")
    st.markdown(
        "Enter two or more prompts and click **Compare** to generate and diff their "
        "CoT graphs simultaneously."
    )

    demo_prompts = load_demo_prompts()
    demo_options = ["(none)"] + [p["label"] for p in demo_prompts]

    prompts = []
    cols = st.columns(2)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**Prompt {chr(65 + i)}**")
            pick = st.selectbox(
                "Quick pick", demo_options, key=f"batch_pick_{i}",
                label_visibility="collapsed",
            )
            default = (
                next((p["prompt"] for p in demo_prompts if p["label"] == pick), "")
                if pick != "(none)" else ""
            )
            txt = st.text_area(
                f"Prompt {chr(65 + i)}",
                value=default,
                height=80,
                placeholder=f"Enter prompt {chr(65 + i)}…",
                key=f"batch_txt_{i}",
                label_visibility="collapsed",
            )
            prompts.append(txt.strip())

    # Optional third prompt
    with st.expander("➕ Add a third prompt"):
        pick_c = st.selectbox("Quick pick C", demo_options, key="batch_pick_2",
                              label_visibility="collapsed")
        default_c = (
            next((p["prompt"] for p in demo_prompts if p["label"] == pick_c), "")
            if pick_c != "(none)" else ""
        )
        txt_c = st.text_area(
            "Prompt C", value=default_c, height=80, placeholder="Enter prompt C…",
            key="batch_txt_2", label_visibility="collapsed",
        )
        if txt_c.strip():
            prompts.append(txt_c.strip())

    st.markdown("")
    if st.button("🔍 Compare", type="primary", key="batch_compare"):
        active = [p for p in prompts if p]
        if len(active) < 2:
            st.warning("Please enter at least two prompts.")
        else:
            with st.spinner(f"Generating {len(active)} reasoning graphs…"):
                engine = get_engine()
                graphs = engine.batch_analyze(active)
                st.session_state.batch_graphs = list(zip(active, graphs))
            st.rerun()

    if "batch_graphs" not in st.session_state:
        return

    batch = st.session_state.batch_graphs
    st.divider()
    st.subheader("📊 Comparison Results")

    # One column per graph
    result_cols = st.columns(len(batch))
    for col, (prompt, graph) in zip(result_cols, batch):
        with col:
            label = prompt[:60] + ("…" if len(prompt) > 60 else "")
            st.markdown(f"**{label}**")
            s = graph.stats()
            gen_ms = f"{s['generation_time_ms']:.0f} ms" if s["generation_time_ms"] else "—"
            st.caption(
                f"Nodes: {s['node_count']} · "
                f"Avg conf: {s['avg_confidence']:.0%} · "
                f"{gen_ms}"
            )
            render_mermaid(graph.to_mermaid(), height=320)

    st.subheader("💡 Conclusion Comparison")
    conc_cols = st.columns(len(batch))
    for col, (prompt, graph) in zip(conc_cols, batch):
        with col:
            label = prompt[:40] + ("…" if len(prompt) > 40 else "")
            st.markdown(f"**{label}**")
            conclusion = graph.get_conclusion()
            if conclusion:
                st.info(conclusion)

    # Download all as JSON
    batch_export = [
        {"prompt": p, "graph": g.to_dict()} for p, g in batch
    ]
    st.download_button(
        "📥 Export Batch JSON",
        json.dumps(batch_export, indent=2),
        file_name="batch_graphs.json",
        mime="application/json",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sidebar()

    st.title("🧠 CoT Surgeon")
    st.markdown(
        "_Visualise, edit, and surgically repair LLM chain-of-thought reasoning._"
    )

    tab1, tab2 = st.tabs(["🔬 Single Analysis", "📊 Batch Compare"])

    with tab1:
        tab_single()

    with tab2:
        tab_batch()


if __name__ == "__main__":
    main()
