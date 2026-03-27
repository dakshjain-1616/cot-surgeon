"""
reasoning_engine.py — CoT Surgeon core engine.

Orchestrates LLM calls to generate chain-of-thought reasoning graphs,
supports node editing, recalculates downstream conclusions.

New in this version:
  - Node confidence scores (0.0–1.0)
  - Node timestamps (created_at / edited_at)
  - Node alternative branches
  - Graph undo history (snapshot / undo)
  - Graph statistics (generation_time_ms, edit_count, avg_confidence)
  - Batch analysis (batch_analyze)
  - Configurable CoT step count (COT_STEPS env var)
  - Retry logic with exponential back-off for OpenRouter calls
  - Richer mock templates (gravity, chemistry, biology, rainbow, blackhole)
"""

import os
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    """Return current UTC time as an ISO-8601 string (seconds precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    FACT = "fact"
    REASONING = "reasoning"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningNode:
    id: str
    label: str
    content: str
    node_type: NodeType
    dependencies: List[str] = field(default_factory=list)
    edited: bool = False
    # New fields
    confidence: float = 1.0          # 0.0–1.0; how well-supported this node is
    created_at: str = field(default_factory=_utcnow)
    edited_at: Optional[str] = None  # set when content is changed
    alternatives: List[str] = field(default_factory=list)  # branching alternatives

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "content": self.content,
            "node_type": self.node_type.value,
            "dependencies": self.dependencies,
            "edited": self.edited,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "edited_at": self.edited_at,
            "alternatives": self.alternatives,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningNode":
        return cls(
            id=data["id"],
            label=data["label"],
            content=data["content"],
            node_type=NodeType(data["node_type"]),
            dependencies=data.get("dependencies", []),
            edited=data.get("edited", False),
            confidence=float(data.get("confidence", 1.0)),
            created_at=data.get("created_at", _utcnow()),
            edited_at=data.get("edited_at"),
            alternatives=data.get("alternatives", []),
        )


@dataclass
class ReasoningGraph:
    nodes: List[ReasoningNode]
    prompt: str
    version: int = 1
    # New fields
    generation_time_ms: Optional[float] = None
    edit_count: int = 0
    history: List[dict] = field(default_factory=list)  # undo stack (not serialized)

    # ------------------------------------------------------------------
    # Node accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_node_index(self, node_id: str) -> int:
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                return i
        return -1

    # ------------------------------------------------------------------
    # Undo history
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Save current state to the undo history stack."""
        self.history.append({
            "version": self.version,
            "edit_count": self.edit_count,
            "nodes": [n.to_dict() for n in self.nodes],
        })
        max_hist = int(os.getenv("MAX_HISTORY", "20"))
        if len(self.history) > max_hist:
            self.history = self.history[-max_hist:]

    def can_undo(self) -> bool:
        return len(self.history) > 0

    def undo(self) -> bool:
        """Restore the most recent snapshot. Returns True if successful."""
        if not self.history:
            return False
        snap = self.history.pop()
        self.nodes = [ReasoningNode.from_dict(n) for n in snap["nodes"]]
        self.version = snap["version"]
        self.edit_count = snap.get("edit_count", self.edit_count)
        return True

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_node(self, node_id: str, content: str) -> None:
        node = self.get_node(node_id)
        if node:
            self.snapshot()
            node.content = content
            node.edited = True
            node.edited_at = _utcnow()
            self.edit_count += 1

    def add_alternative(self, node_id: str, content: str) -> None:
        """Store an alternative content branch without overwriting the current value."""
        node = self.get_node(node_id)
        if node and content and content != node.content and content not in node.alternatives:
            node.alternatives.append(content)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def low_confidence_nodes(self, threshold: Optional[float] = None) -> List[ReasoningNode]:
        if threshold is None:
            threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        return [n for n in self.nodes if n.confidence < threshold]

    def stats(self) -> dict:
        avg_conf = (
            round(sum(n.confidence for n in self.nodes) / len(self.nodes), 3)
            if self.nodes else None
        )
        return {
            "version": self.version,
            "node_count": len(self.nodes),
            "edit_count": self.edit_count,
            "edited_nodes": sum(1 for n in self.nodes if n.edited),
            "generation_time_ms": self.generation_time_ms,
            "avg_confidence": avg_conf,
            "low_confidence_count": len(self.low_confidence_nodes()),
            "has_alternatives": any(bool(n.alternatives) for n in self.nodes),
        }

    # ------------------------------------------------------------------
    # Mermaid export
    # ------------------------------------------------------------------

    def to_mermaid(self, selected_node_id: str = None) -> str:
        lines = ["graph TD"]

        for node in self.nodes:
            display = node.content[:55].replace('"', "'").replace("\n", " ")
            if len(node.content) > 55:
                display += "..."
            conf_label = f" [{node.confidence:.0%}]" if node.confidence < 1.0 else ""
            lines.append(f'    {node.id}["{node.label}{conf_label}\\n{display}"]')

        for node in self.nodes:
            for dep in node.dependencies:
                lines.append(f"    {dep} --> {node.id}")

        for node in self.nodes:
            if node.id == selected_node_id:
                lines.append(f"    style {node.id} fill:#FF5722,color:#fff,stroke:#FF5722,stroke-width:3px")
            elif node.edited:
                lines.append(f"    style {node.id} fill:#9C27B0,color:#fff")
            elif node.node_type == NodeType.FACT:
                lines.append(f"    style {node.id} fill:#4CAF50,color:#fff")
            elif node.node_type == NodeType.REASONING:
                lines.append(f"    style {node.id} fill:#2196F3,color:#fff")
            else:
                lines.append(f"    style {node.id} fill:#FF9800,color:#fff")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "version": self.version,
            "nodes": [n.to_dict() for n in self.nodes],
            "generation_time_ms": self.generation_time_ms,
            "edit_count": self.edit_count,
            "stats": self.stats(),
            # history is runtime state — not serialized
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningGraph":
        nodes = [ReasoningNode.from_dict(n) for n in data["nodes"]]
        return cls(
            nodes=nodes,
            prompt=data["prompt"],
            version=data.get("version", 1),
            generation_time_ms=data.get("generation_time_ms"),
            edit_count=data.get("edit_count", 0),
        )

    def get_conclusion(self) -> Optional[str]:
        for node in reversed(self.nodes):
            if node.node_type == NodeType.CONCLUSION:
                return node.content
        return self.nodes[-1].content if self.nodes else None


# ---------------------------------------------------------------------------
# Mock responses — realistic CoT output for demo/testing without API keys
# ---------------------------------------------------------------------------

_MOCK_COT: dict = {
    "sky": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Sunlight contains the full visible spectrum (400–700 nm). "
                    "Earth's atmosphere consists mainly of nitrogen and oxygen molecules."
                ),
                "node_type": "fact",
                "confidence": 0.98,
            },
            {
                "label": "Physics",
                "content": (
                    "Rayleigh scattering: shorter wavelengths (blue ≈450 nm) scatter "
                    "roughly 10× more than longer wavelengths (red ≈700 nm) when "
                    "photons interact with atmospheric molecules."
                ),
                "node_type": "reasoning",
                "confidence": 0.97,
            },
            {
                "label": "Conclusion",
                "content": (
                    "The sky appears blue because scattered blue light reaches our "
                    "eyes from every direction, while red/orange wavelengths travel "
                    "more directly and only dominate at sunrise/sunset."
                ),
                "node_type": "conclusion",
                "confidence": 0.95,
            },
        ]
    },
    "math": {
        "nodes": [
            {
                "label": "Setup",
                "content": "Identify the operation: percentage calculation requires multiplying the base by (percent / 100).",
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Computation",
                "content": "Apply the formula: value × (percent ÷ 100). Follow standard order of operations.",
                "node_type": "reasoning",
                "confidence": 0.98,
            },
            {
                "label": "Answer",
                "content": "The computation yields the precise numerical result according to the formula.",
                "node_type": "conclusion",
                "confidence": 0.97,
            },
        ]
    },
    "gravity": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Earth has mass (~5.97 × 10²⁴ kg). Newton's law of universal gravitation "
                    "states that all massive objects attract each other with force F = Gm₁m₂/r²."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Physics",
                "content": (
                    "Near Earth's surface, gravitational acceleration g ≈ 9.81 m/s² acts "
                    "downward on all objects. Without a supporting surface, objects accelerate "
                    "continuously toward Earth's center."
                ),
                "node_type": "reasoning",
                "confidence": 0.98,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Objects fall toward Earth because gravitational attraction between their mass "
                    "and Earth's mass produces a net downward acceleration of ~9.81 m/s², "
                    "as described by Newton's law of universal gravitation."
                ),
                "node_type": "conclusion",
                "confidence": 0.97,
            },
        ]
    },
    "chemistry": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Water (H₂O) boils when molecules gain enough kinetic energy to overcome "
                    "intermolecular hydrogen bonds. At sea level, atmospheric pressure is 101.325 kPa."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Chemistry",
                "content": (
                    "Boiling occurs when vapor pressure equals atmospheric pressure. "
                    "At sea level (101.325 kPa), water reaches this equilibrium at exactly 100°C. "
                    "At higher altitudes, lower air pressure means water boils at lower temperatures."
                ),
                "node_type": "reasoning",
                "confidence": 0.97,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Water boils at 100°C at sea level because that is the temperature at which "
                    "its vapor pressure equals standard atmospheric pressure (101.325 kPa), "
                    "allowing bubble formation throughout the liquid."
                ),
                "node_type": "conclusion",
                "confidence": 0.96,
            },
        ]
    },
    "biology": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Plants contain chlorophyll in chloroplasts. Chlorophyll absorbs light "
                    "(mainly red 680 nm and blue 430 nm). CO₂ enters via stomata; "
                    "water is absorbed through roots."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Biology",
                "content": (
                    "Light reactions split water and capture photon energy as ATP + NADPH. "
                    "The Calvin cycle uses ATP + NADPH + CO₂ to synthesize glucose (C₆H₁₂O₆) "
                    "— releasing O₂ as a byproduct."
                ),
                "node_type": "reasoning",
                "confidence": 0.96,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Plants produce energy by converting sunlight, CO₂, and water into glucose "
                    "via photosynthesis. Summary: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."
                ),
                "node_type": "conclusion",
                "confidence": 0.95,
            },
        ]
    },
    "rainbow": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Rainbows form when sunlight interacts with water droplets. "
                    "White light contains all visible wavelengths (400–700 nm)."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Optics",
                "content": (
                    "Each droplet acts as a prism: light refracts entering, reflects off the "
                    "back surface, and refracts again exiting. Dispersion separates colors. "
                    "The observer sees each color from droplets at a specific angle "
                    "(42° for red, 40° for violet)."
                ),
                "node_type": "reasoning",
                "confidence": 0.96,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Rainbows appear as arcs of color because sunlight is refracted, dispersed, "
                    "and internally reflected inside water droplets, with each color reaching "
                    "the observer's eye from droplets at a specific angular elevation."
                ),
                "node_type": "conclusion",
                "confidence": 0.95,
            },
        ]
    },
    "blackhole": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "A black hole is a region of spacetime with curvature so extreme that "
                    "escape velocity exceeds c (~3×10⁸ m/s). The boundary is the event horizon."
                ),
                "node_type": "fact",
                "confidence": 0.98,
            },
            {
                "label": "Physics",
                "content": (
                    "General relativity: mass-energy curves spacetime. Beyond the event horizon, "
                    "all geodesics (paths in spacetime) point inward toward the singularity. "
                    "Since light follows geodesics and cannot exceed c, it cannot escape."
                ),
                "node_type": "reasoning",
                "confidence": 0.93,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Light cannot escape a black hole because within the event horizon the "
                    "curvature of spacetime is so extreme that every possible path — including "
                    "light's path — leads inward. Escape would require exceeding c, "
                    "which is physically impossible."
                ),
                "node_type": "conclusion",
                "confidence": 0.92,
            },
        ]
    },
    "seasons": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "Earth's rotational axis is tilted ~23.5° relative to its orbital plane "
                    "around the Sun. This tilt remains roughly constant as Earth orbits."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Astronomy",
                "content": (
                    "When a hemisphere tilts toward the Sun, it receives sunlight at a steeper "
                    "angle and for longer each day — more energy per unit area → summer. "
                    "The opposite hemisphere simultaneously receives less direct sunlight → winter."
                ),
                "node_type": "reasoning",
                "confidence": 0.97,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Seasons are caused by Earth's axial tilt, not its distance from the Sun. "
                    "The hemisphere tilted toward the Sun experiences summer; "
                    "the hemisphere tilted away experiences winter simultaneously."
                ),
                "node_type": "conclusion",
                "confidence": 0.96,
            },
        ]
    },
    "dna": {
        "nodes": [
            {
                "label": "Fact",
                "content": (
                    "DNA is a double helix composed of complementary base pairs "
                    "(A-T, G-C). Before replication, the helix unwinds via helicase."
                ),
                "node_type": "fact",
                "confidence": 0.99,
            },
            {
                "label": "Biology",
                "content": (
                    "DNA polymerase reads each strand 3'→5' and synthesizes a new complementary "
                    "strand 5'→3'. The leading strand is synthesized continuously; the lagging "
                    "strand in short Okazaki fragments, later joined by DNA ligase."
                ),
                "node_type": "reasoning",
                "confidence": 0.95,
            },
            {
                "label": "Conclusion",
                "content": (
                    "DNA replication produces two identical double helices (semi-conservative "
                    "replication). Each new helix retains one original strand and one newly "
                    "synthesized strand, ensuring genetic fidelity."
                ),
                "node_type": "conclusion",
                "confidence": 0.94,
            },
        ]
    },
    "default": {
        "nodes": [
            {
                "label": "Fact",
                "content": "A key factual observation relevant to the question establishes the foundation.",
                "node_type": "fact",
                "confidence": 0.85,
            },
            {
                "label": "Analysis",
                "content": (
                    "Examining the underlying mechanism reveals a cause-and-effect "
                    "relationship between the primary factors involved."
                ),
                "node_type": "reasoning",
                "confidence": 0.80,
            },
            {
                "label": "Conclusion",
                "content": (
                    "Synthesizing the facts and analysis: the phenomenon occurs due "
                    "to the interplay of these identified mechanisms."
                ),
                "node_type": "conclusion",
                "confidence": 0.75,
            },
        ]
    },
}


def _pick_mock_template(prompt: str) -> list:
    p = prompt.lower()
    if any(w in p for w in ["sky", "blue", "colour", "color", "rayleigh", "scatter"]):
        return _MOCK_COT["sky"]["nodes"]
    if any(w in p for w in ["math", "calculat", "percent", "equation", "solve", "number", "%", "invest", "interest", "compound"]):
        return _MOCK_COT["math"]["nodes"]
    if any(w in p for w in ["fall", "gravity", "gravit", "weight", "newton", "drop"]):
        return _MOCK_COT["gravity"]["nodes"]
    if any(w in p for w in ["boil", "100", "celsius", "fahrenheit", "vapor", "pressure", "temperature"]):
        return _MOCK_COT["chemistry"]["nodes"]
    if any(w in p for w in ["plant", "photosynthesis", "chlorophyll", "leaf", "leaves", "glucose"]):
        return _MOCK_COT["biology"]["nodes"]
    if any(w in p for w in ["rainbow", "prism", "refract", "dispersion", "arc"]):
        return _MOCK_COT["rainbow"]["nodes"]
    if any(w in p for w in ["black hole", "blackhole", "event horizon", "singularity"]):
        return _MOCK_COT["blackhole"]["nodes"]
    if any(w in p for w in ["season", "winter", "summer", "tilt", "axis", "solstice"]):
        return _MOCK_COT["seasons"]["nodes"]
    if any(w in p for w in ["dna", "replicate", "replication", "helix", "genome", "gene"]):
        return _MOCK_COT["dna"]["nodes"]
    return _MOCK_COT["default"]["nodes"]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ReasoningEngine:
    """Orchestrates LLM calls and manages chain-of-thought reasoning graphs."""

    def __init__(self, mode: str = "auto"):
        """
        Args:
            mode: "auto" | "openrouter" | "local" | "mock"
        """
        self.mode = self._resolve_mode(mode)
        self._client: Optional[OpenAI] = None
        self._llama = None
        self.model = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-small-2603")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))
        self.cot_steps = max(2, min(int(os.getenv("COT_STEPS", "3")), 7))

        if self.mode == "openrouter" and OPENAI_AVAILABLE:
            self._client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _resolve_mode(self, requested: str) -> str:
        if requested == "mock":
            return "mock"
        if requested == "local" or os.getenv("LOCAL_MODE", "").lower() in ("1", "true", "yes"):
            return "local"
        if os.getenv("OPENROUTER_API_KEY") and requested in ("auto", "openrouter"):
            return "openrouter"
        return "mock"

    def _init_llama(self) -> bool:
        if not LLAMA_AVAILABLE:
            return False
        model_path = os.getenv("LLAMA_MODEL_PATH", "")
        if not model_path or not os.path.exists(model_path):
            return False
        try:
            self._llama = Llama(
                model_path=model_path,
                n_ctx=int(os.getenv("LLAMA_CTX_SIZE", "2048")),
                n_threads=int(os.getenv("LLAMA_THREADS", "4")),
                verbose=False,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # LLM call routing
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[dict]) -> str:
        if self.mode == "mock":
            return self._mock_generate(messages[-1]["content"])
        if self.mode == "local":
            return self._call_llama(messages)
        return self._call_openrouter(messages)

    def _call_openrouter(self, messages: List[dict]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
                    temperature=float(os.getenv("TEMPERATURE", "0.7")),
                    extra_headers={
                        "HTTP-Referer": os.getenv("APP_URL", "https://github.com/dakshjain-1616/cot-surgeon"),
                        "X-Title": "CoT Surgeon",
                    },
                )
                return resp.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
        # All retries exhausted — fall back to mock
        return self._mock_generate(messages[-1]["content"])

    def _call_llama(self, messages: List[dict]) -> str:
        if self._llama is None and not self._init_llama():
            return self._mock_generate(messages[-1]["content"])
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        prompt += "\nASSISTANT:"
        try:
            out = self._llama(
                prompt,
                max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                stop=["USER:", "\n\n\n"],
            )
            return out["choices"][0]["text"]
        except Exception:
            return self._mock_generate(messages[-1]["content"])

    # ------------------------------------------------------------------
    # Mock generation
    # ------------------------------------------------------------------

    def _mock_generate(self, prompt: str) -> str:
        template = _pick_mock_template(prompt)
        return json.dumps({"nodes": template})

    def _mock_recalculate(self, graph: "ReasoningGraph", edited_node_id: str) -> str:
        edited_node = graph.get_node(edited_node_id)
        edited_idx = graph.get_node_index(edited_node_id)
        content_lower = edited_node.content.lower()

        if any(w in content_lower for w in ["cloud", "clouds"]):
            new_conclusion = (
                "Clouds scatter and absorb light differently than gas molecules. "
                "When clouds dominate the sky, they block and diffuse light, "
                "making the sky appear white or grey rather than blue."
            )
            new_conf = 0.88
        elif any(w in content_lower for w in ["scatter", "rayleigh", "wavelength"]):
            new_conclusion = (
                "The updated scattering model confirms that differential wavelength "
                "scattering determines perceived sky color across different conditions."
            )
            new_conf = 0.90
        elif any(w in content_lower for w in ["formula", "equation", "calculat"]):
            new_conclusion = (
                "Applying the updated formula yields the recalculated numerical answer."
            )
            new_conf = 0.93
        else:
            snippet = edited_node.content[:60]
            new_conclusion = (
                f"Given the revised understanding ({snippet}...), the conclusion "
                "is updated to reflect the new logical pathway through the reasoning chain."
            )
            new_conf = 0.78

        subsequent = []
        for node in graph.nodes[edited_idx + 1:]:
            if node.node_type == NodeType.CONCLUSION:
                subsequent.append({
                    "label": node.label,
                    "content": new_conclusion,
                    "node_type": "conclusion",
                    "confidence": new_conf,
                })
            else:
                snippet = edited_node.content[:55]
                subsequent.append({
                    "label": node.label,
                    "content": f"Re-evaluated in light of: {snippet}...",
                    "node_type": node.node_type.value,
                    "confidence": round(new_conf - 0.05, 2),
                })
        return json.dumps({"nodes": subsequent})

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_nodes(self, response: str, id_offset: int = 0) -> List[ReasoningNode]:
        # 1. Try JSON extraction
        try:
            m = re.search(r"\{[\s\S]*\}", response)
            if m:
                data = json.loads(m.group())
                nodes: List[ReasoningNode] = []
                prev_id: Optional[str] = None
                for i, nd in enumerate(data.get("nodes", [])):
                    nid = f"node_{id_offset + i + 1}"
                    node = ReasoningNode(
                        id=nid,
                        label=nd.get("label", f"Step {i + 1}"),
                        content=nd.get("content", ""),
                        node_type=NodeType(nd.get("node_type", "reasoning")),
                        dependencies=[prev_id] if prev_id else [],
                        confidence=float(nd.get("confidence", 1.0)),
                    )
                    nodes.append(node)
                    prev_id = nid
                if nodes:
                    return nodes
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # 2. Regex fallback for bracket-tagged text
        patterns = [
            (r"\[Fact\](.*?)(?=\[|$)", NodeType.FACT, "Fact"),
            (r"\[Physics\](.*?)(?=\[|$)", NodeType.REASONING, "Physics"),
            (r"\[Analysis\](.*?)(?=\[|$)", NodeType.REASONING, "Analysis"),
            (r"\[Setup\](.*?)(?=\[|$)", NodeType.FACT, "Setup"),
            (r"\[Computation\](.*?)(?=\[|$)", NodeType.REASONING, "Computation"),
            (r"\[Conclusion\](.*?)(?=\[|$)", NodeType.CONCLUSION, "Conclusion"),
            (r"\[Answer\](.*?)(?=\[|$)", NodeType.CONCLUSION, "Answer"),
        ]
        nodes = []
        prev_id = None
        i = 0
        for pattern, ntype, label in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                nid = f"node_{id_offset + i + 1}"
                node = ReasoningNode(
                    id=nid,
                    label=label,
                    content=match.group(1).strip(),
                    node_type=ntype,
                    dependencies=[prev_id] if prev_id else [],
                )
                nodes.append(node)
                prev_id = nid
                i += 1
        if nodes:
            return nodes

        # 3. Fallback: single node with raw text
        return [
            ReasoningNode(
                id=f"node_{id_offset + 1}",
                label="Analysis",
                content=response[:300] if response else "No response generated.",
                node_type=NodeType.REASONING,
                dependencies=[],
            )
        ]

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    def _build_system_generate(self) -> str:
        steps = self.cot_steps
        example_nodes = []
        if steps >= 1:
            example_nodes.append(
                '{"label": "Fact", "content": "A key factual premise", "node_type": "fact", "confidence": 0.95}'
            )
        for i in range(steps - 2):
            example_nodes.append(
                f'{{"label": "Analysis {i + 1}", "content": "Mechanism or derivation step {i + 1}",'
                f' "node_type": "reasoning", "confidence": 0.88}}'
            )
        if steps >= 2:
            example_nodes.append(
                '{"label": "Conclusion", "content": "Final answer synthesising the above",'
                ' "node_type": "conclusion", "confidence": 0.90}'
            )
        nodes_json = ",\n    ".join(example_nodes)

        return f"""\
You are a chain-of-thought reasoning assistant. Break the question into exactly {steps} steps.

Return ONLY valid JSON — no prose, no markdown fences:
{{
  "nodes": [
    {nodes_json}
  ]
}}

Include a "confidence" field (0.0–1.0) for each node, reflecting how well-supported the claim is.

Label guidelines:
- Science/nature: Fact / Physics / Conclusion
- Math/calculation: Setup / Computation / Answer
- General: Fact / Analysis / Conclusion"""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_cot(self, prompt: str) -> ReasoningGraph:
        """Generate a chain-of-thought ReasoningGraph for *prompt*."""
        messages = [
            {"role": "system", "content": self._build_system_generate()},
            {"role": "user", "content": prompt},
        ]
        t0 = time.monotonic()
        response = self._call_llm(messages)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
        nodes = self._parse_nodes(response)
        return ReasoningGraph(nodes=nodes, prompt=prompt, generation_time_ms=elapsed_ms)

    def recalculate_from_node(self, graph: ReasoningGraph, edited_node_id: str) -> ReasoningGraph:
        """Recalculate all nodes downstream of *edited_node_id*."""
        edited_idx = graph.get_node_index(edited_node_id)
        if edited_idx == -1:
            return graph

        kept = graph.nodes[: edited_idx + 1]
        downstream_count = len(graph.nodes) - edited_idx - 1

        # Snapshot before modifying so undo can revert the recalculation
        graph.snapshot()

        if downstream_count == 0:
            graph.version += 1
            return graph

        if self.mode == "mock":
            raw = self._mock_recalculate(graph, edited_node_id)
        else:
            context = "\n".join(f"[{n.label}] {n.content}" for n in kept)
            system = (
                f"The original question: {graph.prompt}\n\n"
                f"Reasoning so far (may include edits):\n{context}\n\n"
                f"Generate {downstream_count} more step(s) to complete the chain.\n"
                "Return ONLY valid JSON:\n"
                '{"nodes": [{"label": "...", "content": "...", "node_type": "reasoning_or_conclusion", "confidence": 0.9}]}'
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": "Continue the reasoning chain."},
            ]
            raw = self._call_llm(messages)

        new_nodes = self._parse_nodes(raw, id_offset=edited_idx + 1)

        # Preserve original IDs / labels / types for structural consistency
        originals = graph.nodes[edited_idx + 1:]
        for orig, new in zip(originals, new_nodes):
            new.id = orig.id
            new.label = orig.label
            new.node_type = orig.node_type
            new.dependencies = orig.dependencies

        graph.nodes = kept + new_nodes[:downstream_count]
        graph.version += 1
        return graph

    def batch_analyze(self, prompts: List[str]) -> List[ReasoningGraph]:
        """Generate CoT graphs for multiple prompts and return them as a list."""
        return [self.generate_cot(p) for p in prompts]
