"""
mcp/brain_mcp.py — MCP server for HumanBrainDT.
8 tools: stimulate, read, observe, inject_memory,
         modify_weight, save_state, load_state, visualize
"""
from __future__ import annotations
import os
import sys
import json
import time
import pickle

# Add both chimera root (for mcp package) and its parent if needed
_CHIMERA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CHIMERA_ROOT not in sys.path:
    sys.path.insert(0, _CHIMERA_ROOT)

from mcp.server.fastmcp import FastMCP
from HumanBrainDT import HumanBrain

mcp = FastMCP("human-brain")

_BASE  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # chimera root
_brain: HumanBrain | None = None


def _get_brain() -> HumanBrain:
    global _brain
    if _brain is None:
        _brain = HumanBrain()
    return _brain


# ── 1. stimulate ──────────────────────────────────────
@mcp.tool()
def stimulate(text: str) -> str:
    """
    Send natural-language input to the human brain simulation.
    Runs 200 LIF timesteps across all 10 regions and returns
    motor commands, dominant region, and first-person experience.

    Args:
        text: Any input (English/Korean/Chinese/Japanese).
              Examples: "danger", "remember", "run", "see light"
    """
    brain = _get_brain()
    state = brain.stimulate(text)
    return json.dumps(brain.state_to_dict(state), ensure_ascii=False, indent=2)


# ── 2. read ───────────────────────────────────────────
@mcp.tool()
def read(region: str) -> str:
    """
    Read the current activity state of a specific brain region.

    Args:
        region: Region ID. One of: frontal_cortex, parietal_cortex,
                temporal_cortex, occipital_cortex, basal_ganglia,
                thalamus, hippocampus, amygdala, cerebellum, brainstem
    """
    brain = _get_brain()
    if region not in brain.regions:
        return json.dumps({"error": f"Unknown region: {region}. "
                           f"Valid: {list(brain.regions.keys())}"})
    r = brain.regions[region]
    pop = r.population
    result = {
        "region":    region,
        "label":     r.label,
        "n_neurons": pop.n_neurons,
        "channels":  list(pop.channel_idx.keys()),
        "fwd_neurons":  len(pop.fwd_idx),
        "back_neurons": len(pop.back_idx),
        "current_spikes": r.spike_count(),
    }
    if r.vectors is not None:
        result["activity_sum"] = float(r.vectors.activity.sum())
        result["memory_norm"]  = float(
            (r.vectors.memory ** 2).sum() ** 0.5)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── 3. observe ────────────────────────────────────────
@mcp.tool()
def observe(text: str, n_steps: int = 5) -> str:
    """
    Run the same stimulus multiple times and return activity over time.
    Shows how the brain responds repeatedly to the same input.

    Args:
        text:    Input stimulus text
        n_steps: Number of simulation runs (1–20, default 5)
    """
    brain  = _get_brain()
    n_steps = max(1, min(20, n_steps))
    timeline = []
    for i in range(n_steps):
        state = brain.stimulate(text)
        timeline.append({
            "step":            i + 1,
            "total_spikes":    state.total_spikes,
            "dominant_region": state.dominant_region,
            "experience":      state.experience,
            "motor":           {
                "approach": state.motor_command.approach,
                "avoid":    state.motor_command.avoid,
                "withdraw": state.motor_command.withdraw,
            },
        })
    return json.dumps({"stimulus": text, "timeline": timeline},
                      ensure_ascii=False, indent=2)


# ── 4. inject_memory ─────────────────────────────────
@mcp.tool()
def inject_memory(region: str, content: str,
                  strength: float = 0.5) -> str:
    """
    Inject a memory trace into a specific brain region.
    Encodes content as a hash-based vector and accumulates
    into the region's memory vector.

    Args:
        region:   Target region ID
        content:  Text content to encode as memory
        strength: Memory injection strength 0.0–1.0 (default 0.5)
    """
    import hashlib
    import numpy as np

    brain = _get_brain()
    if region not in brain.regions:
        return json.dumps({"error": f"Unknown region: {region}"})

    r = brain.regions[region]
    if r.vectors is None:
        return json.dumps({"error": "Region has no vector store"})

    # Deterministic pseudo-embedding from content hash
    h = hashlib.sha256(content.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng  = np.random.default_rng(seed)
    vec  = rng.standard_normal(512).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)

    strength = max(0.0, min(1.0, strength))
    r.vectors.memory = (r.vectors.memory * 0.9 +
                        vec * strength * 0.1).astype(np.float32)
    norm = float(np.linalg.norm(r.vectors.memory))

    return json.dumps({
        "region":   region,
        "content":  content,
        "strength": strength,
        "memory_norm_after": round(norm, 4),
    }, ensure_ascii=False, indent=2)


# ── 5. modify_weight ─────────────────────────────────
@mcp.tool()
def modify_weight(src_region: str, dst_region: str,
                  delta: float) -> str:
    """
    Modify the inter-region connection weight between two regions.
    Positive delta strengthens, negative weakens the connection.

    Args:
        src_region: Source region ID
        dst_region: Destination region ID
        delta:      Change in weight (-1.0 to 1.0)
    """
    import numpy as np
    from HumanBrainDT.regions.builder import _REGION_ORDER

    brain = _get_brain()
    if src_region not in brain.regions:
        return json.dumps({"error": f"Unknown src: {src_region}"})
    if dst_region not in brain.regions:
        return json.dumps({"error": f"Unknown dst: {dst_region}"})

    si = _REGION_ORDER.index(src_region)
    di = _REGION_ORDER.index(dst_region)
    old = float(brain.simulator.inter_W[si, di])
    brain.simulator.inter_W[si, di] = float(
        np.clip(old + delta, 0.0, 1.0))
    new = float(brain.simulator.inter_W[si, di])

    return json.dumps({
        "src": src_region, "dst": dst_region,
        "weight_before": round(old, 4),
        "weight_after":  round(new, 4),
        "delta":         delta,
    }, ensure_ascii=False, indent=2)


# ── 6. save_state ─────────────────────────────────────
@mcp.tool()
def save_state(path: str = "") -> str:
    """
    Save the current brain state (memory vectors, inter-region weights)
    to disk.

    Args:
        path: File path to save to. Defaults to brain_state.pkl
              in the HumanBrainDT directory.
    """
    import numpy as np

    brain = _get_brain()
    if not path:
        path = os.path.join(_BASE, "HumanBrainDT", "brain_state.pkl")

    payload = {
        "inter_W": brain.simulator.inter_W,
        "memories": {
            rid: r.vectors.memory.copy()
            for rid, r in brain.regions.items()
            if r.vectors is not None
        },
        "timestamp": time.time(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    return json.dumps({
        "saved_to": path,
        "regions":  list(payload["memories"].keys()),
        "timestamp": payload["timestamp"],
    }, ensure_ascii=False, indent=2)


# ── 7. load_state ─────────────────────────────────────
@mcp.tool()
def load_state(path: str = "") -> str:
    """
    Load a previously saved brain state from disk.

    Args:
        path: File path to load from. Defaults to brain_state.pkl.
    """
    brain = _get_brain()
    if not path:
        path = os.path.join(_BASE, "HumanBrainDT", "brain_state.pkl")

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})

    with open(path, "rb") as f:
        payload = pickle.load(f)

    brain.simulator.inter_W = payload["inter_W"]
    for rid, mem in payload.get("memories", {}).items():
        if rid in brain.regions and brain.regions[rid].vectors is not None:
            brain.regions[rid].vectors.memory = mem

    return json.dumps({
        "loaded_from": path,
        "regions_restored": list(payload.get("memories", {}).keys()),
        "saved_at": payload.get("timestamp"),
    }, ensure_ascii=False, indent=2)


# ── 8. visualize ──────────────────────────────────────
@mcp.tool()
def visualize(region: str = "") -> str:
    """
    Return activity data for all regions (or one region) formatted
    for visualization. Includes spike counts, motor outputs, and
    inter-region connectivity weights.

    Args:
        region: Optional region ID to focus on. Empty = all regions.
    """
    import numpy as np
    from HumanBrainDT.regions.builder import _REGION_ORDER

    brain = _get_brain()

    nodes = []
    for rid, r in brain.regions.items():
        spikes = r.spike_count()
        nodes.append({
            "id":      rid,
            "label":   r.label,
            "parent":  r.parent,
            "spikes":  spikes,
            "n_neurons": r.population.n_neurons,
            "activity": round(spikes / max(r.population.n_neurons, 1), 4),
            "memory_norm": round(
                float(np.linalg.norm(r.vectors.memory))
                if r.vectors is not None else 0.0, 4),
        })

    edges = []
    for si, src in enumerate(_REGION_ORDER):
        for di, dst in enumerate(_REGION_ORDER):
            w = float(brain.simulator.inter_W[si, di])
            if w > 0.1:
                edges.append({"src": src, "dst": dst, "weight": round(w, 3)})

    result = {"nodes": nodes, "edges": edges}
    if region:
        if region not in brain.regions:
            return json.dumps({"error": f"Unknown region: {region}"})
        node = next(n for n in nodes if n["id"] == region)
        region_edges = [e for e in edges
                        if e["src"] == region or e["dst"] == region]
        result = {"focus": node, "connections": region_edges}

    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
