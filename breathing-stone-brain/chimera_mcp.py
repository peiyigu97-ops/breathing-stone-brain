"""
chimera_mcp.py — MCP server for CHIMERA
Exposes the connectome brain as MCP tools, no MuJoCo viewer.
"""

import os
import sys
import json

# Add chimera directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MCP_CMD_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_cmd.txt")

def _send_cmd(text: str):
    """Write command text to mcp_cmd.txt for the viewer to pick up."""
    tmp = _MCP_CMD_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, _MCP_CMD_FILE)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("chimera")

# ── Paths ─────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
CONNECTOME_JSON = os.path.join(_BASE, "chimera", "connectome", "real_neurons.json")
CONNECTOME_NPY  = os.path.join(_BASE, "chimera", "connectome", "real_weight_matrix.npy")
FALLBACK_JSON   = os.path.join(_BASE, "data", "fallback_neurons.json")
FALLBACK_NPY    = os.path.join(_BASE, "data", "fallback_weight_matrix.npy")

# ── Lazy-load brain & voice (loaded once on first call) ───
_brain = None
_voice = None
_parser = None


def _get_brain():
    global _brain
    if _brain is not None:
        return _brain

    import numpy as np
    import json as _json

    from chimera_app import Brain, ensure_fallback, Voice, make_parser

    ensure_fallback()

    if os.path.exists(CONNECTOME_NPY) and os.path.exists(CONNECTOME_JSON):
        npy, jsn = CONNECTOME_NPY, CONNECTOME_JSON
    else:
        npy, jsn = FALLBACK_NPY, FALLBACK_JSON

    _brain = Brain(npy, jsn)

    global _voice, _parser
    MODEL_GGUF = os.path.join(_BASE, "chimera", "models",
                               "Qwen2.5-0.5B-Instruct.Q4_K_M.gguf")
    _voice = Voice(MODEL_GGUF)
    _parser = make_parser(_voice.llm)

    return _brain


# ── Tools ─────────────────────────────────────────────

@mcp.tool()
def stimulate(text: str) -> str:
    """
    Send text input to the CHIMERA brain. The connectome simulates 200
    steps of LIF neuron firing and returns motor signals plus a
    first-person experience sentence.

    Args:
        text: Any natural-language input (English, Korean, Japanese, Chinese).
              Examples: "danger", "fly", "food smell", "위험해", "뛰어"
    """
    brain = _get_brain()
    stim = _parser(text)
    sig  = brain.run(stim)
    word = _voice.speak(stim, sig, input_text=text)

    _send_cmd(text)

    result = {
        "input": text,
        "sensory_channels": stim,
        "motor_signals": {
            "forward":    round(sig["forward"],    3),
            "backward":   round(sig["backward"],   3),
            "turn_left":  round(sig["turn_left"],  3),
            "turn_right": round(sig["turn_right"], 3),
            "wing":       round(sig["wing"],        3),
            "curl":       round(sig["curl"],        3),
            "eat":        round(sig["eat"],         3),
            "tremble":    round(sig["tremble"],     3),
        },
        "total_spikes": sig["_total_spikes"],
        "experience":   word,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def stimulate_direct(
    touch_front:  float = 0.0,
    touch_back:   float = 0.0,
    nociception:  float = 0.0,
    chemical:     float = 0.0,
    olfactory:    float = 0.0,
    visual:       float = 0.0,
) -> str:
    """
    Directly inject sensory channel values (0.0–1.0) into the brain,
    bypassing the text parser. Useful for precise stimulus control.

    Args:
        touch_front:  Front touch / forward locomotion signal (0-1)
        touch_back:   Rear touch / backward signal (0-1)
        nociception:  Pain / danger signal (0-1)
        chemical:     Food / gustatory signal (0-1)
        olfactory:    Smell / curiosity signal (0-1)
        visual:       Visual signal (0-1)
    """
    brain = _get_brain()
    stim = {
        k: v for k, v in {
            "touch_front": touch_front,
            "touch_back":  touch_back,
            "nociception": nociception,
            "chemical":    chemical,
            "olfactory":   olfactory,
            "visual":      visual,
        }.items() if v > 0
    }
    if not stim:
        stim = {"olfactory": 0.05}

    sig  = brain.run(stim)
    word = _voice.speak(stim, sig)

    result = {
        "sensory_channels": stim,
        "motor_signals": {
            "forward":    round(sig["forward"],    3),
            "backward":   round(sig["backward"],   3),
            "turn_left":  round(sig["turn_left"],  3),
            "turn_right": round(sig["turn_right"], 3),
            "wing":       round(sig["wing"],        3),
            "curl":       round(sig["curl"],        3),
            "eat":        round(sig["eat"],         3),
            "tremble":    round(sig["tremble"],     3),
        },
        "total_spikes": sig["_total_spikes"],
        "experience":   word,
        "type_fires":   sig.get("_type_fires", {}),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def brain_status() -> str:
    """
    Return information about the loaded connectome: neuron count,
    synapse count, sensory channels, and motor neuron counts.
    """
    brain = _get_brain()
    import numpy as np

    channels = {ch: len(idxs) for ch, idxs in brain.channel_idx.items()}
    result = {
        "neurons":      brain.N,
        "synapses":     int(np.count_nonzero(brain.W)),
        "channels":     channels,
        "fwd_neurons":  len(brain.fwd_idx),
        "back_neurons": len(brain.back_idx),
        "neuron_types": list(set(brain.type_map.values())),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
