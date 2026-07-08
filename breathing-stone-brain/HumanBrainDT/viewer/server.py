"""
viewer/server.py — FastAPI WebSocket server for HumanBrainDT viewer.
Pushes brain activity JSON to connected browsers every 100ms when active.
Also serves CSV tail and training log via REST for the bottom panel.
"""
from __future__ import annotations
import asyncio
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from HumanBrainDT import HumanBrain
from HumanBrainDT.regions.builder import _REGION_ORDER

app = FastAPI()

_brain: HumanBrain | None = None
_last_state: dict | None = None
_clients: list[WebSocket] = []

STATIC_DIR = Path(__file__).parent / "static"

# Paths for live CSV + training log (stone_bridge writes these)
_CSV_PATH  = Path(__file__).parent.parent.parent / "stone_session.csv"
_LOG_PATH  = Path(__file__).parent.parent.parent / "stone_session_training.jsonl"


def get_brain() -> HumanBrain:
    global _brain
    if _brain is None:
        _brain = HumanBrain()
    return _brain


def _brain_snapshot() -> dict:
    """Current region activity snapshot for viewer."""
    import math
    brain = get_brain()
    nodes = []
    for rid, r in brain.regions.items():
        raw = r.spike_count()
        steps = 200
        activity = math.tanh(raw / max(r.population.n_neurons * steps * 0.005, 1))
        nodes.append({
            "id":        rid,
            "label":     r.label,
            "parent":    r.parent,
            "spikes":    int(raw),
            "activity":  round(activity, 4),
            "n_neurons": r.population.n_neurons,
        })

    edges = []
    for si, src in enumerate(_REGION_ORDER):
        for di, dst in enumerate(_REGION_ORDER):
            w = float(brain.simulator.inter_W[si, di])
            if w > 0.05:
                edges.append({"src": src, "dst": dst, "weight": round(w, 3)})

    return {"nodes": nodes, "edges": edges, "ts": time.time()}


async def _broadcast(data: dict):
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    await ws.send_text(json.dumps(_brain_snapshot()))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "stimulate":
                text = msg.get("text", "")
                brain = get_brain()
                state = brain.stimulate(text)
                snap  = _brain_snapshot()
                snap["event"] = {
                    "type":         "stimulate",
                    "text":         text,
                    "experience":   state.experience,
                    "dominant":     state.dominant_region,
                    "total_spikes": state.total_spikes,
                    "motor": {
                        "approach":  state.motor_command.approach,
                        "avoid":     state.motor_command.avoid,
                        "withdraw":  state.motor_command.withdraw,
                        "engage":    state.motor_command.engage,
                    },
                    "ans": {
                        "sympathetic":     state.ans.sympathetic,
                        "parasympathetic": state.ans.parasympathetic,
                        "hrv_index":       state.ans.hrv_index,
                        "cortisol":        state.ans.cortisol,
                    },
                    "anxiety": {
                        "baseline":      state.anxiety.baseline,
                        "anticipatory":  state.anxiety.anticipatory,
                        "somatic":       state.anxiety.somatic,
                        "regulation":    state.anxiety.regulation,
                        "healing_index": state.anxiety.healing_index,
                    },
                    "sensory": {
                        "touch_light":     state.sensory.touch_light,
                        "touch_deep":      state.sensory.touch_deep,
                        "touch_vibration": state.sensory.touch_vibration,
                        "thermal_warm":    state.sensory.thermal_warm,
                        "thermal_cold":    state.sensory.thermal_cold,
                        "interoception":   state.sensory.interoception,
                        "rhythmic":        state.sensory.rhythmic,
                    },
                    "conduction_path": [
                        {"from": p.from_region, "to": p.to_region,
                         "channel": p.channel, "weight": p.weight,
                         "latency_ms": p.latency_ms}
                        for p in state.conduction_path
                    ],
                }
                await _broadcast(snap)
    except WebSocketDisconnect:
        if ws in _clients:
            _clients.remove(ws)


# ── REST: last N rows of CSV ──────────────────────────────────────────────────
@app.get("/api/csv")
async def get_csv_tail(n: int = 60, path: str = ""):
    """Return last N rows of stone_session.csv as JSON array."""
    target = Path(path) if path else _CSV_PATH
    if not target.exists():
        return JSONResponse({"rows": [], "fields": [], "exists": False})
    try:
        with open(target, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        tail = rows[-n:] if len(rows) > n else rows
        fields = list(tail[0].keys()) if tail else []
        # cast numerics
        typed = []
        for row in tail:
            typed.append({k: _try_float(v) for k, v in row.items()})
        return JSONResponse({"rows": typed, "fields": fields,
                             "total": len(rows), "exists": True})
    except Exception as e:
        return JSONResponse({"error": str(e), "exists": True, "rows": []})


# ── REST: last N lines of training log ───────────────────────────────────────
@app.get("/api/log")
async def get_log_tail(n: int = 60, path: str = ""):
    """Return last N records of training JSONL as JSON array."""
    target = Path(path) if path else _LOG_PATH
    if not target.exists():
        return JSONResponse({"records": [], "exists": False})
    try:
        with open(target, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        tail = lines[-n:]
        records = [json.loads(l) for l in tail]
        return JSONResponse({"records": records, "total": len(lines), "exists": True})
    except Exception as e:
        return JSONResponse({"error": str(e), "exists": True, "records": []})


def _try_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    print("HumanBrainDT Viewer -> http://localhost:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
