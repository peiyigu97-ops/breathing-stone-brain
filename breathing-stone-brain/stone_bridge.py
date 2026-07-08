"""
stone_bridge.py
===============
Breathing Stone ↔ HumanBrainDT bidirectional bridge.

Architecture:
  ESP32 (stone) ──BLE/Serial──▶  stone_bridge.py  ──▶  HumanBrainDT LIF sim
                                                   ◀──  intervention command

This module handles:
  1. Polyvagal 3-state inference from sensor data
  2. Sensor → sensory channel mapping
  3. LIF simulation via HumanBrainDT
  4. BrainState → intervention parameters (gas pump, temp, vibration)
  5. TGAM CSV logging
  6. WebSocket push to HumanBrainDT viewer

Run modes:
  python stone_bridge.py --demo          # synthetic data loop
  python stone_bridge.py --serial COM3   # live ESP32 via serial JSON
  python stone_bridge.py --ble XX:XX:..  # live ESP32 via BLE (needs bleak)

ESP32 expected JSON (sent every ~1s):
  {"skin_temp": 33.2, "grip_n": 18.5, "contact": 1, "rhythm_phase": 0.4}

Bridge sends back JSON command to ESP32:
  {"mode": "calm", "temp_c": 34.0, "rhythm_bpm": 6.0, "vibrate": 0}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path

# Force UTF-8 on Windows consoles (GBK default breaks emoji in HumanBrainDT prints)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from typing import Optional

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from HumanBrainDT import HumanBrain
from tgam_csv_generator import (
    TGAMWriter, brainstate_to_tgam_row,
    esp32_to_stim, RawEEGSynth
)


# ─────────────────────────────────────────────────────────────────────────────
# Polyvagal state inference
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonalBaseline:
    """Running personal baseline (7-day window equivalent, decays slowly)."""
    skin_temp_mean:  float = 33.5   # °C default
    skin_temp_std:   float = 0.5
    grip_mean:       float = 15.0   # Newtons
    grip_std:        float = 3.0
    n_samples:       int   = 0
    _alpha:          float = 0.02   # EWM decay (≈ 50-sample window)

    def update(self, skin_temp: float, grip_n: float):
        if self.n_samples == 0:
            self.skin_temp_mean = skin_temp
            self.grip_mean      = grip_n
        else:
            a = self._alpha
            self.skin_temp_mean = (1-a)*self.skin_temp_mean + a*skin_temp
            self.grip_mean      = (1-a)*self.grip_mean      + a*grip_n
            # Running std approximation
            self.skin_temp_std = max(0.2, (1-a)*self.skin_temp_std +
                                     a*abs(skin_temp - self.skin_temp_mean))
            self.grip_std      = max(1.0, (1-a)*self.grip_std +
                                     a*abs(grip_n - self.grip_mean))
        self.n_samples += 1

    def z_temp(self, t: float) -> float:
        """Z-score: positive = warmer than baseline."""
        return (t - self.skin_temp_mean) / max(self.skin_temp_std, 0.1)

    def z_grip(self, g: float) -> float:
        """Z-score: positive = gripping harder than baseline."""
        return (g - self.grip_mean) / max(self.grip_std, 0.5)


def infer_polyvagal_state(sensor: dict,
                           baseline: PersonalBaseline) -> str:
    """
    Infer Polyvagal state from sensor readings.

    Returns one of:
      "ventral"     — calm, ventral vagal (normal)
      "sympathetic" — fight/flight, anxious
      "dorsal"      — freeze/shutdown, low energy

    Decision rule (Section 11.2 of breathing stone implementation doc):
      cool skin (z_temp < -1) + high grip (z_grip > +1)  → sympathetic
      cool skin (z_temp < -1) + low  grip (z_grip < -0.5) → dorsal vagal
      otherwise → ventral vagal
    """
    zt = baseline.z_temp(sensor.get("skin_temp", baseline.skin_temp_mean))
    zg = baseline.z_grip(sensor.get("grip_n",    baseline.grip_mean))

    if zt < -1.0 and zg > 1.0:
        return "sympathetic"
    if zt < -1.0 and zg < -0.5:
        return "dorsal"
    return "ventral"


# ─────────────────────────────────────────────────────────────────────────────
# Intervention parameter generator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InterventionCmd:
    """Command sent back to ESP32."""
    mode:         str   = "default"   # "calm" | "activate" | "default"
    temp_c:       float = 35.0        # target stone surface temperature
    inhale_s:     float = 4.0         # inhale phase duration (seconds)
    exhale_s:     float = 6.0         # exhale phase duration (seconds)
    cycles:       int   = 4           # number of breath cycles per session
    vibrate:      int   = 0           # 1 = pulse wake vibration now
    trigger_now:  bool  = False       # True = fire intervention immediately

    def to_esp32_json(self) -> str:
        return json.dumps({
            "mode":      self.mode,
            "temp_c":    round(self.temp_c, 1),
            "inhale_s":  self.inhale_s,
            "exhale_s":  self.exhale_s,
            "cycles":    self.cycles,
            "vibrate":   self.vibrate,
            "trigger":   int(self.trigger_now),
        })


def state_to_intervention(pv_state: str,
                           brain_state,
                           baseline: PersonalBaseline) -> InterventionCmd:
    """
    Map Polyvagal state + BrainState → intervention parameters.

    Calm mode (sympathetic):   exhale dominant, cool temp, 5 cycles
    Activate mode (dorsal):    inhale dominant, warm temp, 3 cycles
    Default (ventral):         standard 4/6, maintain temp
    """
    ans     = brain_state.ans
    anxiety = brain_state.anxiety

    if pv_state == "sympathetic":
        # High sympathetic: extend exhale, cool slightly (vagal cold reflex)
        return InterventionCmd(
            mode      = "calm",
            temp_c    = 34.0,
            inhale_s  = 4.0,
            exhale_s  = 8.0,     # extended exhale
            cycles    = 5,
            vibrate   = 0,
            trigger_now = anxiety.baseline > 0.4,
        )
    elif pv_state == "dorsal":
        # Shutdown: warm up, shorter cycles, stimulate
        return InterventionCmd(
            mode      = "activate",
            temp_c    = 37.0,    # warmer = social safety signal
            inhale_s  = 5.0,
            exhale_s  = 4.0,     # inhale dominant (mild sympathetic activation)
            cycles    = 3,
            vibrate   = 1,       # active pulse to break freeze
            trigger_now = True,
        )
    else:
        # Ventral vagal: maintenance
        return InterventionCmd(
            mode      = "default",
            temp_c    = 35.0,
            inhale_s  = 4.0,
            exhale_s  = 6.0,
            cycles    = 4,
            vibrate   = 0,
            trigger_now = False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Use-frequency tracker (detects when user initiates more than usual)
# ─────────────────────────────────────────────────────────────────────────────

class UseFrequencyTracker:
    """
    Tracks how often the user picks up the stone spontaneously.
    High spontaneous use = anxiety signal even before sensors spike.
    """
    def __init__(self, window_s: float = 300):
        self._events: deque = deque()   # timestamps of pick-up events
        self._window = window_s         # 5-minute window

    def record_pickup(self):
        now = time.time()
        self._events.append(now)
        # Trim old events
        while self._events and now - self._events[0] > self._window:
            self._events.popleft()

    @property
    def rate_per_min(self) -> float:
        now = time.time()
        recent = sum(1 for t in self._events if now - t <= 60)
        return float(recent)

    @property
    def normalised(self) -> float:
        """0-1 normalised frequency (>3/min = 1.0)."""
        return min(self.rate_per_min / 3.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Core bridge class
# ─────────────────────────────────────────────────────────────────────────────

class StoneBridge:
    """
    Main bridge: sensor data in → brain simulation → intervention out.
    Thread-safe; can be driven by serial, BLE, or synthetic data.
    """

    def __init__(self, csv_path: Optional[str] = None):
        print("Loading HumanBrainDT...")
        self.brain    = HumanBrain()
        self.baseline = PersonalBaseline()
        self.freq     = UseFrequencyTracker()
        self.synth    = RawEEGSynth()
        self._lock    = threading.Lock()
        self._last_state  = None
        self._last_pv     = "ventral"
        self._last_cmd    = InterventionCmd()

        self.writer = TGAMWriter(csv_path) if csv_path else None
        self.csv_path = csv_path
        if csv_path:
            print(f"📄 Logging → {csv_path}")

        # Training log (JSONL, one record per process() call)
        self._log_path = csv_path.replace(".csv", "_training.jsonl") if csv_path else None
        self._step = 0

        # WebSocket broadcast queue (filled by process(), drained by viewer)
        self._ws_queue: deque = deque(maxlen=10)

    def process(self, sensor: dict) -> InterventionCmd:
        """
        Main processing loop: one sensor reading → one intervention command.
        Call this every ~1s from serial/BLE/demo thread.
        """
        # Track spontaneous pickup
        prev_contact = getattr(self, "_prev_contact", 0)
        if sensor.get("contact", 0) > 0.5 and prev_contact < 0.5:
            self.freq.record_pickup()
        self._prev_contact = sensor.get("contact", 0)

        # Update personal baseline
        self.baseline.update(
            skin_temp = sensor.get("skin_temp", self.baseline.skin_temp_mean),
            grip_n    = sensor.get("grip_n",    self.baseline.grip_mean),
        )

        # Infer Polyvagal state
        pv_state = infer_polyvagal_state(sensor, self.baseline)

        # Build enriched sensor dict for esp32_to_stim
        enriched = dict(sensor)
        enriched["use_frequency"] = self.freq.normalised

        # Normalise raw values → 0-1
        temp_drop = max(0.0,
            (self.baseline.skin_temp_mean - sensor.get("skin_temp", self.baseline.skin_temp_mean))
            / max(self.baseline.skin_temp_std * 3, 0.5))
        grip_norm = max(0.0, min(1.0,
            sensor.get("grip_n", 0) / 40.0))  # 40N = max grip

        enriched["skin_temp_delta"] = min(temp_drop, 1.0)
        enriched["grip_force"]      = grip_norm
        enriched["grip_force_delta"]= max(0, self.baseline.z_grip(
            sensor.get("grip_n", self.baseline.grip_mean)) / 3)
        enriched["session_active"]  = float(sensor.get("contact", 0) > 0.5)

        # Map to CHIMERA sensory channels
        stim = esp32_to_stim(enriched)

        # Run LIF simulation
        brain_state = self.brain.stimulate_direct(stim)

        # Generate intervention command
        cmd = state_to_intervention(pv_state, brain_state, self.baseline)

        # Log to CSV
        raw_sample = 0
        if self.writer:
            raw_sample = self.synth.next_sample(
                anxiety_level = brain_state.anxiety.baseline,
                healing_level = brain_state.anxiety.healing_index,
            )
            self.writer.write(brainstate_to_tgam_row(brain_state, raw_sample))

        # Training log (JSONL)
        self._write_training_log(sensor, pv_state, brain_state, cmd, raw_sample)
        self._step += 1

        # Store for viewer broadcast
        snap = self._build_ws_payload(sensor, brain_state, pv_state, cmd)
        self._ws_queue.append(snap)

        with self._lock:
            self._last_state = brain_state
            self._last_pv    = pv_state
            self._last_cmd   = cmd

        self._print_status(sensor, pv_state, brain_state, cmd)
        return cmd

    def _write_training_log(self, sensor, pv_state, state, cmd, raw_eeg):
        """Append one JSONL record to training log."""
        if not self._log_path:
            return
        record = {
            "step":      self._step,
            "ts":        round(state.timestamp, 3),
            "pv_state":  pv_state,
            # raw sensor
            "skin_temp":      round(sensor.get("skin_temp", 0), 2),
            "grip_n":         round(sensor.get("grip_n", 0), 1),
            "contact":        int(sensor.get("contact", 0) > 0.5),
            # personal baseline
            "bl_temp":        round(self.baseline.skin_temp_mean, 2),
            "bl_grip":        round(self.baseline.grip_mean, 1),
            # CHIMERA output
            "healing_index":  round(state.anxiety.healing_index, 3),
            "hrv_index":      round(state.ans.hrv_index, 3),
            "cortisol":       round(state.ans.cortisol, 3),
            "sympathetic":    round(state.ans.sympathetic, 3),
            "parasympathetic":round(state.ans.parasympathetic, 3),
            "anx_baseline":   round(state.anxiety.baseline, 3),
            "anx_anticipatory":round(state.anxiety.anticipatory, 3),
            "regulation":     round(state.anxiety.regulation, 3),
            "dominant":       state.dominant_region,
            "total_spikes":   state.total_spikes,
            # TGAM fields (mirrored)
            "attention":   int(np.clip(state.motor_command.approach * 100, 0, 100)),
            "meditation":  int(np.clip(
                state.ans.parasympathetic * (1 - state.ans.sympathetic * 0.6) * 100, 0, 100)),
            "raw_eeg":     raw_eeg,
            # intervention
            "interv_mode":    cmd.mode,
            "interv_temp_c":  cmd.temp_c,
            "interv_inhale_s":cmd.inhale_s,
            "interv_exhale_s":cmd.exhale_s,
            "trigger_now":    int(cmd.trigger_now),
            # use freq
            "use_freq_per_min": round(self.freq.rate_per_min, 1),
        }
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _build_ws_payload(self, sensor, state, pv_state, cmd) -> dict:
        """Build JSON payload for HumanBrainDT viewer WebSocket."""
        nodes = []
        for rid, r in self.brain.regions.items():
            raw      = r.spike_count()
            activity = math.tanh(raw / max(r.population.n_neurons * 200 * 0.005, 1))
            nodes.append({"id": rid, "label": r.label, "activity": round(activity, 4)})

        return {
            "nodes": nodes,
            "ts":    time.time(),
            "event": {
                "type":         "stone",
                "experience":   state.experience,
                "dominant":     state.dominant_region,
                "total_spikes": state.total_spikes,
                "pv_state":     pv_state,
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
                    "touch_deep":      state.sensory.touch_deep,
                    "touch_vibration": state.sensory.touch_vibration,
                    "thermal_cold":    state.sensory.thermal_cold,
                    "rhythmic":        state.sensory.rhythmic,
                    "interoception":   state.sensory.interoception,
                },
                "intervention": asdict(cmd),
                "baseline": {
                    "skin_temp_mean": round(self.baseline.skin_temp_mean, 2),
                    "grip_mean":      round(self.baseline.grip_mean, 1),
                    "n_samples":      self.baseline.n_samples,
                },
                "use_freq_per_min": round(self.freq.rate_per_min, 1),
            },
        }

    def _print_status(self, sensor, pv, state, cmd):
        icons = {"ventral": "🟢", "sympathetic": "🔴", "dorsal": "🟡"}
        print(
            f"  {icons.get(pv,'⚪')} {pv:12s} | "
            f"T={sensor.get('skin_temp',0):.1f}°C  "
            f"G={sensor.get('grip_n',0):.0f}N | "
            f"heal={state.anxiety.healing_index:.2f}  "
            f"hrv={state.ans.hrv_index:.2f}  "
            f"cort={state.ans.cortisol:.2f} | "
            f"→ {cmd.mode} {cmd.temp_c:.0f}°C "
            f"{cmd.inhale_s:.0f}↑/{cmd.exhale_s:.0f}↓"
        )

    def close(self):
        if self.writer:
            self.writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(csv_path: str, n_steps: int = 120, interval_s: float = 1.0):
    """Synthetic scenario: calm → anxiety → stone use → healing."""
    bridge = StoneBridge(csv_path)

    # Seed baseline with 20 "normal" readings so z-scores are meaningful
    print("  Warming up baseline (20 steps)...")
    for _ in range(20):
        bridge.process({
            "skin_temp": float(np.clip(34.2 + np.random.normal(0, 0.3), 30, 38)),
            "grip_n":    float(np.clip(14.0 + np.random.normal(0, 1.5), 0, 40)),
            "contact": 0.0, "rhythm_phase": 0.0,
        })

    PHASES = [
        # (steps, sensor_template, label)
        # Phase 1: sympathetic activation — cool skin + high grip
        (30,  {"skin_temp": 31.0, "grip_n": 26.0, "contact": 0.0,
               "rhythm_phase": 0.0}, "Anxiety (sympathetic)"),
        # Phase 2: stone active — warming + grip moderate + rhythmic breath
        (60,  {"skin_temp": 33.0, "grip_n": 18.0, "contact": 1.0,
               "rhythm_phase": 0.5}, "Stone active (healing)"),
        # Phase 3: post-session — skin warmed up, low grip
        (30,  {"skin_temp": 34.5, "grip_n": 12.0, "contact": 0.0,
               "rhythm_phase": 0.0}, "Post-session (recovered)"),
    ]

    print(f"\n{'─'*70}")
    print(f"  STONE BRIDGE — DEMO  ({n_steps} steps, {interval_s}s/step)")
    print(f"{'─'*70}")

    step = 0
    for phase_len, template, label in PHASES:
        print(f"\n▶ {label}")
        for _ in range(phase_len):
            if step >= n_steps:
                break
            sensor = {k: float(np.clip(v + np.random.normal(0, 0.15), 0, 40))
                      for k, v in template.items()}
            sensor["rhythm_phase"] = (step * 0.1) % 1.0
            bridge.process(sensor)
            step += 1
            time.sleep(interval_s)

    bridge.close()
    print(f"\n✓ Done. CSV → {csv_path}")


def run_serial(port: str, baud: int, csv_path: str):
    """Live ESP32 serial bridge."""
    try:
        import serial
    except ImportError:
        sys.exit("pip install pyserial")

    bridge = StoneBridge(csv_path)
    ser    = serial.Serial(port, baud, timeout=1.0)

    print(f"\n🔌 Serial bridge: {port} @ {baud} baud")
    print("Expected ESP32 JSON: {'skin_temp':33.5,'grip_n':18.0,'contact':1,'rhythm_phase':0.4}")
    print("Press Ctrl-C to stop.\n")

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                sensor = json.loads(line)
            except json.JSONDecodeError:
                continue
            cmd = bridge.process(sensor)
            ser.write((cmd.to_esp32_json() + "\n").encode())
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        ser.close()


def run_ble(address: str, csv_path: str):
    """Live ESP32 BLE bridge (requires bleak)."""
    try:
        import bleak
    except ImportError:
        sys.exit("pip install bleak")

    # BLE characteristic UUIDs — match your ESP32 firmware
    SENSOR_UUID = "12345678-1234-5678-1234-56789abcdef0"
    CMD_UUID    = "12345678-1234-5678-1234-56789abcdef1"

    bridge = StoneBridge(csv_path)

    async def _run():
        from bleak import BleakClient
        async with BleakClient(address) as client:
            print(f"🔵 BLE connected: {address}")

            async def on_notify(_, data: bytearray):
                try:
                    sensor = json.loads(data.decode())
                except Exception:
                    return
                cmd = bridge.process(sensor)
                await client.write_gatt_char(
                    CMD_UUID, cmd.to_esp32_json().encode(), response=False)

            await client.start_notify(SENSOR_UUID, on_notify)
            print("Streaming... Ctrl-C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            await client.stop_notify(SENSOR_UUID)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Breathing Stone ↔ CHIMERA bridge")
    p.add_argument("--demo",   action="store_true", help="Synthetic demo")
    p.add_argument("--serial", metavar="PORT",      help="Serial port (ESP32)")
    p.add_argument("--ble",    metavar="ADDR",      help="BLE address (ESP32)")
    p.add_argument("--baud",   type=int, default=115200)
    p.add_argument("--out",    default="stone_session.csv")
    p.add_argument("--steps",  type=int, default=120)
    p.add_argument("--fast",   action="store_true",
                   help="Demo: no sleep between steps (benchmark mode)")
    a = p.parse_args()

    if a.demo:
        run_demo(a.out, a.steps, interval_s=0.0 if a.fast else 1.0)
    elif a.serial:
        run_serial(a.serial, a.baud, a.out)
    elif a.ble:
        run_ble(a.ble, a.out)
    else:
        p.print_help()
        print("\nExample:")
        print("  python stone_bridge.py --demo --steps 60 --fast")
        print("  python stone_bridge.py --serial COM3 --out live.csv")
        print("  python stone_bridge.py --ble AA:BB:CC:DD:EE:FF")
