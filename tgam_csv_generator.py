"""
tgam_csv_generator.py
=====================
Generates CSV files in NeuroSky TGAM (ThinkGear ASIC Module) output format.

TGAM output fields (NeuroSky official spec):
  timestamp        - Unix time (float, seconds)
  signal_quality   - 0=no signal, 200=poor, 1-100=good (lower=better contact)
  attention        - eSense attention    [0-100]
  meditation       - eSense meditation   [0-100]
  delta            - EEG band power [0.5-2.75 Hz]  raw int
  theta            - EEG band power [3.5-6.75 Hz]  raw int
  low_alpha        - EEG band power [7.5-9.25 Hz]  raw int
  high_alpha       - EEG band power [10-11.75 Hz]  raw int
  low_beta         - EEG band power [13-16.75 Hz]  raw int
  high_beta        - EEG band power [18-29.75 Hz]  raw int
  low_gamma        - EEG band power [31-39.75 Hz]  raw int
  mid_gamma        - EEG band power [41-49.75 Hz]  raw int
  raw_eeg          - Raw EEG sample [-2048 to 2047] at 512 Hz

Usage modes:
  1. Standalone: generate synthetic TGAM CSV from HumanBrainDT BrainState
  2. Bridge: receive ESP32 sensor data → map to CHIMERA stim → simulate → write CSV
  3. Replay: read existing CSV and feed rows into CHIMERA

Run standalone demo:
  python tgam_csv_generator.py --demo

Run bridge (ESP32 via serial):
  python tgam_csv_generator.py --serial /dev/ttyUSB0 --baud 115200
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── TGAM field spec ───────────────────────────────────────────────────────────
TGAM_FIELDS = [
    "timestamp",
    "signal_quality",   # 0 (best) – 200 (no contact)
    "attention",        # 0–100  eSense
    "meditation",       # 0–100  eSense
    "delta",            # raw band power integers
    "theta",
    "low_alpha",
    "high_alpha",
    "low_beta",
    "high_beta",
    "low_gamma",
    "mid_gamma",
    "raw_eeg",          # single raw sample per row, -2048..2047
]


# ── CHIMERA → TGAM mapping ────────────────────────────────────────────────────
def brainstate_to_tgam_row(state, raw_eeg_sample: int = 0) -> dict:
    """
    Map HumanBrainDT BrainState → one TGAM CSV row.

    Mapping rationale:
      attention  ← motor_command.approach (goal-directed engagement)
      meditation ← ans.parasympathetic * (1 - ans.sympathetic) * 100
      delta      ← anxiety.baseline (deep slow wave proxy: high anxiety = high delta)
      theta      ← anxiety.anticipatory (theta linked to rumination/memory)
      low_alpha  ← healing_index * 0.5  (alpha rises during calm)
      high_alpha ← ans.hrv_index (HRV coherence ↔ alpha peak)
      low_beta   ← ans.sympathetic (beta ↑ during arousal)
      high_beta  ← motor_command.avoid (avoidance/threat)
      low_gamma  ← motor_command.engage (cognitive engagement)
      mid_gamma  ← sensory.rhythmic (rhythmic entrainment ↔ gamma binding)
    """
    ans     = state.ans
    anxiety = state.anxiety
    motor   = state.motor_command
    sensory = state.sensory

    # eSense scores (0-100 integer)
    attention  = int(np.clip(motor.approach * 100, 0, 100))
    meditation = int(np.clip(
        ans.parasympathetic * (1 - ans.sympathetic * 0.6) * 100, 0, 100))

    # Signal quality: higher cortisol/anxiety → worse "contact" proxy
    # 0 = perfect, 200 = no signal; we use 0-50 range for a functioning device
    signal_quality = int(np.clip(anxiety.baseline * 50, 0, 50))

    # Band powers (scale 0–65535 int, typical TGAM range)
    SCALE = 65535
    delta     = int(np.clip(anxiety.baseline            * SCALE, 0, SCALE))
    theta     = int(np.clip(anxiety.anticipatory        * SCALE, 0, SCALE))
    low_alpha = int(np.clip(anxiety.healing_index * 0.5 * SCALE, 0, SCALE))
    high_alpha= int(np.clip(ans.hrv_index               * SCALE, 0, SCALE))
    low_beta  = int(np.clip(ans.sympathetic              * SCALE, 0, SCALE))
    high_beta = int(np.clip(motor.avoid                  * SCALE, 0, SCALE))
    low_gamma = int(np.clip(motor.engage                 * SCALE, 0, SCALE))
    mid_gamma = int(np.clip(sensory.rhythmic             * SCALE, 0, SCALE))

    return {
        "timestamp":     round(state.timestamp, 3),
        "signal_quality": signal_quality,
        "attention":      attention,
        "meditation":     meditation,
        "delta":          delta,
        "theta":          theta,
        "low_alpha":      low_alpha,
        "high_alpha":     high_alpha,
        "low_beta":       low_beta,
        "high_beta":      high_beta,
        "low_gamma":      low_gamma,
        "mid_gamma":      mid_gamma,
        "raw_eeg":        int(np.clip(raw_eeg_sample, -2048, 2047)),
    }


# ── ESP32 sensor → CHIMERA stim mapping ──────────────────────────────────────
def esp32_to_stim(sensor: dict) -> dict:
    """
    Map ESP32 Breathing Stone sensor data to HumanBrainDT sensory channels.

    Expected sensor dict keys (all floats 0.0–1.0 unless noted):
      skin_temp_delta   : normalised drop from baseline (0=normal, 1=max drop)
                          positive = hand cooler than baseline (sympathetic activation)
      grip_force        : normalised grip force (0=none, 1=max)
      grip_force_delta  : normalised delta from personal baseline
      use_frequency     : how often user initiates (0=rare, 1=very frequent)
      session_active    : 1 if stone is being held right now, 0 otherwise
      rhythm_phase      : float 0-1 indicating position in 4s/6s breath cycle

    Polyvagal state inference:
      sympathetic (anxiety)   : skin_temp↓ + grip↑
      dorsal vagal (shutdown) : skin_temp↓ + grip↓
      ventral vagal (calm)    : skin_temp≈baseline + grip low
    """
    s = {k: float(v) for k, v in sensor.items()}

    temp_drop   = s.get("skin_temp_delta",   0.0)  # 0→no drop, 1→max drop
    grip        = s.get("grip_force",         0.0)
    grip_delta  = s.get("grip_force_delta",   0.0)
    use_freq    = s.get("use_frequency",       0.0)
    active      = s.get("session_active",     0.0)
    rhythm      = s.get("rhythm_phase",        0.0)  # 0-1 breath phase

    stim: dict[str, float] = {}

    # ── Sympathetic activation (anxiety) ──────────────────
    # Cool skin + high grip = classic fight-or-flight
    sym_signal = temp_drop * 0.6 + grip_delta * 0.4
    if sym_signal > 0.25:
        stim["anxiety_anticipatory"] = np.clip(sym_signal * 0.9, 0, 1)
        stim["threat_input"]         = np.clip(sym_signal * 0.5, 0, 1)

    # ── Dorsal vagal (freeze/shutdown): cool + low grip ───
    dorsal = temp_drop * 0.5 - grip * 0.3
    if dorsal > 0.15 and grip < 0.3:
        stim["emotional_input"]  = np.clip(dorsal * 0.6, 0, 1)

    # ── Active stone use → healing channels ───────────────
    if active > 0.5:
        # Deep pressure from grip
        stim["touch_deep"]      = np.clip(grip * 0.9, 0, 1)
        # Touch vibration (stone breathing)
        stim["touch_vibration"] = np.clip(0.6, 0, 1)
        # Rhythmic channel: strongest at exhale phase (0.4-1.0)
        rhythm_strength = 0.5 + 0.5 * math.sin(rhythm * 2 * math.pi)
        stim["rhythmic"]        = np.clip(rhythm_strength * 0.85, 0, 1)
        # Interoception: holding stone + following breath = body awareness
        stim["interoception"]   = np.clip(0.55, 0, 1)

    # ── Temperature channel ────────────────────────────────
    # Stone at 34-36°C is slightly cool → cold channel activation (mild vagal)
    stim["thermal_cold"]    = np.clip(0.25, 0, 1)   # mild, always
    stim["thermal_neutral"] = np.clip(1.0 - temp_drop * 0.5, 0, 1)

    # ── Use frequency: high freq = persistent anxiety signal ──
    if use_freq > 0.6:
        stim["anxiety_anticipatory"] = max(
            stim.get("anxiety_anticipatory", 0), use_freq * 0.5)

    return stim


# ── Raw EEG synthesiser (plausible noise for CSV completeness) ────────────────
class RawEEGSynth:
    """
    Synthesise plausible raw EEG samples from brain state.
    Not meant to be accurate — provides non-zero raw_eeg column
    so downstream tools that expect it don't choke.
    """
    def __init__(self, sr=512):
        self.sr = sr
        self.t  = 0.0
        self.dt = 1.0 / sr

    def next_sample(self, anxiety_level: float = 0.0,
                    healing_level: float = 0.5) -> int:
        """One raw EEG sample."""
        self.t += self.dt

        # Alpha (10 Hz) dominant when calm
        alpha_amp = (1 - anxiety_level) * healing_level * 80
        alpha = alpha_amp * math.sin(2 * math.pi * 10 * self.t)

        # Beta (20 Hz) dominant when anxious
        beta_amp = anxiety_level * 60
        beta = beta_amp * math.sin(2 * math.pi * 20 * self.t)

        # Theta (6 Hz) during meditation/drowsy
        theta_amp = healing_level * 40
        theta = theta_amp * math.sin(2 * math.pi * 6 * self.t)

        # Noise floor
        noise = np.random.normal(0, 15)

        raw = alpha + beta + theta + noise
        return int(np.clip(raw, -2048, 2047))


# ── CSV writer ────────────────────────────────────────────────────────────────
class TGAMWriter:
    def __init__(self, path: str, append: bool = False):
        self.path   = path
        mode        = "a" if append and os.path.exists(path) else "w"
        self._f     = open(path, mode, newline="", encoding="utf-8")
        self._csv   = csv.DictWriter(self._f, fieldnames=TGAM_FIELDS)
        if mode == "w":
            self._csv.writeheader()
        self._lock  = threading.Lock()

    def write(self, row: dict):
        with self._lock:
            self._csv.writerow(row)
            self._f.flush()

    def close(self):
        self._f.close()


# ── CSV reader (replay) ───────────────────────────────────────────────────────
def read_tgam_csv(path: str):
    """Yield dicts from a TGAM CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k: float(v) if k != "timestamp" else float(v)
                   for k, v in row.items()}


# ── Bridge: Serial (ESP32) → CHIMERA → CSV ───────────────────────────────────
def run_serial_bridge(port: str, baud: int, out_csv: str):
    """
    Read JSON lines from ESP32 over serial.
    Each line: {"skin_temp_delta":0.3,"grip_force":0.7,...}
    Map → CHIMERA stim → BrainState → TGAM row → CSV.
    """
    try:
        import serial
    except ImportError:
        print("pip install pyserial")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent))
    from HumanBrainDT import HumanBrain

    brain  = HumanBrain()
    writer = TGAMWriter(out_csv)
    synth  = RawEEGSynth()

    print(f"Bridge: {port} @{baud} → {out_csv}")
    ser = serial.Serial(port, baud, timeout=1.0)

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                sensor = json.loads(line)
            except json.JSONDecodeError:
                continue

            stim  = esp32_to_stim(sensor)
            state = brain.stimulate_direct(stim)
            raw   = synth.next_sample(
                anxiety_level = state.anxiety.baseline,
                healing_level = state.anxiety.healing_index,
            )
            row = brainstate_to_tgam_row(state, raw)
            writer.write(row)
            print(f"  att={row['attention']:3d}  med={row['meditation']:3d}  "
                  f"heal={state.anxiety.healing_index:.2f}  "
                  f"hrv={state.ans.hrv_index:.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        ser.close()
        print(f"Saved → {out_csv}")


# ── Demo: generate synthetic session CSV ─────────────────────────────────────
def run_demo(out_csv: str = "tgam_demo.csv", n_rows: int = 120):
    """
    Generate a synthetic 2-minute session CSV:
      0-30s   : high anxiety (deadline pressure)
      30-90s  : stone active, gradual healing
      90-120s : recovered, calm
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from HumanBrainDT import HumanBrain

    brain  = HumanBrain()
    writer = TGAMWriter(out_csv)
    synth  = RawEEGSynth()

    SCENARIO = [
        # (start_row, end_row, sensor_dict, label)
        (0,   30,  {"skin_temp_delta": 0.6, "grip_force": 0.75,
                    "grip_force_delta": 0.5, "use_frequency": 0.8,
                    "session_active": 0.0, "rhythm_phase": 0.0},
         "High anxiety — pre-session"),
        (30,  90,  {"skin_temp_delta": 0.4, "grip_force": 0.6,
                    "grip_force_delta": 0.3, "use_frequency": 0.5,
                    "session_active": 1.0, "rhythm_phase": 0.5},
         "Stone active — healing"),
        (90,  120, {"skin_temp_delta": 0.1, "grip_force": 0.2,
                    "grip_force_delta": 0.05, "use_frequency": 0.2,
                    "session_active": 0.0, "rhythm_phase": 0.0},
         "Post-session — calm"),
    ]

    t0 = time.time()
    for i in range(n_rows):
        # Pick scenario for this row
        sensor = SCENARIO[0][2]
        for start, end, s, label in SCENARIO:
            if start <= i < end:
                sensor = s
                break

        # Add small noise to make it realistic
        noisy = {k: float(np.clip(v + np.random.normal(0, 0.05), 0, 1))
                 for k, v in sensor.items()}

        stim  = esp32_to_stim(noisy)
        state = brain.stimulate_direct(stim)

        # Override timestamp to simulate 1-second intervals
        state.timestamp = t0 + i

        raw = synth.next_sample(
            anxiety_level = state.anxiety.baseline,
            healing_level = state.anxiety.healing_index,
        )
        row = brainstate_to_tgam_row(state, raw)
        writer.write(row)

        if i % 10 == 0:
            phase = next(l for s2,e,_,l in SCENARIO if s2<=i<e) \
                    if i < 120 else "done"
            print(f"  [{i:3d}] {phase[:25]:25s}  "
                  f"att={row['attention']:3d}  med={row['meditation']:3d}  "
                  f"delta={row['delta']:6d}  heal={state.anxiety.healing_index:.2f}")

    writer.close()
    print(f"\n✓ {n_rows} rows → {out_csv}")
    return out_csv


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGAM CSV Generator for CHIMERA")
    parser.add_argument("--demo",   action="store_true",
                        help="Generate synthetic demo CSV")
    parser.add_argument("--serial", metavar="PORT",
                        help="Serial port for ESP32 bridge (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud",   type=int, default=115200)
    parser.add_argument("--out",    default="tgam_output.csv",
                        help="Output CSV file path")
    parser.add_argument("--rows",   type=int, default=120,
                        help="Number of rows for demo mode")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.out, args.rows)
    elif args.serial:
        run_serial_bridge(args.serial, args.baud, args.out)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python tgam_csv_generator.py --demo --out session.csv")
        print("  python tgam_csv_generator.py --serial COM3 --out live.csv")
