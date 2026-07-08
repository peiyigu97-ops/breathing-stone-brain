"""
core/signal.py — BrainState and related data structures.
Extended for sensory-anxiety healing research.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MotorCommand:
    approach:  float = 0.0
    avoid:     float = 0.0
    freeze:    float = 0.0
    engage:    float = 0.0
    orient:    float = 0.0
    withdraw:  float = 0.0


@dataclass
class ANSState:
    """Autonomic Nervous System output."""
    sympathetic:  float = 0.0   # fight-or-flight arousal [0,1]
    parasympathetic: float = 0.0  # rest-and-digest [0,1]
    hrv_index:    float = 0.0   # heart rate variability proxy (higher = calmer)
    cortisol:     float = 0.0   # stress hormone proxy [0,1]


@dataclass
class SensoryState:
    """Current sensory processing state."""
    touch_light:     float = 0.0   # light touch (Aβ fibers)
    touch_deep:      float = 0.0   # deep pressure (Aβ + proprioception)
    touch_vibration: float = 0.0   # vibration / rhythm
    thermal_warm:    float = 0.0   # 36-40°C warmth
    thermal_neutral: float = 0.0   # 20-25°C neutral
    thermal_cold:    float = 0.0   # <15°C cold
    interoception:   float = 0.0   # heartbeat / breath / gut awareness
    rhythmic:        float = 0.0   # rhythmic input (breath sync, rocking)


@dataclass
class AnxietyState:
    """Decomposed anxiety profile."""
    baseline:       float = 0.0   # tonic amygdala activation
    anticipatory:   float = 0.0   # prefrontal-amygdala worry loop
    somatic:        float = 0.0   # body-based anxiety (insula)
    regulation:     float = 0.0   # PFC → amygdala top-down control [0=none,1=full]
    healing_index:  float = 0.0   # composite: how much sensory input is reducing anxiety


@dataclass
class ConductionPath:
    """One step in a sensory transmission pathway."""
    from_region: str
    to_region:   str
    channel:     str
    weight:      float
    latency_ms:  float   # simulated conduction delay


@dataclass
class BrainState:
    timestamp:       float
    region_activity: dict          # region_id → spike count
    total_spikes:    int
    dominant_region: str
    motor_command:   MotorCommand
    ans:             ANSState
    sensory:         SensoryState
    anxiety:         AnxietyState
    experience:      str
    conduction_path: list[ConductionPath] = field(default_factory=list)
    type_fires:      dict = field(default_factory=dict)


@dataclass
class MemoryTrace:
    region_id: str
    vector:    np.ndarray
    strength:  float = 0.0
    timestamp: float = 0.0
