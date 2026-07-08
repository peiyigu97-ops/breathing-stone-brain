"""
core/lif_engine.py — Generic LIF simulator for human brain regions.
Extended for sensory-anxiety healing research.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from .region import BrainRegion
from .signal import (BrainState, MotorCommand, ANSState,
                     SensoryState, AnxietyState, ConductionPath)

# ── LIF constants ─────────────────────────────────────
TAU     = 8.0
V_REST  = -70.0
V_TH    = -52.0
V_RESET = -75.0
DT      = 1.0
REFRAC  = 2

# ── Sensory channels (extended) ───────────────────────
SENSORY_CHANNELS = [
    # Tactile
    "touch_light",        # light touch → Aβ → somatosensory cortex → insula
    "touch_deep",         # deep pressure → Aβ + proprioception → somatosensory + cerebellum
    "touch_vibration",    # vibration / rhythmic → somatosensory + cerebellum
    # Thermal
    "thermal_warm",       # 36-40°C warmth → insula → parasympathetic
    "thermal_neutral",    # 20-25°C neutral → somatosensory
    "thermal_cold",       # <15°C cold → brainstem → amygdala
    # Interoception
    "interoception",      # heartbeat / breath / gut → insula → amygdala
    "rhythmic",           # rhythmic input (breath sync, rocking) → cerebellum → thalamus
    # Higher-order
    "visual_input",
    "auditory_input",
    "language_input",
    "executive_input",
    "spatial_input",
    "episodic_input",
    # Arousal / valence
    "threat_input",       # acute threat → amygdala → brainstem
    "anxiety_anticipatory",  # worry / rumination → frontal → amygdala
    "reward_signal",
    "emotional_input",
    "motor_intent",
    "pain_input",
]

# ── Channel routing: which regions receive each channel ──
# Insula added as part of parietal (approximation — no standalone insula region yet)
CHANNEL_ROUTING: dict[str, list[str]] = {
    # Tactile → somatosensory cortex (parietal) + thalamus relay
    "touch_light":          ["parietal_cortex", "thalamus"],
    "touch_deep":           ["parietal_cortex", "thalamus", "cerebellum"],
    "touch_vibration":      ["parietal_cortex", "cerebellum", "thalamus"],
    # Thermal: warm → calming pathway; cold → alerting pathway
    "thermal_warm":         ["parietal_cortex", "thalamus"],
    "thermal_neutral":      ["parietal_cortex", "thalamus"],
    "thermal_cold":         ["brainstem", "amygdala", "thalamus"],
    # Interoception → insula proxy (parietal) → amygdala
    "interoception":        ["parietal_cortex", "amygdala"],
    # Rhythmic → cerebellum (timing) → thalamus → cortex
    "rhythmic":             ["cerebellum", "thalamus", "frontal_cortex"],
    # Higher-order
    "visual_input":         ["occipital_cortex", "thalamus"],
    "auditory_input":       ["temporal_cortex",  "thalamus"],
    "language_input":       ["frontal_cortex",   "temporal_cortex"],
    "executive_input":      ["frontal_cortex"],
    "spatial_input":        ["parietal_cortex",  "hippocampus"],
    "episodic_input":       ["hippocampus"],
    # Threat / anxiety
    "threat_input":         ["amygdala", "brainstem"],
    "anxiety_anticipatory": ["frontal_cortex", "amygdala", "hippocampus"],
    "reward_signal":        ["basal_ganglia"],
    "emotional_input":      ["amygdala"],
    "motor_intent":         ["basal_ganglia", "cerebellum"],
    "pain_input":           ["brainstem", "amygdala"],
}

# ── Healing weights: how much each channel suppresses amygdala ──
# Positive = inhibits amygdala (calming); derived from neuroscience literature
HEALING_WEIGHTS: dict[str, float] = {
    "touch_deep":           0.85,  # deep pressure → oxytocin, vagal tone ↑
    "thermal_warm":         0.75,  # warm touch → insula → parasympathetic
    "touch_vibration":      0.60,  # rhythmic vibration → cerebellum → thalamus gating
    "rhythmic":             0.70,  # breath sync / rocking → vagal tone ↑
    "touch_light":          0.35,  # light touch → moderate calming
    "interoception":        0.45,  # mindful interoception → PFC regulation
    "thermal_neutral":      0.15,
    "reward_signal":        0.50,
    "thermal_cold":        -0.30,  # cold → sympathetic ↑ (can increase anxiety)
    "pain_input":          -0.90,
    "threat_input":        -1.00,
    "anxiety_anticipatory":-0.80,
}

# ── Conduction latencies (ms, simulated) ──────────────
LATENCY: dict[tuple, float] = {
    ("touch_deep",    "parietal_cortex"):  12,
    ("touch_deep",    "thalamus"):          8,
    ("touch_deep",    "cerebellum"):       18,
    ("thermal_warm",  "parietal_cortex"):  20,
    ("thermal_warm",  "thalamus"):         15,
    ("rhythmic",      "cerebellum"):       10,
    ("rhythmic",      "thalamus"):         18,
    ("rhythmic",      "frontal_cortex"):   35,
    ("interoception", "parietal_cortex"):  25,
    ("interoception", "amygdala"):         30,
    ("threat_input",  "amygdala"):          5,
    ("threat_input",  "brainstem"):         3,
}

_REGION_LABELS: dict[str, str] = {
    "frontal_cortex":   "Prefrontal Cortex",
    "parietal_cortex":  "Parietal Cortex / Insula",
    "temporal_cortex":  "Temporal Cortex",
    "occipital_cortex": "Visual Cortex",
    "basal_ganglia":    "Basal Ganglia",
    "thalamus":         "Thalamus",
    "hippocampus":      "Hippocampus",
    "amygdala":         "Amygdala",
    "cerebellum":       "Cerebellum",
    "brainstem":        "Brain Stem",
}


class RegionSimulator:
    """LIF engine for a single BrainRegion."""

    def __init__(self, region: BrainRegion):
        self.region = region
        pop = region.population
        self.N           = pop.n_neurons
        self.W           = pop.W
        self.channel_idx = pop.channel_idx
        self.fwd_idx     = pop.fwd_idx
        self.back_idx    = pop.back_idx
        self._reset()

    def _reset(self):
        self.V       = np.full(self.N, V_REST, dtype=np.float32)
        self.spikes  = np.zeros(self.N, bool)
        self.refrac  = np.zeros(self.N, int)
        self.I_ext   = np.zeros(self.N, dtype=np.float32)
        self.history = np.zeros(self.N, dtype=np.float32)

    def stimulate(self, stim: dict):
        self.I_ext[:] = 0.0
        for ch, val in stim.items():
            for i in self.channel_idx.get(ch, [])[:30]:
                self.I_ext[i] = val * 80.0

    def step(self):
        prev  = self.spikes.copy()
        I_syn = self.W.T @ prev.astype(np.float32) * 400.0
        self.V += (-(self.V - V_REST) + I_syn + self.I_ext) / TAU * DT
        in_ref = self.refrac > 0
        self.V[in_ref]    = V_RESET
        self.refrac[in_ref] -= 1
        fired = (self.V >= V_TH) & ~in_ref
        self.V[fired]      = V_RESET
        self.refrac[fired] = REFRAC
        self.spikes  = fired
        self.history += fired.astype(np.float32)
        self.I_ext   *= 0.995

    def run(self, stim: dict, steps: int = 200) -> dict:
        self._reset()
        self.stimulate(stim)
        for _ in range(steps):
            self.step()

        def rate(idx_list):
            if not idx_list:
                return 0.0
            total = float(sum(self.history[i] for i in idx_list if i < self.N))
            return float(np.tanh(total / max(steps * 2.0, 1)))

        fwd  = rate(self.fwd_idx)
        back = rate(self.back_idx)
        lv   = rate(self.fwd_idx[:len(self.fwd_idx) // 2])
        rv   = rate(self.fwd_idx[len(self.fwd_idx) // 2:])

        pain_ch    = self.channel_idx.get("pain_input",
                     self.channel_idx.get("threat_input", []))
        pain_fired = float(sum(self.history[i] for i in pain_ch if i < self.N))
        pain_rate  = pain_fired / max(len(pain_ch) * steps * 0.05, 1)
        withdraw   = float(np.tanh(pain_rate * 2.0))

        total  = int(self.history.sum())
        active = float(np.tanh(total / max(steps * self.N * 0.005, 1)))

        if self.region.vectors is not None:
            self.region.vectors.activity = self.history.copy()

        return {
            "forward":       fwd,
            "backward":      back,
            "turn_left":     max(0.0, rv - lv),
            "turn_right":    max(0.0, lv - rv),
            "active":        active,
            "withdraw":      withdraw,
            "_total_spikes": total,
        }


class BrainSimulator:
    """
    Simulates all human brain regions with anxiety-sensory healing model.
    Tracks: ANS state, anxiety decomposition, sensory state, conduction paths.
    """

    def __init__(self, regions: list[BrainRegion],
                 inter_region_W: Optional[np.ndarray] = None):
        self.regions      = {r.id: r for r in regions}
        self.simulators   = {r.id: RegionSimulator(r) for r in regions}
        self.region_order = [r.id for r in regions]
        n = len(regions)
        self.inter_W = inter_region_W if inter_region_W is not None \
                       else np.eye(n, dtype=np.float32) * 0.1
        # Persistent anxiety baseline (decays slowly between stimulations)
        self._anxiety_baseline: float = 0.0

    def run(self, stim: dict, steps: int = 200) -> BrainState:
        results: dict[str, dict] = {}
        for rid, sim in self.simulators.items():
            local_stim = {
                ch: val for ch, val in stim.items()
                if rid in CHANNEL_ROUTING.get(ch, [])
            }
            results[rid] = sim.run(local_stim, steps)

        region_spikes = {rid: r["_total_spikes"] for rid, r in results.items()}
        total_spikes  = sum(region_spikes.values())
        dominant      = max(region_spikes, key=region_spikes.get) \
                        if region_spikes else "brainstem"

        # Update persistent anxiety baseline
        threat_val  = stim.get("threat_input", 0) + stim.get("anxiety_anticipatory", 0)
        healing_val = sum(max(0, HEALING_WEIGHTS.get(ch, 0)) * v
                         for ch, v in stim.items())
        self._anxiety_baseline = float(np.clip(
            self._anxiety_baseline * 0.85 + threat_val * 0.3 - healing_val * 0.2,
            0.0, 1.0))

        cmd      = self._aggregate_motor(results)
        ans      = self._compute_ans(stim, results)
        sensory  = self._compute_sensory(stim)
        anxiety  = self._compute_anxiety(stim, results, ans)
        path     = self._conduction_path(stim)
        exp      = self._experience(stim, results, dominant, total_spikes, anxiety)

        return BrainState(
            timestamp       = time.time(),
            region_activity = region_spikes,
            total_spikes    = total_spikes,
            dominant_region = dominant,
            motor_command   = cmd,
            ans             = ans,
            sensory         = sensory,
            anxiety         = anxiety,
            experience      = exp,
            conduction_path = path,
        )

    def _compute_ans(self, stim: dict, results: dict) -> ANSState:
        # Sympathetic drivers
        sym = (stim.get("threat_input", 0) * 0.9 +
               stim.get("anxiety_anticipatory", 0) * 0.7 +
               stim.get("pain_input", 0) * 0.8 +
               stim.get("thermal_cold", 0) * 0.5)
        # Parasympathetic drivers (healing channels)
        para = (stim.get("touch_deep", 0) * 0.85 +
                stim.get("thermal_warm", 0) * 0.75 +
                stim.get("rhythmic", 0) * 0.70 +
                stim.get("touch_vibration", 0) * 0.55 +
                stim.get("interoception", 0) * 0.45 +
                stim.get("touch_light", 0) * 0.30)
        # Amygdala activity adds to sympathetic
        amy_spikes = results.get("amygdala", {}).get("_total_spikes", 0)
        sym += float(np.tanh(amy_spikes / 2000)) * 0.4
        # PFC activity (frontal) adds to parasympathetic via top-down regulation
        pfc_spikes = results.get("frontal_cortex", {}).get("_total_spikes", 0)
        para += float(np.tanh(pfc_spikes / 3000)) * 0.3

        sym  = float(np.clip(sym,  0, 1))
        para = float(np.clip(para, 0, 1))
        hrv  = float(np.clip(para - sym * 0.5 + 0.3, 0, 1))
        cort = float(np.clip(self._anxiety_baseline * 0.7 + sym * 0.3, 0, 1))

        return ANSState(
            sympathetic     = round(sym,  3),
            parasympathetic = round(para, 3),
            hrv_index       = round(hrv,  3),
            cortisol        = round(cort, 3),
        )

    def _compute_sensory(self, stim: dict) -> SensoryState:
        return SensoryState(
            touch_light     = round(stim.get("touch_light",     0), 3),
            touch_deep      = round(stim.get("touch_deep",      0), 3),
            touch_vibration = round(stim.get("touch_vibration", 0), 3),
            thermal_warm    = round(stim.get("thermal_warm",    0), 3),
            thermal_neutral = round(stim.get("thermal_neutral", 0), 3),
            thermal_cold    = round(stim.get("thermal_cold",    0), 3),
            interoception   = round(stim.get("interoception",   0), 3),
            rhythmic        = round(stim.get("rhythmic",        0), 3),
        )

    def _compute_anxiety(self, stim: dict, results: dict,
                         ans: ANSState) -> AnxietyState:
        amy   = results.get("amygdala",      {}).get("_total_spikes", 0)
        pfc   = results.get("frontal_cortex",{}).get("_total_spikes", 0)
        hipp  = results.get("hippocampus",   {}).get("_total_spikes", 0)
        insula= results.get("parietal_cortex",{}).get("_total_spikes", 0)

        baseline     = float(np.clip(self._anxiety_baseline, 0, 1))
        anticipatory = float(np.tanh(
            (stim.get("anxiety_anticipatory", 0) * 0.8 +
             hipp / max(hipp + pfc + 1, 1) * 0.4)))
        somatic      = float(np.tanh(
            insula / max(insula + 1, 1) * 0.003 +
            stim.get("interoception", 0) * 0.5 * ans.sympathetic))
        regulation   = float(np.tanh(pfc / max(amy + 1, 1) * 0.8))

        # Healing index: net calming effect of current sensory input
        healing_raw = sum(HEALING_WEIGHTS.get(ch, 0) * v
                         for ch, v in stim.items())
        healing = float(np.clip(healing_raw / max(len(stim), 1), 0, 1))

        return AnxietyState(
            baseline      = round(baseline,     3),
            anticipatory  = round(anticipatory, 3),
            somatic       = round(somatic,      3),
            regulation    = round(regulation,   3),
            healing_index = round(healing,      3),
        )

    def _conduction_path(self, stim: dict) -> list[ConductionPath]:
        """Build the primary sensory conduction pathway for this stimulus."""
        paths = []
        for ch, val in stim.items():
            if val < 0.1:
                continue
            targets = CHANNEL_ROUTING.get(ch, [])
            for t in targets[:3]:  # max 3 hops shown
                lat = LATENCY.get((ch, t), 25.0)
                paths.append(ConductionPath(
                    from_region = ch,
                    to_region   = t,
                    channel     = ch,
                    weight      = round(val, 3),
                    latency_ms  = lat,
                ))
        return paths

    def _aggregate_motor(self, results: dict) -> MotorCommand:
        def g(region, key):
            return results.get(region, {}).get(key, 0.0)

        approach = max(g("frontal_cortex", "forward"),
                       g("basal_ganglia",  "forward"))
        avoid    = max(g("amygdala",  "backward"),
                       g("brainstem", "backward"))
        withdraw = max(g("amygdala",  "withdraw"),
                       g("brainstem", "withdraw"))
        engage   = g("frontal_cortex", "active")
        orient   = max(g("parietal_cortex",  "turn_left"),
                       g("parietal_cortex",  "turn_right"),
                       g("occipital_cortex", "active"))
        freeze   = max(0.0, 1.0 - max(approach, avoid, engage)) \
                   if avoid > 0.7 else 0.0

        return MotorCommand(
            approach = round(approach, 3),
            avoid    = round(avoid,    3),
            freeze   = round(freeze,   3),
            engage   = round(engage,   3),
            orient   = round(orient,   3),
            withdraw = round(withdraw, 3),
        )

    def _experience(self, stim: dict, results: dict,
                    dominant: str, total: int,
                    anxiety: AnxietyState) -> str:
        if total == 0:
            return "Quiet. No significant neural activity."

        dom = _REGION_LABELS.get(dominant, dominant.replace("_", " ").title())

        # Describe healing vs arousal state
        h = anxiety.healing_index
        b = anxiety.baseline
        if h > 0.5:
            tone = "Calming. "
        elif h > 0.2:
            tone = "Settling. "
        elif b > 0.6:
            tone = "Anxious. "
        elif b > 0.3:
            tone = "Tense. "
        else:
            tone = ""

        # Describe active sensory channels
        sensory_exp = {
            "touch_deep":      "Deep pressure activating vagal pathway.",
            "thermal_warm":    "Warmth flowing through thalamus.",
            "rhythmic":        "Rhythm entraining the cerebellum.",
            "touch_vibration": "Vibration calming somatosensory cortex.",
            "interoception":   "Body awareness rising.",
            "touch_light":     "Light touch reaching parietal cortex.",
            "thermal_cold":    "Cold signal alerting brainstem.",
            "threat_input":    "Threat circuit firing.",
            "anxiety_anticipatory": "Worry loop active between frontal and amygdala.",
            "pain_input":      "Pain signal ascending.",
        }
        active = [sensory_exp[ch] for ch in stim
                  if stim[ch] > 0.2 and ch in sensory_exp]

        parts = [f"{tone}{dom} most active."]
        parts += active[:2]
        if anxiety.regulation > 0.5:
            parts.append("Prefrontal regulation engaged.")
        return " ".join(parts)
from .signal import BrainState, MotorCommand

# ── LIF constants ─────────────────────────────────────
TAU     = 8.0
V_REST  = -70.0
V_TH    = -52.0
V_RESET = -75.0
DT      = 1.0
REFRAC  = 2

# ── Sensory channels ──────────────────────────────────
SENSORY_CHANNELS = [
    "visual_input",
    "auditory_input",
    "somatosensory",
    "language_input",
    "executive_input",
    "spatial_input",
    "episodic_input",
    "threat_input",
    "reward_signal",
    "emotional_input",
    "motor_intent",
    "pain_input",
]

# Which regions receive each channel
_REGION_LABELS: dict[str, str] = {
    "frontal_cortex":   "Prefrontal cortex",
    "parietal_cortex":  "Parietal cortex",
    "temporal_cortex":  "Temporal cortex",
    "occipital_cortex": "Visual cortex",
    "basal_ganglia":    "Basal ganglia",
    "thalamus":         "Thalamus",
    "hippocampus":      "Hippocampus",
    "amygdala":         "Amygdala",
    "cerebellum":       "Cerebellum",
    "brainstem":        "Brain stem",
}

_CHANNEL_EXP: dict[str, str] = {
    "threat_input":    "threat detected",
    "pain_input":      "pain signal",
    "reward_signal":   "reward anticipated",
    "episodic_input":  "memory retrieval",
    "language_input":  "language processing",
    "visual_input":    "visual processing",
    "motor_intent":    "motor preparation",
    "emotional_input": "emotional response",
    "executive_input": "executive control",
    "auditory_input":  "auditory processing",
    "spatial_input":   "spatial awareness",
    "somatosensory":   "touch sensation",
}


class RegionSimulator:
    """LIF engine for a single BrainRegion."""

    def __init__(self, region: BrainRegion):
        self.region = region
        pop = region.population
        self.N           = pop.n_neurons
        self.W           = pop.W
        self.channel_idx = pop.channel_idx
        self.fwd_idx     = pop.fwd_idx
        self.back_idx    = pop.back_idx
        self._reset()

    def _reset(self):
        self.V       = np.full(self.N, V_REST, dtype=np.float32)
        self.spikes  = np.zeros(self.N, bool)
        self.refrac  = np.zeros(self.N, int)
        self.I_ext   = np.zeros(self.N, dtype=np.float32)
        self.history = np.zeros(self.N, dtype=np.float32)

    def stimulate(self, stim: dict):
        self.I_ext[:] = 0.0
        for ch, val in stim.items():
            for i in self.channel_idx.get(ch, [])[:30]:
                self.I_ext[i] = val * 80.0

    def step(self):
        prev  = self.spikes.copy()
        I_syn = self.W.T @ prev.astype(np.float32) * 400.0
        self.V += (-(self.V - V_REST) + I_syn + self.I_ext) / TAU * DT
        in_ref = self.refrac > 0
        self.V[in_ref]    = V_RESET
        self.refrac[in_ref] -= 1
        fired = (self.V >= V_TH) & ~in_ref
        self.V[fired]      = V_RESET
        self.refrac[fired] = REFRAC
        self.spikes  = fired
        self.history += fired.astype(np.float32)
        self.I_ext   *= 0.995

    def run(self, stim: dict, steps: int = 200) -> dict:
        self._reset()
        self.stimulate(stim)
        for _ in range(steps):
            self.step()

        def rate(idx_list):
            if not idx_list:
                return 0.0
            total = float(sum(self.history[i] for i in idx_list if i < self.N))
            return float(np.tanh(total / max(steps * 2.0, 1)))

        fwd  = rate(self.fwd_idx)
        back = rate(self.back_idx)
        lv   = rate(self.fwd_idx[:len(self.fwd_idx) // 2])
        rv   = rate(self.fwd_idx[len(self.fwd_idx) // 2:])

        pain_ch    = self.channel_idx.get("pain_input",
                     self.channel_idx.get("threat_input", []))
        pain_fired = float(sum(self.history[i] for i in pain_ch if i < self.N))
        pain_rate  = pain_fired / max(len(pain_ch) * steps * 0.05, 1)
        withdraw   = float(np.tanh(pain_rate * 2.0))

        total  = int(self.history.sum())
        active = float(np.tanh(total / max(steps * self.N * 0.005, 1)))

        if self.region.vectors is not None:
            self.region.vectors.activity = self.history.copy()

        return {
            "forward":       fwd,
            "backward":      back,
            "turn_left":     max(0.0, rv - lv),
            "turn_right":    max(0.0, lv - rv),
            "active":        active,
            "withdraw":      withdraw,
            "_total_spikes": total,
        }


class BrainSimulator:
    """
    Simulates all human brain regions with anxiety-sensory healing model.
    Tracks: ANS state, anxiety decomposition, sensory state, conduction paths.
    """

    def __init__(self, regions: list[BrainRegion],
                 inter_region_W: Optional[np.ndarray] = None):
        self.regions      = {r.id: r for r in regions}
        self.simulators   = {r.id: RegionSimulator(r) for r in regions}
        self.region_order = [r.id for r in regions]
        n = len(regions)
        self.inter_W = inter_region_W if inter_region_W is not None \
                       else np.eye(n, dtype=np.float32) * 0.1
        self._anxiety_baseline: float = 0.0

    def run(self, stim: dict, steps: int = 200) -> BrainState:
        results: dict[str, dict] = {}
        for rid, sim in self.simulators.items():
            local_stim = {
                ch: val for ch, val in stim.items()
                if rid in CHANNEL_ROUTING.get(ch, [])
            }
            results[rid] = sim.run(local_stim, steps)

        region_spikes = {rid: r["_total_spikes"] for rid, r in results.items()}
        total_spikes  = sum(region_spikes.values())
        dominant      = max(region_spikes, key=region_spikes.get) \
                        if region_spikes else "brainstem"

        threat_val  = stim.get("threat_input", 0) + stim.get("anxiety_anticipatory", 0)
        healing_val = sum(max(0, HEALING_WEIGHTS.get(ch, 0)) * v
                         for ch, v in stim.items())
        self._anxiety_baseline = float(np.clip(
            self._anxiety_baseline * 0.85 + threat_val * 0.3 - healing_val * 0.2,
            0.0, 1.0))

        cmd     = self._aggregate_motor(results)
        ans     = self._compute_ans(stim, results)
        sensory = self._compute_sensory(stim)
        anxiety = self._compute_anxiety(stim, results, ans)
        path    = self._conduction_path(stim)
        exp     = self._experience(stim, results, dominant, total_spikes, anxiety)

        return BrainState(
            timestamp       = time.time(),
            region_activity = region_spikes,
            total_spikes    = total_spikes,
            dominant_region = dominant,
            motor_command   = cmd,
            ans             = ans,
            sensory         = sensory,
            anxiety         = anxiety,
            experience      = exp,
            conduction_path = path,
        )

    def _compute_ans(self, stim: dict, results: dict) -> ANSState:
        sym = (stim.get("threat_input", 0) * 0.9 +
               stim.get("anxiety_anticipatory", 0) * 0.7 +
               stim.get("pain_input", 0) * 0.8 +
               stim.get("thermal_cold", 0) * 0.5)
        para = (stim.get("touch_deep", 0) * 0.85 +
                stim.get("thermal_warm", 0) * 0.75 +
                stim.get("rhythmic", 0) * 0.70 +
                stim.get("touch_vibration", 0) * 0.55 +
                stim.get("interoception", 0) * 0.45 +
                stim.get("touch_light", 0) * 0.30)
        amy_spikes = results.get("amygdala", {}).get("_total_spikes", 0)
        sym += float(np.tanh(amy_spikes / 2000)) * 0.4
        pfc_spikes = results.get("frontal_cortex", {}).get("_total_spikes", 0)
        para += float(np.tanh(pfc_spikes / 3000)) * 0.3
        sym  = float(np.clip(sym,  0, 1))
        para = float(np.clip(para, 0, 1))
        hrv  = float(np.clip(para - sym * 0.5 + 0.3, 0, 1))
        cort = float(np.clip(self._anxiety_baseline * 0.7 + sym * 0.3, 0, 1))
        return ANSState(
            sympathetic     = round(sym,  3),
            parasympathetic = round(para, 3),
            hrv_index       = round(hrv,  3),
            cortisol        = round(cort, 3),
        )

    def _compute_sensory(self, stim: dict) -> SensoryState:
        return SensoryState(
            touch_light     = round(stim.get("touch_light",     0), 3),
            touch_deep      = round(stim.get("touch_deep",      0), 3),
            touch_vibration = round(stim.get("touch_vibration", 0), 3),
            thermal_warm    = round(stim.get("thermal_warm",    0), 3),
            thermal_neutral = round(stim.get("thermal_neutral", 0), 3),
            thermal_cold    = round(stim.get("thermal_cold",    0), 3),
            interoception   = round(stim.get("interoception",   0), 3),
            rhythmic        = round(stim.get("rhythmic",        0), 3),
        )

    def _compute_anxiety(self, stim: dict, results: dict,
                         ans: ANSState) -> AnxietyState:
        amy   = results.get("amygdala",       {}).get("_total_spikes", 0)
        pfc   = results.get("frontal_cortex", {}).get("_total_spikes", 0)
        hipp  = results.get("hippocampus",    {}).get("_total_spikes", 0)
        insula= results.get("parietal_cortex",{}).get("_total_spikes", 0)

        baseline     = float(np.clip(self._anxiety_baseline, 0, 1))
        anticipatory = float(np.tanh(
            stim.get("anxiety_anticipatory", 0) * 0.8 +
            hipp / max(hipp + pfc + 1, 1) * 0.4))
        somatic      = float(np.tanh(
            insula / max(insula + 1, 1) * 0.003 +
            stim.get("interoception", 0) * 0.5 * ans.sympathetic))
        regulation   = float(np.tanh(pfc / max(amy + 1, 1) * 0.8))
        healing_raw  = sum(HEALING_WEIGHTS.get(ch, 0) * v
                          for ch, v in stim.items())
        healing      = float(np.clip(healing_raw / max(len(stim), 1), 0, 1))

        return AnxietyState(
            baseline      = round(baseline,     3),
            anticipatory  = round(anticipatory, 3),
            somatic       = round(somatic,      3),
            regulation    = round(regulation,   3),
            healing_index = round(healing,      3),
        )

    def _conduction_path(self, stim: dict) -> list:
        paths = []
        for ch, val in stim.items():
            if val < 0.1:
                continue
            for t in CHANNEL_ROUTING.get(ch, [])[:3]:
                lat = LATENCY.get((ch, t), 25.0)
                paths.append(ConductionPath(
                    from_region=ch, to_region=t,
                    channel=ch, weight=round(val, 3), latency_ms=lat,
                ))
        return paths

    def _aggregate_motor(self, results: dict) -> MotorCommand:
        def g(region, key):
            return results.get(region, {}).get(key, 0.0)
        approach = max(g("frontal_cortex", "forward"), g("basal_ganglia", "forward"))
        avoid    = max(g("amygdala", "backward"), g("brainstem", "backward"))
        withdraw = max(g("amygdala", "withdraw"), g("brainstem", "withdraw"))
        engage   = g("frontal_cortex", "active")
        orient   = max(g("parietal_cortex", "turn_left"),
                       g("parietal_cortex", "turn_right"),
                       g("occipital_cortex", "active"))
        freeze   = max(0.0, 1.0 - max(approach, avoid, engage)) if avoid > 0.7 else 0.0
        return MotorCommand(
            approach=round(approach,3), avoid=round(avoid,3),
            freeze=round(freeze,3),    engage=round(engage,3),
            orient=round(orient,3),    withdraw=round(withdraw,3),
        )

    def _experience(self, stim: dict, results: dict,
                    dominant: str, total: int,
                    anxiety: AnxietyState) -> str:
        if total == 0:
            return "Quiet. No significant neural activity."
        dom = _REGION_LABELS.get(dominant, dominant.replace("_", " ").title())
        h, b = anxiety.healing_index, anxiety.baseline
        if h > 0.5:    tone = "Calming. "
        elif h > 0.2:  tone = "Settling. "
        elif b > 0.6:  tone = "Anxious. "
        elif b > 0.3:  tone = "Tense. "
        else:          tone = ""
        sensory_exp = {
            "touch_deep":           "Deep pressure activating vagal pathway.",
            "thermal_warm":         "Warmth flowing through thalamus.",
            "rhythmic":             "Rhythm entraining the cerebellum.",
            "touch_vibration":      "Vibration calming somatosensory cortex.",
            "interoception":        "Body awareness rising.",
            "touch_light":          "Light touch reaching parietal cortex.",
            "thermal_cold":         "Cold signal alerting brainstem.",
            "threat_input":         "Threat circuit firing.",
            "anxiety_anticipatory": "Worry loop active between frontal and amygdala.",
            "pain_input":           "Pain signal ascending.",
        }
        active = [sensory_exp[ch] for ch in stim
                  if stim[ch] > 0.2 and ch in sensory_exp]
        parts = [f"{tone}{dom} most active."] + active[:2]
        if anxiety.regulation > 0.5:
            parts.append("Prefrontal regulation engaged.")
        return " ".join(parts)
