"""
HumanBrainDT/brain.py — Top-level HumanBrain class.
Sensory-anxiety healing research model.
"""
from __future__ import annotations
from dataclasses import asdict

from .core import BrainSimulator, BrainState, SENSORY_CHANNELS
from .regions import build_all_regions


class HumanBrain:
    """
    Digital twin of the human brain.
    10 regions, LIF simulation with anxiety-sensory healing model.
    """

    def __init__(self):
        regions, inter_W = build_all_regions()
        self.simulator = BrainSimulator(regions, inter_W)
        self.regions   = self.simulator.regions
        n_total = sum(r.population.n_neurons for r in regions)
        n_syn   = sum((r.population.W != 0).sum() for r in regions)
        print(f"🧠 HumanBrain: {len(regions)} regions  "
              f"{n_total} neurons  {int(n_syn):,} synapses")

    def stimulate(self, text: str, steps: int = 200) -> BrainState:
        return self.simulator.run(self._parse(text), steps)

    def stimulate_direct(self, stim: dict, steps: int = 200) -> BrainState:
        return self.simulator.run(stim, steps)

    def state_to_dict(self, state: BrainState) -> dict:
        return {
            "timestamp":       state.timestamp,
            "total_spikes":    state.total_spikes,
            "dominant_region": state.dominant_region,
            "region_activity": state.region_activity,
            "motor_command":   asdict(state.motor_command),
            "ans": {
                "sympathetic":     state.ans.sympathetic,
                "parasympathetic": state.ans.parasympathetic,
                "hrv_index":       state.ans.hrv_index,
                "cortisol":        state.ans.cortisol,
            },
            "sensory": {
                "touch_light":     state.sensory.touch_light,
                "touch_deep":      state.sensory.touch_deep,
                "touch_vibration": state.sensory.touch_vibration,
                "thermal_warm":    state.sensory.thermal_warm,
                "thermal_neutral": state.sensory.thermal_neutral,
                "thermal_cold":    state.sensory.thermal_cold,
                "interoception":   state.sensory.interoception,
                "rhythmic":        state.sensory.rhythmic,
            },
            "anxiety": {
                "baseline":      state.anxiety.baseline,
                "anticipatory":  state.anxiety.anticipatory,
                "somatic":       state.anxiety.somatic,
                "regulation":    state.anxiety.regulation,
                "healing_index": state.anxiety.healing_index,
            },
            "conduction_path": [
                {"from": p.from_region, "to": p.to_region,
                 "channel": p.channel, "weight": p.weight,
                 "latency_ms": p.latency_ms}
                for p in state.conduction_path
            ],
            "experience": state.experience,
        }

    # ── Keyword parser ─────────────────────────────────────
    _KMAP = {
        # Deep pressure
        "deep pressure":  {"touch_deep": 0.95},
        "weighted":       {"touch_deep": 0.9},
        "hug":            {"touch_deep": 0.85, "thermal_warm": 0.4},
        "squeeze":        {"touch_deep": 0.8},
        "hold":           {"touch_deep": 0.7},
        "massage":        {"touch_deep": 0.85, "touch_vibration": 0.3},
        "压":             {"touch_deep": 0.9},
        "压迫":           {"touch_deep": 0.9},
        "拥抱":           {"touch_deep": 0.8, "thermal_warm": 0.4},
        # Light touch
        "touch":          {"touch_light": 0.8},
        "stroke":         {"touch_light": 0.75},
        "caress":         {"touch_light": 0.7},
        "轻触":           {"touch_light": 0.8},
        "触摸":           {"touch_light": 0.75},
        # Vibration / rhythm
        "vibrate":        {"touch_vibration": 0.85},
        "rock":           {"touch_vibration": 0.7, "rhythmic": 0.7},
        "sway":           {"touch_vibration": 0.5, "rhythmic": 0.65},
        "rhythm":         {"rhythmic": 0.85},
        "breathe":        {"rhythmic": 0.9, "interoception": 0.6},
        "breathing":      {"rhythmic": 0.9, "interoception": 0.6},
        "inhale":         {"rhythmic": 0.7, "interoception": 0.5},
        "exhale":         {"rhythmic": 0.8, "interoception": 0.55},
        "呼吸":           {"rhythmic": 0.9, "interoception": 0.6},
        "节奏":           {"rhythmic": 0.8},
        # Thermal warm
        "warm":           {"thermal_warm": 0.85},
        "warmth":         {"thermal_warm": 0.9},
        "hot":            {"thermal_warm": 0.6},
        "heat":           {"thermal_warm": 0.7},
        "bath":           {"thermal_warm": 0.85, "touch_deep": 0.3},
        "hot spring":     {"thermal_warm": 0.95, "touch_deep": 0.4},
        "温暖":           {"thermal_warm": 0.9},
        "暖":             {"thermal_warm": 0.8},
        # Thermal neutral
        "cool":           {"thermal_neutral": 0.7},
        "breeze":         {"thermal_neutral": 0.6, "rhythmic": 0.3},
        # Thermal cold
        "cold":           {"thermal_cold": 0.8},
        "cold water":     {"thermal_cold": 0.85},
        "冷":             {"thermal_cold": 0.8},
        # Interoception
        "heartbeat":      {"interoception": 0.8, "rhythmic": 0.5},
        "body scan":      {"interoception": 0.9, "executive_input": 0.4},
        "grounded":       {"interoception": 0.7, "touch_deep": 0.5},
        "mindful":        {"interoception": 0.8, "executive_input": 0.5},
        "内感受":         {"interoception": 0.9},
        "正念":           {"interoception": 0.8, "executive_input": 0.5},
        # Anxiety / threat
        "anxiety":        {"anxiety_anticipatory": 0.8, "emotional_input": 0.6},
        "anxious":        {"anxiety_anticipatory": 0.75, "emotional_input": 0.55},
        "worry":          {"anxiety_anticipatory": 0.85},
        "stress":         {"anxiety_anticipatory": 0.7, "threat_input": 0.4},
        "panic":          {"threat_input": 0.9, "anxiety_anticipatory": 0.7},
        "fear":           {"threat_input": 0.85, "emotional_input": 0.65},
        "danger":         {"threat_input": 1.0},
        "紧张":           {"anxiety_anticipatory": 0.8, "emotional_input": 0.5},
        "焦虑":           {"anxiety_anticipatory": 0.85, "emotional_input": 0.6},
        "恐惧":           {"threat_input": 0.85, "emotional_input": 0.65},
        # Positive valence
        "calm":           {"rhythmic": 0.5, "touch_deep": 0.3, "reward_signal": 0.4},
        "relax":          {"thermal_warm": 0.5, "rhythmic": 0.6, "reward_signal": 0.4},
        "safe":           {"reward_signal": 0.6, "executive_input": 0.4},
        "peace":          {"rhythmic": 0.6, "reward_signal": 0.5},
        "happy":          {"reward_signal": 0.8, "emotional_input": 0.5},
        "放松":           {"thermal_warm": 0.5, "rhythmic": 0.6},
        "平静":           {"rhythmic": 0.6, "reward_signal": 0.5},
        # Pain
        "pain":           {"pain_input": 1.0},
        "hurt":           {"pain_input": 0.85},
        "ache":           {"pain_input": 0.7},
        "疼":             {"pain_input": 0.9},
        "痛":             {"pain_input": 0.9},
        # Cognitive
        "remember":       {"episodic_input": 0.9},
        "think":          {"executive_input": 0.7},
        "focus":          {"executive_input": 0.8},
        "see":            {"visual_input": 0.9},
        "hear":           {"auditory_input": 0.9},
        # Stop
        "stop": {}, "rest": {}, "quiet": {}, "still": {},
    }

    def _parse(self, text: str) -> dict:
        t = text.lower().strip()
        stim: dict = {}
        for kw, channels in self._KMAP.items():
            if kw in t:
                for ch, val in channels.items():
                    stim[ch] = max(stim.get(ch, 0.0), val)
        if not stim and t not in ("stop", "rest", "quiet", "still", ""):
            stim = {"executive_input": 0.3, "language_input": 0.2}
        return stim
