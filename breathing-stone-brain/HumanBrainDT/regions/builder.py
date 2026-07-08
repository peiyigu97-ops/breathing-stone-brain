"""
regions/builder.py — Build all 10 human brain regions with
randomized-but-structured weight matrices and channel mappings.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

from ..core.region import BrainRegion, NeuronPopulation, RegionVectors

# ── Region definitions ────────────────────────────────
# (id, label, parent, n_neurons, input_channels, fwd_fraction, back_fraction)
REGION_SPECS = [
    ("frontal_cortex",   "Frontal Cortex",    "cortex",        120,
     ["executive_input", "language_input",
      "rhythmic", "interoception"],            0.35, 0.10),
    ("parietal_cortex",  "Parietal Cortex / Insula", "cortex", 80,
     ["somatosensory",   "spatial_input",
      "touch_light",     "touch_deep",
      "touch_vibration", "thermal_warm",
      "thermal_neutral", "thermal_cold",
      "interoception"],                        0.30, 0.10),
    ("temporal_cortex",  "Temporal Cortex",   "cortex",         80,
     ["auditory_input",  "language_input"],   0.30, 0.10),
    ("occipital_cortex", "Occipital Cortex",  "cortex",         60,
     ["visual_input"],                        0.30, 0.10),
    ("basal_ganglia",    "Basal Ganglia",      None,             60,
     ["reward_signal",   "motor_intent"],     0.40, 0.15),
    ("thalamus",         "Thalamus",           None,             50,
     ["visual_input",    "auditory_input",
      "somatosensory",   "touch_light",
      "touch_deep",      "touch_vibration",
      "thermal_warm",    "thermal_cold",
      "rhythmic"],                             0.30, 0.10),
    ("hippocampus",      "Hippocampus",        None,             70,
     ["episodic_input",  "spatial_input",
      "anxiety_anticipatory"],                0.25, 0.10),
    ("amygdala",         "Amygdala",           None,             50,
     ["threat_input",    "emotional_input",
      "pain_input",      "thermal_cold",
      "interoception",   "anxiety_anticipatory"], 0.20, 0.35),
    ("cerebellum",       "Cerebellum",         None,             80,
     ["motor_intent",    "touch_vibration",
      "rhythmic"],                             0.40, 0.10),
    ("brainstem",        "Brain Stem",         None,             40,
     ["pain_input",      "threat_input",
      "thermal_cold",    "interoception"],    0.20, 0.30),
]

# ── Inter-region connectivity (10×10, known functional connections) ───
# Row = source region index, Col = target region index
# Values: rough relative connection strengths [0, 1]
_REGION_ORDER = [s[0] for s in REGION_SPECS]

_INTER_W_TEMPLATE = np.array([
    # frt  par  tem  occ  bg   tha  hip  amy  cer  bst
    [0.0, 0.3, 0.4, 0.1, 0.5, 0.2, 0.3, 0.2, 0.3, 0.1],  # frontal
    [0.3, 0.0, 0.2, 0.3, 0.2, 0.3, 0.2, 0.1, 0.2, 0.1],  # parietal
    [0.4, 0.2, 0.0, 0.1, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1],  # temporal
    [0.1, 0.3, 0.1, 0.0, 0.1, 0.4, 0.1, 0.1, 0.1, 0.0],  # occipital
    [0.5, 0.2, 0.2, 0.1, 0.0, 0.3, 0.2, 0.3, 0.5, 0.2],  # basal ganglia
    [0.2, 0.3, 0.3, 0.4, 0.3, 0.0, 0.2, 0.2, 0.2, 0.2],  # thalamus
    [0.3, 0.2, 0.3, 0.1, 0.2, 0.2, 0.0, 0.4, 0.1, 0.1],  # hippocampus
    [0.2, 0.1, 0.2, 0.1, 0.3, 0.2, 0.4, 0.0, 0.1, 0.4],  # amygdala
    [0.3, 0.2, 0.1, 0.1, 0.5, 0.2, 0.1, 0.1, 0.0, 0.2],  # cerebellum
    [0.1, 0.1, 0.1, 0.0, 0.2, 0.2, 0.1, 0.4, 0.2, 0.0],  # brainstem
], dtype=np.float32)


def _make_weight_matrix(n: int, density: float = 0.15,
                        seed: Optional[int] = None) -> np.ndarray:
    """Build a sparse random weight matrix for a region's internal connections."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n), dtype=np.float32)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    # 80% excitatory, 20% inhibitory
    signs = np.where(rng.random((n, n)) < 0.8, 1.0, -1.0)
    W[mask] = (rng.random((n, n)) * 0.6 + 0.1)[mask] * signs[mask]
    # Row-normalize by p99 to keep firing stable
    vals = np.abs(W[W != 0])
    if len(vals):
        p99 = float(np.percentile(vals, 99))
        if p99 > 0:
            W = np.clip(W / p99, -1.0, 1.0)
    return W


def _make_channel_idx(n: int, channels: list[str],
                      seed: Optional[int] = None) -> dict:
    """Assign random neuron indices to each input channel."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    chunk = max(1, n // max(len(channels), 1))
    idx_map = {}
    for i, ch in enumerate(channels):
        start = i * chunk
        end   = min(start + chunk, n)
        idx_map[ch] = list(rng.choice(all_idx[start:end],
                           size=min(chunk, end - start),
                           replace=False))
    return idx_map


def build_region(spec: tuple, seed_base: int = 0) -> BrainRegion:
    region_id, label, parent, n, channels, fwd_frac, back_frac = spec
    seed = seed_base + abs(hash(region_id)) % 10000

    W = _make_weight_matrix(n, density=0.15, seed=seed)
    channel_idx = _make_channel_idx(n, channels, seed=seed + 1)

    rng = np.random.default_rng(seed + 2)
    all_idx = list(range(n))
    n_fwd  = max(1, int(n * fwd_frac))
    n_back = max(1, int(n * back_frac))
    fwd_idx  = list(rng.choice(all_idx, size=n_fwd,  replace=False))
    back_idx = list(rng.choice(
        [i for i in all_idx if i not in fwd_idx],
        size=min(n_back, n - n_fwd), replace=False
    ))

    pop = NeuronPopulation(
        region_id   = region_id,
        n_neurons   = n,
        W           = W,
        channel_idx = channel_idx,
        fwd_idx     = fwd_idx,
        back_idx    = back_idx,
    )

    vectors = RegionVectors(
        semantic     = np.zeros(512, dtype=np.float32),
        activity     = np.zeros(n,   dtype=np.float32),
        memory       = np.zeros(512, dtype=np.float32),
        connectivity = _INTER_W_TEMPLATE[_REGION_ORDER.index(region_id)].copy(),
    )

    return BrainRegion(id=region_id, label=label,
                       parent=parent, population=pop, vectors=vectors)


def build_all_regions() -> tuple[list[BrainRegion], np.ndarray]:
    """Return (regions list, inter-region weight matrix)."""
    regions = [build_region(spec) for spec in REGION_SPECS]
    return regions, _INTER_W_TEMPLATE.copy()
