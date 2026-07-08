"""
core/region.py — BrainRegion, NeuronPopulation, RegionVectors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RegionVectors:
    semantic:     np.ndarray   # (512,) functional identity embedding
    activity:     np.ndarray   # (n_neurons,) current spike rates
    memory:       np.ndarray   # (512,) accumulated memory trace
    connectivity: np.ndarray   # (n_regions,) inter-region weights


@dataclass
class NeuronPopulation:
    region_id:   str
    n_neurons:   int
    W:           np.ndarray    # (n×n) intra-region synaptic weight matrix
    channel_idx: dict          # channel_name → [neuron_idx, ...]
    fwd_idx:     list          # excitatory output indices
    back_idx:    list          # inhibitory / withdrawal output indices
    type_map:    dict = field(default_factory=dict)  # neuron_id → type label


@dataclass
class BrainRegion:
    id:         str
    label:      str
    parent:     Optional[str]
    population: NeuronPopulation
    vectors:    Optional[RegionVectors] = None

    def spike_count(self) -> int:
        if self.vectors is None:
            return 0
        return int(self.vectors.activity.sum())
