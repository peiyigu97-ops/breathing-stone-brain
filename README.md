# CHIMERA + HumanBrainDT

**A digital lifeform and human brain simulation for anxiety research**

![neurons](https://img.shields.io/badge/CHIMERA_neurons-1%2C373-blue)
![synapses](https://img.shields.io/badge/synapses-22%2C400-blue)
![human_regions](https://img.shields.io/badge/HumanBrain_regions-16-purple)
![sensory_channels](https://img.shields.io/badge/sensory_channels-28-blueviolet)
![version](https://img.shields.io/badge/version-4.0.0-brightgreen)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Jetson-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What this is

This repo contains two interconnected systems:

**CHIMERA** — A digital lifeform built from the real *Drosophila* larva connectome (Winding et al., *Science* 2023). 1,373 LIF neurons and 22,400 synapses drive a MuJoCo physics body. Type anything in any language → sensory channels fire → emergent motor behavior.

**HumanBrainDT** — A human brain digital twin with **16 anatomically-positioned regions**, LIF simulation extended for anxiety-sensory research. Models the full Autonomic Nervous System (sympathetic/parasympathetic/HPA axis), anxiety decomposition (baseline, anticipatory, somatic, regulation), 28 sensory/neuromodulator channels, and anatomically correct neural pathways based on MNI atlas coordinates.

**Galaxy View 3D visualiser (v4)** — PyQt6 native window, QWebChannel communication, no browser/server required. Each brain region renders as a primary star + nebula cloud with GPU shader drift. Signal propagation uses fiber-optic pulse architecture. Rendering uses merged draw-call pools and throttled DOM updates for smooth 30fps.

**Breathing Stone Bridge (`stone_bridge.py`)** — Connects a physical haptic device (Breathing Stone, ESP32-based) to HumanBrainDT. Sensor data (skin temperature, grip force) → Polyvagal state inference → LIF simulation → intervention parameters fed back to the device.

---

## Quick start

### v4 — PyQt6 native window (recommended)

```bash
pip install PyQt6 PyQt6-WebEngine numpy
cd v4
python main.py
```

### v3 — Browser viewer (WebSocket)

```bash
pip install fastapi uvicorn numpy
python -m uvicorn HumanBrainDT.viewer.server:app --host 127.0.0.1 --port 7860
# → http://localhost:7860
```

Type any sensory or emotional stimulus — Chinese and English both work:

```
danger          → amygdala + brainstem + locus coeruleus cascade
panic attack    → threat + anticipatory anxiety + LC broadcast
deep pressure   → insula + parietal + vagal pathway
breathing       → cerebellum + thalamus + frontal regulation
nausea          → brainstem + insula + hypothalamus
exhausted       → hypothalamus + locus coeruleus
joy             → substantia nigra + basal ganglia + orbitofrontal
heart racing    → insula + hypothalamus + amygdala
pain            → insula + anterior cingulate + amygdala
```

---

## Brain Regions (v2.0+)

| Region | Role | Color |
|--------|------|-------|
| Frontal Cortex | Executive function, language, regulation | `#4a90d9` steel blue |
| Parietal Cortex | Somatosensory, spatial processing | `#7ec8e3` sky blue |
| Temporal Cortex | Auditory, language, memory | `#48d1a0` seafoam |
| Occipital Cortex | Visual processing | `#38c0c8` teal |
| **Insula** *(new)* | Interoception, pain affect, autonomic drive | `#b060d8` violet |
| **Anterior Cingulate** *(new)* | Conflict monitoring, pain emotion, anxiety regulation | `#e070a0` rose |
| **Orbitofrontal** *(new)* | Reward valuation, impulse inhibition | `#f0a040` amber |
| Basal Ganglia | Motor initiation, reward learning | `#d84848` red |
| Thalamus | Sensory relay, thalamocortical loops | `#a0c060` lime |
| Hippocampus | Episodic memory, spatial navigation | `#40a8e0` azure |
| Amygdala | Threat detection, fear, emotional salience | `#e04060` crimson |
| **Hypothalamus** *(new)* | HPA axis pacemaker, thermoregulation, fatigue | `#f08020` orange |
| **Locus Coeruleus** *(new)* | Norepinephrine broadcast, stress arousal | `#30d8f0` cyan |
| **Substantia Nigra** *(new)* | Dopamine pathway, reward prediction | `#e8c020` gold |
| Cerebellum | Motor coordination, rhythmic timing | `#50d070` green |
| Brain Stem | Autonomic baseline, ascending arousal | `#a0b8d0` slate |

---

## Architecture

```
[Physical device / demo]
  ESP32: skin_temp, grip_n, contact, rhythm_phase
        │        ▼  stone_bridge.py
  Polyvagal state inference (ventral / sympathetic / dorsal)
        │        ▼   HumanBrainDT LIF simulation (16 regions, 945 neurons, ~9,500 synapses)
  ├─ ANS: sympathetic, parasympathetic, HRV, cortisol
  │      (LC norepinephrine broadcast, hypothalamic HPA drive)
  ├─ Anxiety: baseline, anticipatory, somatic, regulation, healing_index
  │          (insula somatic, ACC conflict, PFC–amygdala regulation)
  └─ Sensory (28 channels): touch, thermal, interoception, rhythmic,
              nausea, muscle tension, fatigue, stress hormone, dopamine,
              olfactory, taste, conflict, threat, reward, pain, ...
        │        ├──▶  ESP32 intervention command (mode, temp_c, breath ratio, trigger)
        ├──▶  TGAM CSV  (stone_session.csv)
        ├──▶  Training log  (stone_session_training.jsonl)
        └──▶  v4: QWebChannel → PyQt6 Galaxy View (python main.py)
             v3: WebSocket  → Galaxy View browser  (http://localhost:7860)
```

---

## Repository structure

```
chimera/
  chimera_app.py              CHIMERA main (Drosophila brain + MuJoCo)
  chimera_load_connectome.py  Connectome parser (Winding 2023 data)
  chimera_real_connectome.py  Auto-downloader for connectome data
  chimera_mcp.py              MCP server wrapper for CHIMERA
  setup.py                    One-command installer
  stone_bridge.py             Breathing Stone → HumanBrainDT bridge
  tgam_csv_generator.py       TGAM-format CSV generator + ESP32 sensor mapping
  jetson_setup.sh             Jetson Nano deployment script
  jetson_requirements.txt     Jetson-specific dependencies
  CHANGELOG.md                Version history and iteration notes
  ACKNOWLEDGMENTS.md          Full attribution for all libraries, data, papers

v4/                           → v4 PyQt6 native window (recommended)
  main.py                     Entry point — PyQt6 + QWebChannel
  brain_bridge.py             BrainBridge / BrainWorker (QThread)
  static_server.py            Local HTTP server for static assets
  requirements.txt
  HumanBrainDT/               (same structure as below)
  static/
    galaxy.html               Galaxy View 3D (Three.js, QWebChannel mode)
    js/
      three.module.js
      addons/OrbitControls.js
      qwebchannel.js

HumanBrainDT/
  brain.py                    Top-level HumanBrain class + 120+ keyword KMAP
  core/
    lif_engine.py             LIF simulator, 28-channel routing, ANS/anxiety
    signal.py                 BrainState, ANSState, AnxietyState, SensoryState
    region.py                 BrainRegion + NeuronPopulation
    physio_clock.py           Per-region physiological oscillation profiles
  regions/
    builder.py                Builds 16-region brain with inter-region weights
  viewer/
    server.py                 FastAPI WebSocket server + REST API (v3)
    static/
      index.html              Galaxy View 3D brain visualiser (v3, browser)
      index_v1_particle_3d.html   Archived: anatomical particle cloud (v1)
      index_v2_galaxy_cluster.html Archived: sphere cluster (v2)
      js/
        three.module.js
        addons/OrbitControls.js
  server/
    brain_mcp.py              MCP server for HumanBrainDT (8 tools)
```

---

## Quick start

### v4 — PyQt6 native window (recommended)

```bash
pip install PyQt6 PyQt6-WebEngine numpy
cd v4
python main.py
```

### v3 — Browser viewer (WebSocket)

```bash
pip install fastapi uvicorn[standard] websockets numpy
python3 HumanBrainDT/viewer/server.py
# → http://localhost:7860
```

### CHIMERA (Drosophila brain + MuJoCo body)

```bash
curl -O https://raw.githubusercontent.com/caparison1234/chimera/main/setup.py
python3 setup.py
python3 chimera_app.py
```

### Breathing Stone bridge (demo mode)

```bash
pip install -r jetson_requirements.txt
python3 stone_bridge.py --demo --steps 120
# or live ESP32:
python3 stone_bridge.py --serial COM3 --out session.csv
```

---

## Sensory channels (28 total)

| Channel | Pathway | Healing weight |
|---------|---------|----------------|
| `touch_deep` | Parietal → Thalamus → Insula → Cerebellum | +0.85 |
| `thermal_warm` | Insula → Hypothalamus → Thalamus | +0.75 |
| `rhythmic` | Cerebellum → Thalamus → Frontal | +0.70 |
| `touch_vibration` | Parietal → Cerebellum → Thalamus | +0.60 |
| `interoception` | Insula → Hypothalamus → Amygdala → ACC | +0.55 |
| `touch_light` | Parietal → Thalamus → Insula | +0.35 |
| `dopamine_signal` | Substantia Nigra → Basal Ganglia → OFC | +0.50 |
| `reward_signal` | Basal Ganglia → OFC → Substantia Nigra | +0.55 |
| `nausea_input` | Brainstem → Insula → Hypothalamus | −.55 |
| `muscle_tension` | Insula → Cerebellum → Parietal | −.40 |
| `stress_hormone` | Hypothalamus → LC → Amygdala | −.70 |
| `fatigue_input` | Hypothalamus → LC → Brainstem | −.25 |
| `thermal_cold` | Brainstem → Amygdala → Hypothalamus | −.30 |
| `pain_input` | Brainstem → Insula → Amygdala → ACC | −.90 |
| `threat_input` | Amygdala → Brainstem → LC | −.00 |
| `anxiety_anticipatory` | Frontal → ACC → Amygdala → Hippocampus | −.80 |

---

## TGAM CSV format

`stone_session.csv` follows NeuroSky ThinkGear ASIC Module output conventions.  
See full field mapping in the original README or `tgam_csv_generator.py`.

---

## Polyvagal state inference

| State | Condition | Intervention |
|-------|-----------|-------------|
| **ventral** (calm) | skin temp ≈baseline, grip ≈baseline | default 4→ 6→ at 35°C |
| **sympathetic** (anxiety) | skin temp < baseline−°C AND grip > baseline+1SD | calm: 4→ 8→ at 34°C |
| **dorsal** (freeze) | skin temp < baseline−°C AND grip < baseline−.5SD | activate: 5→ 4→ at 37°C |

---

## Neuroscience references

Healing pathway weights and neural routing in `HumanBrainDT/core/lif_engine.py` are derived from peer-reviewed literature. See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) for the full reference list with DOIs.

Key sources:
- Winding et al. (2023) — connectome data
- Porges (1995, 2022) — Polyvagal Theory
- McGlone et al. (2014) — CT afferents / oxytocin
- Craig (2002) — interoception / temperature
- Seth (2013) — interoceptive prediction coding
- McEwen (1998) — allostatic load
- LeDoux (1996) — amygdala threat circuit
- Etkin et al. (2015) — PFC–amygdala regulation

---

## Acknowledgements

See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md) for full attribution of all libraries, datasets, models, and scientific references used in this project.

---

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md) for version history and iteration notes.

---

## License

MIT — see [`LICENSE`](LICENSE).

The connectome data (Winding et al. 2023) is used under CC BY 4.0.  
The Qwen model weights are used under Qwen License (non-commercial research use).  
MuJoCo is used under Apache 2.0.  
Three.js is used under MIT.
