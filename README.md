# CHIMERA + HumanBrainDT

**A digital lifeform and human brain simulation for anxiety research**

![neurons](https://img.shields.io/badge/CHIMERA_neurons-1%2C373-blue)
![synapses](https://img.shields.io/badge/synapses-22%2C400-blue)
![human_regions](https://img.shields.io/badge/HumanBrain_regions-16-purple)
![sensory_channels](https://img.shields.io/badge/sensory_channels-28-blueviolet)
![platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Jetson-lightgrey)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What this is

This repo contains two interconnected systems:

**CHIMERA** — A digital lifeform built from the real *Drosophila* larva connectome (Winding et al., *Science* 2023). 1,373 LIF neurons and 22,400 synapses drive a MuJoCo physics body. Type anything in any language → sensory channels fire → emergent motor behavior.

**HumanBrainDT** — A human brain digital twin with **16 anatomically-positioned regions**, LIF simulation extended for anxiety-sensory research. Models the full Autonomic Nervous System (sympathetic/parasympathetic/HPA axis), anxiety decomposition (baseline, anticipatory, somatic, regulation), 28 sensory/neuromodulator channels, and correct anatomical neural pathways based on MNI atlas coordinates.

A **Galaxy View 3D visualiser** renders each brain region as a primary star + nebula cloud, with layered activation: neural filaments light first, then the primary star, then the nebula trails behind with a ~1.5s delay.

**Breathing Stone Bridge (`stone_bridge.py`)** — Connects a physical haptic device (Breathing Stone, ESP32-based) to HumanBrainDT. Sensor data (skin temperature, grip force) → Polyvagal state inference → LIF simulation → intervention parameters fed back to the device.

---

## Brain Regions (v2.0)

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
        │
        ▼  stone_bridge.py
  Polyvagal state inference (ventral / sympathetic / dorsal)
        │
        ▼
  HumanBrainDT LIF simulation (16 regions, 945 neurons, ~9,500 synapses)
  ├─ ANS: sympathetic, parasympathetic, HRV, cortisol
  │       (LC norepinephrine broadcast, hypothalamic HPA drive)
  ├─ Anxiety: baseline, anticipatory, somatic, regulation, healing_index
  │           (insula somatic, ACC conflict, PFC–amygdala regulation)
  └─ Sensory (28 channels): touch, thermal, interoception, rhythmic,
              nausea, muscle tension, fatigue, stress hormone, dopamine,
              olfactory, taste, conflict, threat, reward, pain, ...
        │
        ├──▶  ESP32 intervention command (mode, temp_c, breath ratio, trigger)
        ├──▶  TGAM CSV  (stone_session.csv)
        ├──▶  Training log  (stone_session_training.jsonl)
        └──▶  WebSocket → Galaxy View 3D brain viewer (http://localhost:7860)
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
  stone_bridge.py             Breathing Stone ↔ HumanBrainDT bridge
  tgam_csv_generator.py       TGAM-format CSV generator + ESP32 sensor mapping
  jetson_setup.sh             Jetson Nano deployment script
  jetson_requirements.txt     Jetson-specific dependencies
  CHANGELOG.md                Version history and iteration notes
  ACKNOWLEDGMENTS.md          Full attribution for all libraries, data, papers

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
    server.py                 FastAPI WebSocket server + REST API
    static/
      index.html              Galaxy View 3D brain visualiser (Three.js)
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

### HumanBrainDT viewer (Galaxy View)

```bash
pip install fastapi uvicorn[standard] websockets numpy
python3 HumanBrainDT/viewer/server.py
# → http://localhost:7860
```

Type sensory or emotional stimuli in the input box — Chinese and English both work:

```
danger          → amygdala + brainstem + locus coeruleus activate
panic attack    → threat + anticipatory anxiety + LC broadcast
deep pressure   → insula + parietal + vagal pathway
breathing       → cerebellum + thalamus + frontal regulation
nausea          → brainstem + insula + hypothalamus
exhausted       → hypothalamus + locus coeruleus
joy             → substantia nigra + basal ganglia + orbitofrontal
heart racing    → insula + hypothalamus + amygdala
pain            → insula + anterior cingulate + amygdala
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
| `nausea_input` | Brainstem → Insula → Hypothalamus | −0.55 |
| `muscle_tension` | Insula → Cerebellum → Parietal | −0.40 |
| `stress_hormone` | Hypothalamus → LC → Amygdala | −0.70 |
| `fatigue_input` | Hypothalamus → LC → Brainstem | −0.25 |
| `thermal_cold` | Brainstem → Amygdala → Hypothalamus | −0.30 |
| `pain_input` | Brainstem → Insula → Amygdala → ACC | −0.90 |
| `threat_input` | Amygdala → Brainstem → LC | −1.00 |
| `anxiety_anticipatory` | Frontal → ACC → Amygdala → Hippocampus | −0.80 |

---

## TGAM CSV format

`stone_session.csv` follows NeuroSky ThinkGear ASIC Module output conventions.  
See full field mapping in the original README or `tgam_csv_generator.py`.

---

## Polyvagal state inference

| State | Condition | Intervention |
|-------|-----------|-------------|
| **ventral** (calm) | skin temp ≈ baseline, grip ≈ baseline | default 4↑/6↓ at 35°C |
| **sympathetic** (anxiety) | skin temp < baseline−1°C AND grip > baseline+1SD | calm: 4↑/8↓ at 34°C |
| **dorsal** (freeze) | skin temp < baseline−1°C AND grip < baseline−0.5SD | activate: 5↑/4↓ at 37°C |

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

---

## What this is

This repo contains two interconnected systems:

**CHIMERA** — A digital lifeform built from the real *Drosophila* larva connectome (Winding et al., *Science* 2023). 1,373 LIF neurons and 22,400 synapses drive a MuJoCo physics body. Type anything in any language → sensory channels fire → emergent motor behavior.

**HumanBrainDT** — A human brain digital twin with 10 regions, LIF simulation extended for anxiety-sensory healing research. Models the Autonomic Nervous System (sympathetic/parasympathetic), anxiety decomposition (baseline, anticipatory, somatic, regulation), and sensory healing pathways (CT fibers → oxytocin, vagal cold reflex, interoceptive prediction coding).

**Breathing Stone Bridge (`stone_bridge.py`)** — Connects a physical haptic device (Breathing Stone, ESP32-based) to HumanBrainDT. Sensor data (skin temperature, grip force) → Polyvagal state inference → LIF simulation → intervention parameters fed back to the device. Logs TGAM-format CSV and JSONL training records.

---

## Architecture

```
[Physical device / demo]
  ESP32: skin_temp, grip_n, contact, rhythm_phase
        │
        ▼  stone_bridge.py
  Polyvagal state inference (ventral / sympathetic / dorsal)
        │
        ▼
  HumanBrainDT LIF simulation (10 regions, 690 neurons)
  ├─ ANS: sympathetic, parasympathetic, HRV, cortisol
  ├─ Anxiety: baseline, anticipatory, somatic, regulation, healing_index
  └─ Sensory: touch_deep, vibration, thermal_cold, rhythmic, interoception
        │
        ├──▶  ESP32 intervention command (mode, temp_c, breath ratio, trigger)
        ├──▶  TGAM CSV  (stone_session.csv)
        ├──▶  Training log  (stone_session_training.jsonl)
        └──▶  WebSocket → 3D brain viewer (http://localhost:7860)
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
  stone_bridge.py             Breathing Stone ↔ HumanBrainDT bridge
  tgam_csv_generator.py       TGAM-format CSV generator + ESP32 sensor mapping
  jetson_setup.sh             Jetson Nano deployment script
  jetson_requirements.txt     Jetson-specific dependencies

HumanBrainDT/
  brain.py                    Top-level HumanBrain class
  core/
    lif_engine.py             LIF simulator, ANS/anxiety/sensory computation
    signal.py                 BrainState, ANSState, AnxietyState dataclasses
    region.py                 BrainRegion + NeuronPopulation
  regions/
    builder.py                Builds 10-region brain with inter-region weights
  viewer/
    server.py                 FastAPI WebSocket server + /api/csv + /api/log
    static/index.html         3D particle brain viewer (Three.js)
  server/
    brain_mcp.py              MCP server for HumanBrainDT (8 tools)
```

---

## Quick start

### CHIMERA (Drosophila brain + MuJoCo body)

```bash
curl -O https://raw.githubusercontent.com/caparison1234/chimera/main/setup.py
python3 setup.py
python3 chimera_app.py
```

Type `danger`, `run`, `food smell`, `breathe` etc. in any language.  
Without the Qwen model, runs in keyword-matching mode (still fully functional).

### HumanBrainDT viewer

```bash
pip install fastapi uvicorn[standard] websockets numpy
python3 HumanBrainDT/viewer/server.py
# → http://localhost:7860
```

Type sensory stimuli: `deep pressure`, `warm`, `breathing`, `anxiety`, `danger`, `calm`

### Breathing Stone bridge (demo mode)

```bash
pip install -r jetson_requirements.txt
python3 stone_bridge.py --demo --steps 120
# or live ESP32:
python3 stone_bridge.py --serial COM3 --out session.csv
```

Generates `stone_session.csv` (TGAM format) and `stone_session_training.jsonl`.

---

## TGAM CSV format

`stone_session.csv` follows NeuroSky ThinkGear ASIC Module output conventions:

| Field | Type | Range | Mapping source |
|-------|------|-------|---------------|
| timestamp | float | Unix time | `brain_state.timestamp` |
| signal_quality | int | 0–50 | `anxiety.baseline × 50` |
| attention | int | 0–100 | `motor_command.approach × 100` |
| meditation | int | 0–100 | `ans.parasympathetic × (1−sym×0.6) × 100` |
| delta | int | 0–65535 | `anxiety.baseline` |
| theta | int | 0–65535 | `anxiety.anticipatory` |
| low_alpha | int | 0–65535 | `anxiety.healing_index × 0.5` |
| high_alpha | int | 0–65535 | `ans.hrv_index` |
| low_beta | int | 0–65535 | `ans.sympathetic` |
| high_beta | int | 0–65535 | `motor_command.avoid` |
| low_gamma | int | 0–65535 | `motor_command.engage` |
| mid_gamma | int | 0–65535 | `sensory.rhythmic` |
| raw_eeg | int | −2048–2047 | synthesised (alpha/beta/theta mix) |

---

## Polyvagal state inference

`stone_bridge.py` infers one of three Polyvagal states from sensor data:

| State | Condition | Intervention |
|-------|-----------|-------------|
| **ventral** (calm) | skin temp ≈ baseline, grip ≈ baseline | default 4↑/6↓ at 35°C |
| **sympathetic** (anxiety) | skin temp < baseline−1°C AND grip > baseline+1SD | calm mode: 4↑/8↓ at 34°C |
| **dorsal** (freeze) | skin temp < baseline−1°C AND grip < baseline−0.5SD | activate mode: 5↑/4↓ at 37°C |

The personal baseline builds over time using exponential weighted moving average (α = 0.02, ~50-sample window). First 7 days uses population defaults.

---

## Jetson Nano deployment

```bash
chmod +x jetson_setup.sh
./jetson_setup.sh
# Then:
python3 stone_bridge.py --serial /dev/ttyUSB0 --out /tmp/session.csv
python3 HumanBrainDT/viewer/server.py
# → http://<jetson_ip>:7860
```

MuJoCo (`chimera_app.py`) does **not** run on Jetson — no display output. Use the HumanBrainDT viewer (browser-based) instead.

---

## Neuroscience references

The healing pathway weights in `lif_engine.py` (`HEALING_WEIGHTS`) are derived from:

| Channel | Weight | Source |
|---------|--------|--------|
| `touch_deep` | 0.85 | McGlone et al. (2014) CT fibers → oxytocin → HPA axis inhibition; Uvnäs-Moberg et al. (2014) |
| `thermal_warm` | 0.75 | Ijzerman & Semin (2009); Craig (2002) warm → insula → parasympathetic |
| `rhythmic` | 0.70 | Porges (1995) vagal tone ↑ via rhythmic breathing; Kleitman (1963) BRAC |
| `touch_vibration` | 0.60 | Krauss (1987) deep pressure; vibration → cerebellum → thalamic gating |
| `interoception` | 0.45 | Seth (2013) interoceptive prediction coding; Price & Hooven (2018) MABT |
| `thermal_cold` | −0.30 | Cold Face Test (PubMed 2022) — mild cold activates brainstem/amygdala |
| `threat_input` | −1.00 | LeDoux (1996) amygdala threat circuit |

ANS computation follows Porges (2022) Polyvagal Theory: sympathetic/parasympathetic balance, HRV as vagal tone proxy.

Anxiety decomposition follows:
- **baseline**: allostatic overload (McEwen 1998)
- **anticipatory**: prefrontal–amygdala worry loop (Etkin et al. 2015)
- **somatic**: insula interoception (Paulus & Stein 2006)
- **regulation**: PFC → amygdala top-down control (Etkin et al. 2015)
- **healing_index**: net calming effect of current sensory input

---

## Acknowledgements

**Connectome data (CHIMERA)**  
Winding, M., Pedigo, B. D., Barnes, C. L., Patsolic, H. G., Park, Y., Krieger, T., … Zlatic, M. (2023). The connectome of an insect brain. *Science*, 379(6636). https://doi.org/10.1126/science.add9330  
Data: https://github.com/brain-networks/larval-drosophila-connectome

**Language model (input parser)**  
Qwen2.5-0.5B-Instruct, Alibaba Cloud. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF  
Used only for natural language → sensory channel JSON parsing (not for output generation).

**Physics simulation**  
MuJoCo (Todorov et al., 2012). https://mujoco.org

**LLM inference runtime**  
llama.cpp. https://github.com/ggerganov/llama.cpp  
Python binding: llama-cpp-python. https://github.com/abetlen/llama-cpp-python

**3D visualisation**  
Three.js. https://threejs.org

**Neuroscience frameworks used in HumanBrainDT design**  
- Porges, S. W. (1995/2022). Polyvagal Theory. *Psychophysiology* / *Frontiers in Integrative Neuroscience*.  
- Seth, A. K. (2013). Interoceptive inference. *Trends in Cognitive Sciences*, 17(11).  
- McEwen, B. S. (1998). Allostasis. *New England Journal of Medicine*, 338.  
- McGlone, F., Wessberg, J., & Olausson, H. (2014). CT afferents. *Neuron*, 82(4).  
- Craig, A. D. (2002). Temperature and interoception. *Nature Neuroscience*, 3(2).

**No code was copied from any of these sources.** The neuroscience literature was used strictly to derive parameter values and architectural decisions in the simulation model. All code is original.

---

## License

MIT — see LICENSE file.

The connectome data (Winding et al. 2023) is used under its original CC BY 4.0 license.  
The Qwen model weights are used under Qwen License (non-commercial research use).  
MuJoCo is used under its Apache 2.0 license.
