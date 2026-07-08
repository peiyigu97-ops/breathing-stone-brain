# CHIMERA — AI Handoff Document

> This document enables any AI or developer to continue CHIMERA development without prior context.
> Last updated: 2026-03-21

---

## 1. Project Purpose

CHIMERA is a digital lifeform that uses the **real Drosophila larva connectome** (Winding et al., Science 2023) as its brain, connected to a MuJoCo physics body. It is part of a larger project roadmap:

```
GENESIS → CHIMERA → AION
```

| Phase | Description | Status |
|-------|-------------|--------|
| GENESIS | PPO-trained locomotion creature | ✅ Done |
| CHIMERA | Real connectome brain + MuJoCo body | ✅ Done |
| AION Alpha | Reaction-diffusion layer over connectome | 🔜 Planned |
| AION Beta | Evolutionary body generation | 🔜 Planned |
| AION Release | Hebbian synaptic learning | 🔜 Planned |

---

## 2. User Flow

```
User types text (any language)
        ↓
Qwen 0.5B parses input → sensory channel JSON
        ↓
1,373 LIF neurons simulate connectome (200 steps)
        ↓
Motor signals extracted (forward, backward, curl, eat, tremble, wing)
        ↓
MuJoCo 12-actuator body responds physically
        ↓
Neuron-type firing pattern → first-person experience sentence
(language matches input: Korean / English / Japanese / Chinese)
```

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     chimera_app.py                       │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────┐ │
│  │  Voice   │   │  Brain   │   │    MuJoCo Viewer     │ │
│  │ (Qwen    │   │  (LIF    │   │  (12 actuators)      │ │
│  │  parser) │   │  sim)    │   │                      │ │
│  └────┬─────┘   └────┬─────┘   └──────────┬───────────┘ │
│       │              │                     │             │
│  text → stim dict → sig dict → apply_controls()         │
│                                                          │
│  State (shared, thread-safe):                           │
│    sig / stim / speech                                   │
└─────────────────────────────────────────────────────────┘
```

**Threading:**
- Main thread: MuJoCo physics loop
- Daemon thread: user input loop

---

## 4. Feature List

### Working ✅
- 1,373 neuron LIF simulation (Winding 2023 connectome)
- 22,400 synapses with p99-normalized weight matrix
- Sensory channels: touch_front, touch_back, nociception, chemical, olfactory, visual
- Motor signals: forward, backward, wing, curl, eat, tremble
- 12 MuJoCo actuators (4 wings + 6 legs + abdomen + head)
- Behavioral patterns:
  - `curl`: nociception → legs retract + abdomen raises
  - `eat`: chemical + low movement → head scans left/right
  - `tremble`: fwd>0.5 AND back>0.5 simultaneously
- Qwen 0.5B as **input parser only** (text → sensory JSON)
- Rule-based fallback (KMAP: 150+ keywords, 4 languages)
- Neuron-type → first-person experience translation
- Auto language detection (Korean/English/Japanese/Chinese)
- Identity response when no firing ("I am CHIMERA...")
- Goodbye response detection
- Windows + macOS support (mjpython auto-relaunch)
- One-command install: `python3 setup.py`
- GitHub: https://github.com/caparison1234/chimera

### Not Implemented ❌
- AION reaction-diffusion layer
- Hebbian learning
- Evolutionary body generation
- Continuous neural control (currently: 200-step batch → motor signal)
- Qwen output generation (intentionally disabled — too small for reliable Korean output)

---

## 5. Current Implementation State

### Brain (LIF Simulation)
```python
TAU=8.0, V_TH=-52.0, V_REST=-70.0, V_RESET=-75.0, REFRAC=2, DT=1.0
I_syn = W.T @ spikes * 400.0
I_ext = val * 80.0  (per sensory channel)
steps = 200, I_ext *= 0.995 (decay per step)
Weight normalization: p99 percentile clipping
```

### Motor Signal Extraction
```python
# Forward signal
FWD_TYPES  = {'PN-somato', 'LHN', 'PN', 'MB-FBN'}  → fwd=334 neurons
BACK_TYPES = {'ascending', 'MBON'}                   → back=47 neurons
rate = tanh(total_spikes / (steps * 2.0))

# Behavioral signals
curl    = tanh(noci_fired / (noci_count * steps * 0.05) * 2.0)
eat     = tanh(chem_fired / (chem_count * steps * 0.05) * 2.0) * (1 - min(fwd,0.5)*2)
tremble = tanh(min(fwd,back)*4.0)  if fwd>0.5 AND back>0.5 AND total>200
```

### Connectome Data
```
Source: Winding et al., Science 2023
Repo:   github.com/brain-networks/larval-drosophila-connectome
File:   Supplementary-Data-S1.zip
  → ad_connectivity_matrix.csv  (2952×2953 adjacency)
  → annotations.csv             (neuron metadata)
Using:  first 1373×1373 submatrix
```

### Neuron Type Distribution (1,373 neurons)
```
pre-DN-VNC: 238  sensory: 235  KC: 121  PN: 103  LHN: 101
DN-VNC: 91  DN-SEZ: 82  PN-somato: 76  LN: 56  MB-FBN: 54
pre-DN-SEZ: 51  CN: 50  ascending: 23  MBON: 24  MBIN: 14
```

### Sensory Channel Mapping
```
touch_front : 47 neurons  (sensory, mechano-ch)
touch_back  : 47 neurons  (sensory, mechano-ch)
nociception : 44 neurons  (noci 2nd_order pn)
chemical    : 230 neurons (gustatory-external/pharyngeal, gut)
olfactory   : 43 neurons  (olfactory)
visual      : 38 neurons  (visual)
```

---

## 6. Code Structure

```
chimera/                          ← project root
  chimera_app.py                  ← MAIN (all-in-one)
  chimera_load_connectome.py      ← connectome parser
  chimera_real_connectome.py      ← data downloader
  setup.py                        ← one-click installer
  requirements.txt
  .gitignore
  README.md
  chimera.spec                    ← PyInstaller build
  build.bat                       ← Windows .exe builder
  chimera/
    connectome/
      real_weight_matrix.npy      ← parsed weight matrix (gitignored)
      real_neurons.json           ← neuron metadata (gitignored)
    models/
      Qwen2.5-0.5B-Instruct.Q4_K_M.gguf  ← LLM (gitignored)
```

### chimera_app.py Class Structure
```
resource_path()         ← PyInstaller path resolver
ensure_fallback()       ← C. elegans fallback connectome generator

Brain                   ← LIF neuron simulation
  __init__()            ← load .npy + .json, map channels/motors
  stimulate()           ← apply sensory input
  step()                ← single LIF timestep
  run()                 ← 200-step simulation → signal dict

detect_lang()           ← language detection (ko/en/ja/zh)
NEURON_EXP              ← neuron type → experience label table
SENSE_EXP               ← sensory channel → experience label
MOVE_EXP                ← movement → experience label
INTENSITY               ← spike count → intensity adverb

Voice                   ← output + input parsing
  __init__()            ← load Qwen (optional)
  _experience()         ← neuron firing → first-person sentence
  speak()               ← main output method

KMAP / _STRENGTH        ← keyword fallback parser (150+ keywords)
_PARSE_SYSTEM           ← Qwen input parser system prompt
make_parser()           ← returns Qwen or KMAP parser function

apply_controls()        ← signal dict → 12 MuJoCo actuators
State                   ← thread-safe shared state
main()                  ← entry point (macOS mjpython + MuJoCo loop)
```

---

## 7. API / Data Structures

### Sensory Input (stim dict)
```python
{
  "touch_front": 0.0–1.0,
  "touch_back":  0.0–1.0,
  "nociception": 0.0–1.0,
  "chemical":    0.0–1.0,
  "olfactory":   0.0–1.0,
  "visual":      0.0–1.0,
}
```

### Motor Signal Output (sig dict)
```python
{
  "forward":       0.0–1.0,
  "backward":      0.0–1.0,
  "turn_left":     0.0–1.0,
  "turn_right":    0.0–1.0,
  "wing":          0.0–1.0,
  "active":        0.0–1.0,
  "curl":          0.0–1.0,
  "eat":           0.0–1.0,
  "tremble":       0.0–1.0,
  "_total_spikes": int,
  "_type_fires":   {"PN-somato": int, "LHN": int, ...},
}
```

### Qwen Input Parser Output
```json
{"touch_front": 0.9}
{"nociception": 1.0, "touch_back": 0.7}
{}
```

### MuJoCo Actuator Mapping
```
ctrl[0]  aWFL  wing front-left
ctrl[1]  aWFR  wing front-right
ctrl[2]  aWBL  wing back-left
ctrl[3]  aWBR  wing back-right
ctrl[4]  aLFL  leg front-left
ctrl[5]  aLFR  leg front-right
ctrl[6]  aLML  leg mid-left
ctrl[7]  aLMR  leg mid-right
ctrl[8]  aLBL  leg back-left
ctrl[9]  aLBR  leg back-right
ctrl[10] aABD  abdomen bend   ← curl/eat/backward
ctrl[11] aHEAD head direction ← eat scan / turn
```

### real_neurons.json Structure
```json
{
  "source": "Drosophila larva (Winding 2023)",
  "neurons": ["neuron_id", ...],
  "types":   {"neuron_id": "PN-somato", ...},
  "index":   {"neuron_id": 0, ...},
  "count":   1373,
  "channels": {
    "touch_front": ["id1", ...],
    "chemical":    ["id2", ...]
  }
}
```

---

## 8. Known Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| fwd/back values always similar | Low | Connectome circuit property — same neurons fire for most inputs |
| Qwen output disabled | Low | 0.5B too small for reliable Korean instruction following |
| `Task policy set failed` on macOS | Info | macOS system warning, harmless |
| `n_ctx_per_seq (256) < n_ctx_train` | Info | Qwen context warning, harmless for JSON parsing |
| No persistent memory between sessions | Medium | AION phase will address this |

---

## 9. Next Priority Tasks

### Immediate (CHIMERA improvements)
1. **Upgrade to Qwen 3B** for output generation (RAM: 16GB OK, CPU-only)
   - Re-enable `speak()` LLM path with 3B model
   - Model: `qwen2.5-3b-instruct-q4_k_m.gguf` from HuggingFace

2. **Richer behavioral diversity**
   - More distinct fwd/back signal differentiation
   - Additional movement patterns from neuron type combinations

### AION Alpha
3. **Reaction-diffusion layer**
   - Add chemical concentration grid over connectome
   - Dynamic threshold modulation per neuron
   - Same input → different response based on internal state
   - numpy-only, no GPU required

### AION Beta
4. **Evolutionary body generation**
   - GA-based MuJoCo XML generation
   - Body shape adapts to brain firing patterns

### AION Release
5. **Hebbian learning**
   - `ΔW = η * pre * post`
   - Experience accumulates → behavior changes over time
   - Requires state persistence (save/load W matrix)

---

## 10. Development Notes

- **Never modify** `chimera_load_connectome.py` weight normalization (p99) — changing this breaks firing
- **Qwen as input parser only** — using it for output at 0.5B causes system prompt leakage
- **KMAP fallback** — always keep updated; Qwen fails on uncommon inputs
- **macOS**: requires `mjpython` (bundled with `pip install mujoco`); auto-detected via env var `CHIMERA_MJPYTHON`
- **Windows build**: `python -m PyInstaller chimera.spec` (not `pyinstaller` directly — PATH issue)
- **build.bat**: ASCII-only, no Korean characters (Windows encoding issue)
- **chimera.spec**: do not overwrite during builds

---

## 11. Environment

| Item | Version |
|------|---------|
| Python | 3.10+ |
| MuJoCo | 3.5+ |
| numpy | 1.24+ |
| llama-cpp-python | 0.2+ |
| LLM | Qwen2.5-0.5B-Instruct Q4_K_M |
| Platform | Windows 10+ / macOS (Apple Silicon + Intel) |

---

## 12. Repo

```
https://github.com/caparison1234/chimera
```

**Install from scratch:**
```bash
curl -O https://raw.githubusercontent.com/caparison1234/chimera/main/setup.py
python3 setup.py      # downloads all files + connectome + Qwen
python3 chimera_app.py
```

**Reference paper:**
> Winding, M. et al. (2023). The connectome of an insect brain. *Science*, 379(6636).
> https://doi.org/10.1126/science.add9330
