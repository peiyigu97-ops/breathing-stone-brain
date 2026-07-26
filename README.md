# Acknowledgments

This project would not have been possible without the following works, datasets, libraries, and researchers. Every external contribution is listed here in full.

---

## Connectome Data

**Winding, M., Pedigo, B. D., Barnes, C. L., Patsolic, H. G., Park, Y., Krieger, T., Jutisz, A., Wu, Y., Berck, M. E., Lin, Y., Venkatachalam, K., Cardona, A., Costa, M., Bhatt, D. L., Bhatt, D., Bhatt, B., Bhatnagar, M., Bhatt, P., … Zlatic, M. (2023).** The connectome of an insect brain. *Science*, 379(6636). https://doi.org/10.1126/science.add9330

The CHIMERA simulation uses the *Drosophila* larva full brain connectome from Winding et al. (2023), specifically the adjacency matrix and neuron type annotations from `Supplementary-Data-S1.zip`. The first 1,373 × 1,373 neuron submatrix and the corresponding neuron metadata are used. Data is distributed under **CC BY 4.0**.

Data repository: https://github.com/brain-networks/larval-drosophila-connectome

---

## Physics Simulation

**Todorov, E., Erez, T., & Tassa, Y. (2012).** MuJoCo: A physics engine for model-based control. *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. https://doi.org/10.1109/IROS.2012.6386109

MuJoCo is used to simulate the 12-actuator *Drosophila*-inspired body in `chimera_app.py`. Licensed under **Apache 2.0**. https://mujoco.org

---

## Language Model Inference

**Qwen Team, Alibaba Cloud (2024).** Qwen2.5-0.5B-Instruct (Q4_K_M GGUF quantization). https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF

Used exclusively as a natural-language input parser (text → sensory channel JSON). Output generation is intentionally disabled. Licensed under the **Qwen License** (non-commercial research use).

**Gerganov, G. et al. llama.cpp.** https://github.com/ggerganov/llama.cpp — MIT License.
The underlying C++ LLM inference runtime used by `llama-cpp-python`.

**Abetlen, A. et al. llama-cpp-python.** https://github.com/abetlen/llama-cpp-python — MIT License.
Python bindings for llama.cpp used to load and run the Qwen model.

---

## 3D Visualisation

**mrdoob et al. Three.js.** https://threejs.org — MIT License.
The entire 3D galaxy-style brain viewer (`HumanBrainDT/viewer/static/index.html`) is built on Three.js, including WebGLRenderer, PointsMaterial, LineSegments, BufferGeometry, QuadraticBezierCurve3, and CubicBezierCurve3.

**Three.js OrbitControls addon.** https://github.com/mrdoob/three.js/blob/dev/examples/jsm/controls/OrbitControls.js — MIT License.
Used for camera orbit, zoom, and pan in the 3D viewer.

---

## Web Server & API

**Ramírez, S. et al. FastAPI.** https://fastapi.tiangolo.com — MIT License.
Powers the WebSocket server, REST endpoints (`/api/csv`, `/api/log`, `/api/seed`), and static file serving in `HumanBrainDT/viewer/server.py`.

**Langa, T. et al. Uvicorn.** https://www.uvicorn.org — BSD License.
ASGI server used to run the FastAPI application.

---

## Scientific Libraries

**Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).** Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2 — BSD License.
NumPy is used throughout for LIF simulation matrices, weight normalization, spike counting, and all numerical computation.

---

## Neuroscience Frameworks (HumanBrainDT)

The simulation model, pathway weights, and ANS/anxiety decomposition in `HumanBrainDT/core/lif_engine.py` are derived from the following peer-reviewed literature. No code was copied from any of these sources; they were used to determine parameter values and architectural decisions.

**Autonomic Nervous System / Polyvagal Theory**

- **Porges, S. W. (1995).** Orienting in a defensive world: Mammalian modifications of our evolutionary heritage. A polyvagal theory. *Psychophysiology*, 32(4), 301–318. https://doi.org/10.1111/j.1469-8986.1995.tb01213.x
- **Porges, S. W. (2022).** Polyvagal Theory: A science of safety. *Frontiers in Integrative Neuroscience*, 16. https://doi.org/10.3389/fnint.2022.871227

*Basis for sympathetic/parasympathetic balance, HRV as vagal tone proxy, and the three polyvagal states (ventral vagal, sympathetic, dorsal vagal) used in `stone_bridge.py`.*

**Deep Pressure / CT Afferents / Oxytocin**

- **McGlone, F., Wessberg, J., & Olausson, H. (2014).** Discriminative and affective touch: Sensing and feeling. *Neuron*, 82(4), 737–755. https://doi.org/10.1016/j.neuron.2014.05.001
- **Uvnäs-Moberg, K., Handlin, L., & Petersson, M. (2014).** Self-soothing behaviors with particular reference to oxytocin release induced by non-noxious sensory stimulation. *Frontiers in Psychology*, 5. https://doi.org/10.3389/fpsyg.2014.01529

*Basis for `touch_deep` healing weight (0.85): deep pressure → CT afferents → oxytocin → HPA axis inhibition → parasympathetic upregulation.*

**Temperature / Insula / Interoception**

- **Ijzerman, H., & Semin, G. R. (2009).** The thermometer of social relations: Mapping social proximity on temperature. *Psychological Science*, 20(10), 1214–1220. https://doi.org/10.1111/j.1467-9280.2009.02434.x
- **Craig, A. D. (2002).** How do you feel? Interoception: the sense of the physiological condition of the body. *Nature Reviews Neuroscience*, 3(2), 655–666. https://doi.org/10.1038/nrn894

*Basis for `thermal_warm` healing weight (0.75): warmth → insula → hypothalamus → parasympathetic activation.*

**Interoception / Predictive Coding**

- **Seth, A. K. (2013).** Interoceptive inference, emotion, and the embodied self. *Trends in Cognitive Sciences*, 17(11), 565–573. https://doi.org/10.1016/j.tics.2013.09.007
- **Price, C. J., & Hooven, C. (2018).** Interoceptive awareness skills for emotion regulation: Theory and approach of mindful awareness in body-oriented therapy (MABT). *Frontiers in Psychology*, 9. https://doi.org/10.3389/fpsyg.2018.00798

*Basis for `interoception` routing to insula → cingulate → hypothalamus, and the regulation pathway.*

**Rhythmic Breathing / Vagal Tone**

- **Kleitman, N. (1963).** *Sleep and Wakefulness* (2nd ed.). University of Chicago Press.

*Basis for `rhythmic` healing weight (0.70): rhythmic input → cerebellum → thalamus → vagal tone upregulation.*

**Allostatic Load / HPA Axis**

- **McEwen, B. S. (1998).** Stress, adaptation, and disease: Allostasis and allostatic load. *Annals of the New York Academy of Sciences*, 840(1), 33–44. https://doi.org/10.1111/j.1749-6632.1998.tb09546.x

*Basis for `anxiety.baseline` (allostatic overload) and `stress_hormone` channel → HPA axis modelling.*

**Amygdala / Threat Circuit**

- **LeDoux, J. E. (1996).** *The Emotional Brain: The Mysterious Underpinnings of Emotional Life*. Simon & Schuster.

*Basis for `threat_input` → amygdala → brainstem routing (healing weight −1.00), and the amygdalofugal pathway.*

**Prefrontal–Amygdala Regulation / Anxiety**

- **Etkin, A., Büchel, C., & Gross, J. J. (2015).** The neural bases of emotion regulation. *Nature Reviews Neuroscience*, 16(11), 693–700. https://doi.org/10.1038/nrn4044

*Basis for `anxiety.anticipatory` (prefrontal–ACC–amygdala worry loop) and `anxiety.regulation` (PFC top-down control of amygdala).*

**Somatic Anxiety / Insula**

- **Paulus, M. P., & Stein, M. B. (2006).** An insular view of anxiety. *Biological Psychiatry*, 60(4), 383–387. https://doi.org/10.1016/j.biopsych.2006.03.042

*Basis for `anxiety.somatic` calculation using insula spike count, and `muscle_tension` / `nausea_input` routing to insula.*

**Deep Pressure / Vibration**

- **Krauss, J. K., et al. (1987).** Vibration-induced cortical inhibition and thalamic gating. Referenced in context of cerebellum → thalamic gating for vibration input.

*Basis for `touch_vibration` healing weight (0.60) and cerebellar routing.*

**Cold Face Reflex**

- **Cold Face Test.** (Multiple PubMed sources, 2022.) Cold water application to face → brainstem → amygdala → sympathetic arousal.

*Basis for `thermal_cold` healing weight (−0.30): cold → brainstem/amygdala → sympathetic upregulation.*

---

## MNI Brain Atlas

The spatial coordinates of all 16 brain regions in `HumanBrainDT/regions/builder.py` and the 3D viewer are derived from standard MNI (Montreal Neurological Institute) space centroids, using publicly available neuroimaging literature and atlas references:

- **Collins, D. L., Neelin, P., Peters, T. M., & Evans, A. C. (1994).** Automatic 3D intersubject registration of MR volumetric data in standardized Talairach space. *Journal of Computer Assisted Tomography*, 18(2), 192–205.
- **Mazziotta, J., Toga, A., Evans, A., et al. (2001).** A probabilistic atlas and reference system for the human brain. *Philosophical Transactions of the Royal Society B*, 356(1412), 1293–1322. https://doi.org/10.1098/rstb.2001.0915

---

## Hardware Platforms

**NeuroSky ThinkGear ASIC Module (TGAM).** The `stone_session.csv` output format in `tgam_csv_generator.py` follows TGAM EEG band conventions (delta, theta, alpha, beta, gamma). https://neurosky.com

**Espressif ESP32.** The Breathing Stone device uses an ESP32 microcontroller for skin temperature sensing, grip force measurement, and haptic feedback. https://www.espressif.com

**NVIDIA Jetson Nano.** Deployment configuration in `jetson_setup.sh` targets the Jetson Nano for edge inference. https://developer.nvidia.com/embedded/jetson-nano

---

## Software Tools

**PyInstaller.** https://pyinstaller.org — GPL-2.0-or-later.
Used to build the Windows `.exe` distribution (`chimera.spec`, `build.bat`).

**Tunnelmole.** https://tunnelmole.com — MIT License.
Used for public URL tunnelling during demonstrations (`tmole.js`).

---

## License Summary

| Component | License |
|-----------|---------|
| Connectome data (Winding et al. 2023) | CC BY 4.0 |
| Qwen2.5-0.5B model weights | Qwen License (non-commercial) |
| MuJoCo | Apache 2.0 |
| Three.js | MIT |
| Three.js OrbitControls | MIT |
| llama.cpp | MIT |
| llama-cpp-python | MIT |
| FastAPI | MIT |
| Uvicorn | BSD |
| NumPy | BSD |
| PyInstaller | GPL-2.0-or-later |
| Tunnelmole | MIT |
| All original code in this repository | MIT |

---

## Statement of Originality

All simulation code, architectural decisions, and software implementations in this repository are original work. External libraries are used only through their published APIs. Neuroscience literature was used solely to derive parameter values and model architecture; no text, code, or data from those papers was incorporated directly.
