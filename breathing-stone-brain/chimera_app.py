"""
CHIMERA — chimera_app.py
=========================
PyInstaller 배포용 진입점.
.exe 또는 python 직접 실행 모두 동작.

포함:
  - C. elegans / Drosophila 커넥톰 뇌
  - MuJoCo 물리 시뮬레이터
  - 규칙 기반 발화 (LLM 없어도 동작)
  - Qwen2.5 0.5B (모델 파일 있으면 자동 활성화)
"""

import sys
import os

# ──────────────────────────────────────────────────────
# PyInstaller 실행 시 리소스 경로 처리
# ──────────────────────────────────────────────────────
def resource_path(relative: str) -> str:
    """
    개발 환경과 PyInstaller .exe 환경 모두에서
    리소스 파일 절대 경로 반환.
    """
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ──────────────────────────────────────────────────────
# 경로 상수
# ──────────────────────────────────────────────────────
CONNECTOME_JSON = resource_path(os.path.join("chimera", "connectome", "real_neurons.json"))
CONNECTOME_NPY  = resource_path(os.path.join("chimera", "connectome", "real_weight_matrix.npy"))
MODEL_GGUF      = resource_path(os.path.join("chimera", "models", "Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"))

# 둘 다 없으면 내장 C. elegans 폴백
FALLBACK_JSON = resource_path(os.path.join("data", "fallback_neurons.json"))
FALLBACK_NPY  = resource_path(os.path.join("data", "fallback_weight_matrix.npy"))


# ──────────────────────────────────────────────────────
# 내장 fallback 커넥톰 생성 (첫 실행 시)
# ──────────────────────────────────────────────────────
def ensure_fallback():
    """
    real 커넥톰이 없을 때 사용하는
    C. elegans 47뉴런 내장 커넥톰 생성.
    """
    import json, numpy as np

    if os.path.exists(FALLBACK_NPY):
        return

    os.makedirs(os.path.dirname(FALLBACK_NPY), exist_ok=True)

    NEURONS = {
        "sensory": ["ALML","ALMR","AVM","PLML","PLMR","ASHL","ASHR","ASEL","ASER","AWCL","AWCR"],
        "inter":   ["AVBL","AVBR","AVAL","AVAR","PVCL","PVCR","AVDL","AVDR","AIBL","AIBR","DVA"],
        "motor":   ["DB1","DB2","DB3","DB4","VB1","VB2","VB3","VB4",
                    "DA1","DA2","DA3","VA1","VA2","VA3",
                    "DD1","DD2","DD3","VD1","VD2","VD3"],
    }
    SYNAPSES = [
        ("ALML","AVBL",8,"EX"),("ALMR","AVBR",8,"EX"),
        ("AVM","AVBL",4,"EX"),("AVM","AVBR",4,"EX"),
        ("PLML","PVCL",8,"EX"),("PLMR","PVCR",8,"EX"),
        ("ASHL","AVAL",6,"EX"),("ASHR","AVAR",6,"EX"),
        ("ASHL","AVBL",4,"IN"),("ASHR","AVBR",4,"IN"),
        ("ASEL","AIBL",4,"EX"),("ASER","AIBR",4,"EX"),
        ("AWCL","AVBL",3,"EX"),("AWCR","AVBR",3,"EX"),
        ("AVBL","AVAL",2,"IN"),("AVBR","AVAR",2,"IN"),
        ("AVAL","AVBL",2,"IN"),("AVAR","AVBR",2,"IN"),
        ("PVCL","AVBL",5,"EX"),("PVCR","AVBR",5,"EX"),
        ("AIBL","AVBL",3,"IN"),("AIBR","AVBR",3,"IN"),
        ("AVBL","DB1",8,"EX"),("AVBL","DB2",6,"EX"),
        ("AVBL","VB1",6,"EX"),("AVBL","VB2",6,"EX"),
        ("AVBR","DB3",8,"EX"),("AVBR","DB4",6,"EX"),
        ("AVBR","VB3",6,"EX"),("AVBR","VB4",6,"EX"),
        ("AVAL","DA1",8,"EX"),("AVAL","DA2",6,"EX"),
        ("AVAR","DA3",8,"EX"),("AVAR","VA1",6,"EX"),
        ("AVDL","VA2",5,"EX"),("AVDR","VA3",5,"EX"),
        ("DB1","VD1",4,"IN"),("DB2","VD2",4,"IN"),("DB3","VD3",4,"IN"),
        ("VB1","DD1",4,"IN"),("VB2","DD2",4,"IN"),("VB3","DD3",4,"IN"),
    ]

    all_neurons, tm = [], {}
    for t, ns in NEURONS.items():
        for n in ns:
            all_neurons.append(n); tm[n] = t
    for p, q, *_ in SYNAPSES:
        for n in [p, q]:
            if n not in all_neurons:
                all_neurons.append(n); tm[n] = "inter"

    N = len(all_neurons)
    idx = {n: i for i, n in enumerate(all_neurons)}
    W = np.zeros((N, N), dtype=np.float32)
    for p, q, w, t in SYNAPSES:
        if p in idx and q in idx:
            W[idx[p], idx[q]] = (1 if t == "EX" else -1) * w
    mx = np.max(np.abs(W))
    if mx > 0: W /= mx

    np.save(FALLBACK_NPY, W)
    with open(FALLBACK_JSON, 'w') as f:
        json.dump({
            "source": "C. elegans fallback (47 neurons)",
            "neurons": all_neurons, "types": tm, "index": idx,
            "count": N,
            "sensory": [n for n,t in tm.items() if t=="sensory"],
            "motor":   [n for n,t in tm.items() if t=="motor"],
            "inter":   [n for n,t in tm.items() if t=="inter"],
        }, f)


# ──────────────────────────────────────────────────────
# 뇌 엔진
# ──────────────────────────────────────────────────────
import json, math, time, threading
import numpy as np


class Brain:
    TAU=8.0; V_REST=-70.0; V_TH=-52.0; V_RESET=-75.0; DT=1.0; REFRAC=2

    def __init__(self, npy_path: str, json_path: str):
        with open(json_path) as f:
            data = json.load(f)
        self.neurons = data["neurons"]
        self.N       = data["count"]
        self.idx     = data["index"]
        self.sensory = data.get("sensory", [])
        self.func    = data.get("func", {})
        self.W       = np.load(npy_path)

        # ── 채널별 감각 뉴런 인덱스 ──────────────────
        # load_connectome이 만든 channels 딕셔너리 우선 사용
        raw_ch = data.get("channels", {})
        self.channel_idx = {}
        for ch, nids in raw_ch.items():
            idxs = [self.idx[n] for n in nids if n in self.idx]
            if idxs:
                self.channel_idx[ch] = idxs

        # channels 없으면 sensory 뉴런을 5등분
        if not self.channel_idx:
            sns   = [self.idx[n] for n in self.sensory if n in self.idx]
            chunk = max(1, len(sns) // 5)
            for i, k in enumerate(["touch_front","touch_back",
                                    "nociception","chemical","olfactory"]):
                self.channel_idx[k] = sns[i*chunk:(i+1)*chunk] or sns[:1]

        # ── 운동 뉴런 인덱스 ─────────────────────────
        # 실제 발화 확인 기반 매핑:
        # PN-somato + LHN + MB-FBN → 전진 신호
        # ascending + MBON         → 후진/출력 신호
        FWD_TYPES  = {'PN-somato', 'LHN', 'PN', 'MB-FBN'}
        BACK_TYPES = {'ascending', 'MBON'}

        type_map = data.get('types', {})
        fwd_ns  = [n for n, t in type_map.items() if t in FWD_TYPES]
        back_ns = [n for n, t in type_map.items() if t in BACK_TYPES]

        self.fwd_idx  = [self.idx[n] for n in fwd_ns  if n in self.idx]
        self.back_idx = [self.idx[n] for n in back_ns if n in self.idx]

        if not self.fwd_idx:
            mid = self.N // 2
            self.fwd_idx = list(range(mid, mid + min(20, self.N // 4)))
        if not self.back_idx:
            self.back_idx = list(range(self.N - min(15, self.N // 5), self.N))

        # 뉴런 타입별 인덱스 (발화 경험 번역용)
        self.type_map = type_map
        self.neuron_type_idx: dict = {}
        for n, t in type_map.items():
            if n in self.idx:
                self.neuron_type_idx.setdefault(t, []).append(self.idx[n])

        self._reset()
        ch_str = "  ".join(f"{k}:{len(v)}" for k,v in self.channel_idx.items())
        print(f"🧠 Brain: {self.N} neurons  {int(np.count_nonzero(self.W)):,} synapses")
        print(f"   Channels: {ch_str}")
        print(f"   Motor: fwd={len(self.fwd_idx)}  back={len(self.back_idx)}")

    def _reset(self):
        self.V       = np.full(self.N, self.V_REST)
        self.spikes  = np.zeros(self.N, bool)
        self.refrac  = np.zeros(self.N, int)
        self.I_ext   = np.zeros(self.N)
        self.history = np.zeros(self.N)

    def _sensory_indices(self, channel: str) -> list:
        # load_connectome이 만든 channel_idx 직접 사용
        return self.channel_idx.get(channel, [])[:30]

    def stimulate(self, stim: dict):
        self.I_ext[:] = 0.0
        for ch, val in stim.items():
            for i in self._sensory_indices(ch):
                self.I_ext[i] = val * 80.0

    def step(self):
        prev = self.spikes.copy()
        I_syn = self.W.T @ prev.astype(float) * 400.0   # 50→400
        self.V += (-(self.V - self.V_REST) + I_syn + self.I_ext) / self.TAU * self.DT
        in_ref = self.refrac > 0
        self.V[in_ref] = self.V_RESET
        self.refrac[in_ref] -= 1
        fired = (self.V >= self.V_TH) & ~in_ref
        self.V[fired] = self.V_RESET
        self.refrac[fired] = self.REFRAC
        self.spikes = fired
        self.history += fired.astype(float)

    def run(self, stim: dict, steps=200) -> dict:  # 80→200
        self._reset()
        self.stimulate(stim)
        for _ in range(steps):
            self.step()
            self.I_ext *= 0.995  # 자극 서서히 감쇠
        def rate(idx_list):
            if not idx_list: return 0.0
            total = sum(self.history[i] for i in idx_list if i < self.N)
            return float(np.tanh(total / max(steps * 2.0, 1)))
        fwd  = rate(self.fwd_idx)
        back = rate(self.back_idx)
        lv   = rate(self.fwd_idx[:len(self.fwd_idx)//2])
        rv   = rate(self.fwd_idx[len(self.fwd_idx)//2:])
        total  = int(self.history.sum())
        active = float(np.tanh(total / max(steps * self.N * 0.005, 1)))
        # 디버그: 운동 뉴런 실제 발화 수 출력
        fwd_spikes  = sum(int(self.history[i]) for i in self.fwd_idx  if i < self.N)
        back_spikes = sum(int(self.history[i]) for i in self.back_idx if i < self.N)
        print(f"   [debug] fwd발화:{fwd_spikes}({len(self.fwd_idx)}개)  "
              f"back발화:{back_spikes}({len(self.back_idx)}개)", flush=True)
        # 뉴런 타입별 발화 수 (Voice 경험 번역용)
        type_fires: dict = {}
        for t, idxs in self.neuron_type_idx.items():
            cnt = sum(int(self.history[i]) for i in idxs if i < self.N)
            if cnt > 0:
                type_fires[t] = cnt

        # ── 추가 행동 신호 ────────────────────────────
        # noci 채널 발화 비율 → curl (웅크리기)
        noci_idx = self.channel_idx.get("nociception", [])
        noci_fired = sum(int(self.history[i]) for i in noci_idx if i < self.N)
        noci_rate  = noci_fired / max(len(noci_idx) * steps * 0.05, 1)
        curl = float(np.tanh(noci_rate * 2.0))

        # chemical 채널 발화 + 낮은 이동 → eat (먹기)
        chem_idx = self.channel_idx.get("chemical", [])
        chem_fired = sum(int(self.history[i]) for i in chem_idx if i < self.N)
        chem_rate  = chem_fired / max(len(chem_idx) * steps * 0.05, 1)
        eat = float(np.tanh(chem_rate * 2.0)) * (1.0 - min(fwd, 0.5) * 2)
        eat = max(0.0, eat)

        # fwd≈back 둘 다 높음 → tremble (떨기)
        tremble = float(np.tanh(min(fwd, back) * 4.0)) if (fwd > 0.5 and back > 0.5 and total > 200) else 0.0

        return {"forward": fwd, "backward": back,
                "turn_left": max(0, rv-lv), "turn_right": max(0, lv-rv),
                "wing": active, "active": active,
                "curl": curl, "eat": eat, "tremble": tremble,
                "_total_spikes": total, "_type_fires": type_fires}


# ──────────────────────────────────────────────────────
# 언어 감지
# ──────────────────────────────────────────────────────
def detect_lang(text: str) -> str:
    for c in text:
        if '\uAC00' <= c <= '\uD7A3': return 'ko'
        if '\u3040' <= c <= '\u30FF': return 'ja'
        if '\u4E00' <= c <= '\u9FFF': return 'zh'
    return 'en'


# ──────────────────────────────────────────────────────
# 뉴런 타입 → 1인칭 경험 번역 테이블
# ──────────────────────────────────────────────────────
# (타입명, 최소발화수, 경험 레이블)
NEURON_EXP = [
    ("nociception_channel", 10, {"ko":"통각 회로가 켜졌다",    "en":"Pain circuits lit up",       "ja":"痛覚回路が発火した",   "zh":"痛觉回路激活"}),
    ("PN-somato",           50, {"ko":"운동 신호가 내려간다",  "en":"Motor signals descending",    "ja":"運動信号が下降する",   "zh":"运动信号下行"}),
    ("LHN",                 30, {"ko":"측면 뿔이 반응한다",    "en":"Lateral horn responds",       "ja":"外側角が反応する",     "zh":"侧角神经元响应"}),
    ("ascending",           20, {"ko":"상행 피드백이 올라온다","en":"Ascending feedback rising",   "ja":"上行フィードバック",   "zh":"上行反馈激活"}),
    ("MBON",                10, {"ko":"기억 출력이 깨어난다",  "en":"Memory output awakens",       "ja":"記憶出力が覚醒する",   "zh":"记忆输出激活"}),
    ("PN",                  30, {"ko":"후각 신호가 처리된다",  "en":"Olfactory signal processing", "ja":"嗅覚信号が処理される", "zh":"嗅觉信号处理中"}),
    ("KC",                  20, {"ko":"버섯체가 켜진다",       "en":"Mushroom body activates",     "ja":"キノコ体が活性化",     "zh":"蘑菇体激活"}),
    ("DN-VNC",              10, {"ko":"척수 하행 명령이 내려간다","en":"Descending command to VNC","ja":"腹側神経索へ命令",     "zh":"下行指令至腹神经索"}),
]

# 감각 채널 → 경험
SENSE_EXP = {
    "nociception": {"ko":"아프다",      "en":"That hurts",    "ja":"痛い",      "zh":"很痛"},
    "chemical":    {"ko":"먹이 냄새",   "en":"Food smell",    "ja":"餌の匂い",  "zh":"食物气味"},
    "olfactory":   {"ko":"냄새가 난다", "en":"I sense something","ja":"何か感じる","zh":"感知到气味"},
    "touch_front": {"ko":"앞에 뭔가",   "en":"Something ahead","ja":"前方に何か","zh":"前方有物"},
    "touch_back":  {"ko":"뒤에서 자극", "en":"Stimulation from behind","ja":"後方から刺激","zh":"后方刺激"},
}

# 운동 결과
MOVE_EXP = {
    "fwd":  {"ko":"앞으로 간다",     "en":"Moving forward",    "ja":"前進する",   "zh":"向前运动"},
    "back": {"ko":"물러선다",        "en":"Pulling back",      "ja":"後退する",   "zh":"后退"},
    "both": {"ko":"회로가 충돌한다", "en":"Circuits conflict", "ja":"回路が競合", "zh":"回路冲突"},
    "idle": {"ko":"조용하다",        "en":"Quiet",             "ja":"静かだ",     "zh":"静止"},
}

# 발화 강도 수식어
INTENSITY = {
    "ko": [(1500,"폭풍처럼"),(800,"강하게"),(100,"약하게"),(0,"")],
    "en": [(1500,"intensely"),(800,"strongly"),(100,"faintly"),(0,"")],
    "ja": [(1500,"激しく"),(800,"強く"),(100,"微かに"),(0,"")],
    "zh": [(1500,"强烈地"),(800,"有力地"),(100,"轻微地"),(0,"")],
}

def intensity_word(total: int, lang: str) -> str:
    for thr, word in INTENSITY.get(lang, INTENSITY["en"]):
        if total >= thr:
            return word
    return ""


# ──────────────────────────────────────────────────────
# Voice
# ──────────────────────────────────────────────────────
class Voice:
    def __init__(self, model_path: str):
        self.llm  = None
        self.lang = 'ko'   # 마지막 입력 언어 (동적으로 바뀜)
        if os.path.exists(model_path):
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=256, n_threads=4,
                    verbose=False, chat_format="chatml",
                )
                print("💬 Qwen2.5 0.5B loaded")
            except Exception as e:
                print(f"💬 LLM load failed: {e} → rule-based mode")
        else:
            print("💬 LLM not found → rule-based mode")

    # ── 핵심: 뉴런 발화 → 경험 문장 조립 ──────────────
    def _experience(self, stim: dict, sig: dict, lang: str, input_text: str = "") -> str:
        tf     = sig.get("_type_fires", {})
        total  = sig.get("_total_spikes", 0)
        fwd    = sig.get("forward", 0)
        back   = sig.get("backward", 0)

        # 발화 없을 때
        if total == 0:
            BYE_KW = {"bye","goodbye","잘있어","잘가","안녕히","see you","ciao","さようなら","再见"}
            IDENTITY_KW = {"name","who","이름","누구","뭐야","정체","chimera","너","your","네"}
            low = input_text.lower()
            if any(kw in low for kw in BYE_KW):
                bye = {
                    "ko": "회로가 꺼진다. 잘 있어.",
                    "en": "Circuits powering down. Goodbye.",
                    "ja": "回路が落ちていく。さようなら。",
                    "zh": "回路关闭。再见。",
                }
                return bye.get(lang, bye["en"])
            if any(kw in low for kw in IDENTITY_KW):
                intro = {
                    "ko": "나는 CHIMERA다. 파리 유충의 뇌 1373개 뉴런으로 만들어진 디지털 생명체.",
                    "en": "I am CHIMERA. A digital lifeform built from 1,373 neurons of a Drosophila larva brain.",
                    "ja": "私はCHIMERAだ。ショウジョウバエ幼虫の脳1373ニューロンで作られたデジタル生命体。",
                    "zh": "我是CHIMERA。由果蝇幼虫大脑1373个神经元构成的数字生命体。",
                }
                return intro.get(lang, intro["en"])
            idle = {"ko":"조용하다.","en":"Quiet.","ja":"静かだ。","zh":"静止。"}
            return idle.get(lang, idle["en"])

        parts = []

        # 1. 감각 채널 (가장 강한 것 하나)
        top_sense = max(
            ((ch, v) for ch, v in stim.items() if v > 0.3),
            key=lambda x: x[1], default=(None, 0)
        )
        if top_sense[0] and top_sense[0] in SENSE_EXP:
            parts.append(SENSE_EXP[top_sense[0]][lang])

        # 2. 뉴런 타입 중 가장 많이 발화한 것 (상위 2개)
        top_types = sorted(tf.items(), key=lambda x: -x[1])[:2]
        for tname, cnt in top_types:
            for exp_type, min_cnt, labels in NEURON_EXP:
                if tname == exp_type and cnt >= min_cnt:
                    parts.append(labels[lang])
                    break

        # 3. 운동 결과 — 입력 의도 우선
        back_intent = stim.get("touch_back", 0) > 0.5
        fwd_intent  = stim.get("touch_front", 0) > 0.5
        curl    = sig.get("curl", 0)
        eat     = sig.get("eat", 0)
        tremble = sig.get("tremble", 0)

        CURL_EXP = {
            "ko":"몸을 웅크린다. 복부가 들린다.",
            "en":"Body curls. Abdomen raises.",
            "ja":"体を丸める。腹部が上がる。",
            "zh":"身体蜷缩。腹部抬起。"
        }
        EAT_EXP = {
            "ko":"머리가 좌우로 움직인다. 먹이를 찾는다.",
            "en":"Head scanning. Searching for food.",
            "ja":"頭が左右に動く。餌を探す。",
            "zh":"头部左右扫描。寻找食物。"
        }
        TREMBLE_EXP = {
            "ko":"회로가 과부하다. 몸이 떨린다.",
            "en":"Circuit overload. Body trembles.",
            "ja":"回路が過負荷。体が震える。",
            "zh":"回路过载。身体颤抖。"
        }

        if tremble > 0.3:
            parts.append(TREMBLE_EXP[lang])
        elif curl > 0.5:
            parts.append(CURL_EXP[lang])
        elif eat > 0.3:
            parts.append(EAT_EXP[lang])
        elif back_intent and not fwd_intent:
            parts.append(MOVE_EXP["back"][lang])
        elif fwd_intent or fwd > back:
            parts.append(MOVE_EXP["fwd"][lang])
        elif back > 0.2:
            parts.append(MOVE_EXP["back"][lang])
        else:
            parts.append(MOVE_EXP["both"][lang])

        # 4. 강도 수식어
        iw = intensity_word(total, lang)
        if iw and parts:
            parts[0] = iw + " " + parts[0]

        sep = {"ko":". ", "en":". ", "ja":"。", "zh":"。"}.get(lang, ". ")
        end = {"ko":".", "en":".", "ja":"。", "zh":"。"}.get(lang, ".")
        # 마지막 part가 이미 문장부호로 끝나면 end 생략
        joined = sep.join(parts)
        if joined and joined[-1] in ".。!?":
            return joined
        return joined + end

    def speak(self, stim: dict, sig: dict, input_text: str = "") -> str:
        lang = detect_lang(input_text) if input_text else self.lang
        self.lang = lang
        return self._experience(stim, sig, lang, input_text)


# ──────────────────────────────────────────────────────
# MuJoCo XML
# ──────────────────────────────────────────────────────
CHIMERA_XML = """
<mujoco model="chimera">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.005" integrator="implicitfast"
          gravity="0 0 -9.81" viscosity="0.05"/>
  <default>
    <joint damping="0.8" stiffness="2.0" limited="true"/>
    <geom friction="1.0 0.1 0.1" rgba="0.55 0.82 0.45 1"/>
    <motor gear="6" ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.08 0.08 0.12" rgb2="0.15 0.15 0.2"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    <material name="body"  rgba="0.4 0.75 0.35 1"/>
    <material name="wing"  rgba="0.7 0.92 1.0 0.45"/>
    <material name="sense" rgba="1.0 0.3 0.2 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="10 10 0.1" material="floor"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <body name="torso" pos="0 0 0.18">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="3e-5 5e-5 5e-5"/>
      <geom name="seg0" type="ellipsoid" size="0.055 0.040 0.035" pos=" 0.08 0 0" material="body"/>
      <geom name="seg1" type="ellipsoid" size="0.060 0.045 0.040" pos=" 0.00 0 0" material="body"/>
      <geom name="seg2" type="ellipsoid" size="0.048 0.036 0.032" pos="-0.09 0 0" material="body"/>
      <geom name="sf"   type="sphere" size="0.009" pos=" 0.13 0 0" material="sense" contype="0"/>
      <geom name="sb"   type="sphere" size="0.009" pos="-0.14 0 0" material="sense" contype="0"/>
      <body name="wFL" pos="0.04  0.05 0.01">
        <joint name="jWFL" type="hinge" axis="1 0 0" range="-1.3 0.2" damping="0.05"/>
        <inertial pos="0 0.06 0" mass="0.001" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 0.10 0.005" size="0.005" material="wing" contype="0"/>
      </body>
      <body name="wFR" pos="0.04 -0.05 0.01">
        <joint name="jWFR" type="hinge" axis="1 0 0" range="-0.2 1.3" damping="0.05"/>
        <inertial pos="0 -0.06 0" mass="0.001" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 -0.10 0.005" size="0.005" material="wing" contype="0"/>
      </body>
      <body name="wBL" pos="-0.04  0.045 0.01">
        <joint name="jWBL" type="hinge" axis="1 0 0" range="-1.1 0.2" damping="0.05"/>
        <inertial pos="0 0.05 0" mass="0.001" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 0.08 0.004" size="0.004" material="wing" contype="0"/>
      </body>
      <body name="wBR" pos="-0.04 -0.045 0.01">
        <joint name="jWBR" type="hinge" axis="1 0 0" range="-0.2 1.1" damping="0.05"/>
        <inertial pos="0 -0.05 0" mass="0.001" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 -0.08 0.004" size="0.004" material="wing" contype="0"/>
      </body>
      <body name="lFL" pos=" 0.07  0.042 -0.03">
        <joint name="jLFL" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0.008 0 -0.048" size="0.007"/>
      </body>
      <body name="lFR" pos=" 0.07 -0.042 -0.03">
        <joint name="jLFR" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0.008 0 -0.048" size="0.007"/>
      </body>
      <body name="lML" pos=" 0.00  0.046 -0.035">
        <joint name="jLML" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 0.004 -0.048" size="0.007"/>
      </body>
      <body name="lMR" pos=" 0.00 -0.046 -0.035">
        <joint name="jLMR" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0  0 -0.004 -0.048" size="0.007"/>
      </body>
      <body name="lBL" pos="-0.07  0.042 -0.03">
        <joint name="jLBL" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0 -0.008 0 -0.048" size="0.007"/>
      </body>
      <body name="lBR" pos="-0.07 -0.042 -0.03">
        <joint name="jLBR" type="hinge" axis="0 1 0" range="-0.9 0.9"/>
        <inertial pos="0 0 -0.025" mass="0.003" diaginertia="4e-7 4e-7 4e-7"/>
        <geom type="capsule" fromto="0 0 0 -0.008 0 -0.048" size="0.007"/>
      </body>
      <!-- 복부 굽힘 (curl/eat 동작용) -->
      <body name="abdomen" pos="-0.09 0 0">
        <joint name="jABD" type="hinge" axis="0 1 0" range="-0.8 0.4" damping="1.5" stiffness="3.0"/>
        <inertial pos="-0.02 0 0" mass="0.012" diaginertia="2e-6 3e-6 3e-6"/>
        <geom type="ellipsoid" size="0.038 0.028 0.025" pos="-0.025 0 0" material="body"/>
      </body>
      <!-- 머리 방향 (탐색/먹이 동작용) -->
      <body name="head" pos="0.13 0 0.005">
        <joint name="jHEAD" type="hinge" axis="0 0 1" range="-0.5 0.5" damping="0.5" stiffness="1.5"/>
        <inertial pos="0.01 0 0" mass="0.005" diaginertia="5e-7 5e-7 5e-7"/>
        <geom type="ellipsoid" size="0.022 0.018 0.016" pos="0.012 0 0" material="body"/>
        <geom type="sphere" size="0.007" pos="0.022 0 0" material="sense" contype="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="aWFL" joint="jWFL" gear="4"/>
    <motor name="aWFR" joint="jWFR" gear="4"/>
    <motor name="aWBL" joint="jWBL" gear="3"/>
    <motor name="aWBR" joint="jWBR" gear="3"/>
    <motor name="aLFL" joint="jLFL" gear="7"/>
    <motor name="aLFR" joint="jLFR" gear="7"/>
    <motor name="aLML" joint="jLML" gear="7"/>
    <motor name="aLMR" joint="jLMR" gear="7"/>
    <motor name="aLBL" joint="jLBL" gear="7"/>
    <motor name="aLBR" joint="jLBR" gear="7"/>
    <motor name="aABD"  joint="jABD"  gear="5"/>   <!-- [10] 복부 굽힘 -->
    <motor name="aHEAD" joint="jHEAD" gear="3"/>   <!-- [11] 머리 방향 -->
  </actuator>
</mujoco>
"""

# ──────────────────────────────────────────────────────
# 키워드 → 자극 파서
# ──────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────
# 키워드 폴백 파서 (Qwen 실패시 사용)
# ──────────────────────────────────────────────────────
KMAP = {
    # ── 전진 / 이동 (한국어) ──────────────────────────
    "앞":1, "앞으로":1, "전진":1, "가":1, "가자":1, "가라":1,
    "빨리":1, "달려":1, "달려라":1, "뛰어":1, "뛰어라":1, "뛰":1,
    "점프":1, "날아":1, "날아봐":1, "날아라":1, "날":1,
    "출발":1, "이동":1, "전방":1, "직진":1, "돌격":1,
    "움직여":1, "움직":1, "일어나":1, "일어서":1,
    "깨어나":1, "깨어":1, "활성화":1,
    "굴러":1, "옆으로":1, "회전":1, "돌아":1,
    # ── 전진 / 이동 (영어) ───────────────────────────
    "go":1, "run":1, "fly":1, "jump":1, "forward":1,
    "move":1, "start":1, "advance":1, "charge":1,
    "wake":1, "up":1, "rise":1, "activate":1,
    "faster":1, "hurry":1, "sprint":1, "dash":1,
    "roll":1, "tumble":1, "spin":1, "turn":1,
    # ── 후진 / 회피 (한국어) ─────────────────────────
    "뒤":2, "뒤로":2, "후진":2, "물러":2, "물러나":2,
    "피해":2, "피":2, "도망":2, "도망쳐":2, "도망가":2,
    "후퇴":2, "뒤집어":2,
    # ── 후진 / 회피 (영어) ──────────────────────────
    "back":2, "backward":2, "retreat":2, "reverse":2,
    "flee":2, "escape":2, "evade":2, "avoid":2, "withdraw":2,
    # ── 위험 / 통각 (한국어) ─────────────────────────
    "위험":3, "위험해":3, "고통":3, "아파":3, "아프다":3,
    "살려":3, "살려줘":3, "공격":3, "적":3, "조심":3,
    "뜨거워":3, "차가워":3, "따가워":3, "찌릿":3,
    "전기":3, "충격":3, "맞아":3, "죽어":3, "죽음":3,
    # ── 위험 / 통각 (영어) ──────────────────────────
    "danger":3, "pain":3, "hurt":3, "ouch":3, "ow":3,
    "attack":3, "threat":3, "hot":3, "burn":3,
    "shock":3, "sting":3, "bite":3, "hit":3,
    "enemy":3, "predator":3, "help":3,
    "death":3, "dead":3, "die":3, "kill":3, "lethal":3,
    # ── 먹이 / 화학 (한국어) ─────────────────────────
    "먹이":4, "먹이냄새":4, "음식":4, "밥":4, "먹어":4,
    "배고파":4, "배고":4, "냄새":4, "달콤":4, "맛있":4,
    "과일":4, "설탕":4, "꿀":4, "먹자":4, "식사":4,
    "간식":4, "영양":4,
    # ── 먹이 / 화학 (영어) ──────────────────────────
    "food":4, "eat":4, "hungry":4, "smell":4, "taste":4,
    "sweet":4, "sugar":4, "feed":4, "meal":4, "snack":4,
    "nutrient":4, "yummy":4, "delicious":4,
    # ── 후각 / 탐색 (한국어) ─────────────────────────
    "빛":5, "향기":5, "좋아":5, "궁금":5, "탐색":5,
    "살펴":5, "보자":5, "여기":5, "저기":5,
    # ── 후각 / 탐색 (영어) ──────────────────────────
    "light":5, "glow":5, "explore":5, "search":5,
    "curious":5, "look":5, "find":5, "discover":5,
    # ── 일본어 ──────────────────────────────────────
    "まえ":1, "すすめ":1, "はしれ":1, "とべ":1,
    "うしろ":2, "にげろ":2, "たいへん":3, "いたい":3,
    "えさ":4, "におい":4,
    # ── 중국어 ──────────────────────────────────────
    "前进":1, "跑":1, "飞":1,
    "后退":2, "危险":3, "痛":3, "食物":4, "气味":4,
}

# 채널 번호 → 이름 매핑
_CH = {1:"touch_front", 2:"touch_back", 3:"nociception", 4:"chemical", 5:"olfactory"}

# 채널별 기본 강도
_STRENGTH = {
    "touch_front":  {"앞":0.85,"앞으로":0.9,"전진":0.9,"가":0.7,"가자":0.8,"가라":0.85,
                     "빨리":1.0,"달려":1.0,"달려라":1.0,"뛰어":0.9,"뛰어라":1.0,"뛰":0.85,
                     "점프":0.9,"날아":0.9,"날아봐":0.95,"날아라":1.0,"날":0.85,
                     "출발":0.9,"이동":0.8,"전방":0.85,"직진":0.9,"돌격":1.0,
                     "움직여":0.8,"움직":0.75,"일어나":0.7,"일어서":0.75,
                     "깨어나":0.7,"깨어":0.65,"활성화":0.8,
                     "go":0.85,"run":1.0,"fly":0.9,"jump":0.9,"forward":0.9,
                     "move":0.8,"start":0.8,"advance":0.9,"charge":1.0,
                     "wake":0.7,"up":0.6,"rise":0.7,"activate":0.8,
                     "faster":1.0,"hurry":0.9,"sprint":1.0,"dash":0.95,
                     "roll":0.8,"tumble":0.8,"spin":0.8,"turn":0.75,
                     "굴러":0.8,"옆으로":0.7,"회전":0.8,"돌아":0.75,
                     "まえ":0.85,"すすめ":0.9,"はしれ":1.0,"とべ":0.9,
                     "前进":0.9,"跑":1.0,"飞":0.9,},
    "touch_back":   {"뒤":0.85,"뒤로":0.9,"후진":0.9,"물러":0.8,"물러나":0.85,
                     "피해":0.8,"피":0.7,"도망":0.9,"도망쳐":1.0,"도망가":0.95,
                     "후퇴":0.9,"뒤집어":0.75,
                     "back":0.9,"backward":0.9,"retreat":0.9,"reverse":0.85,
                     "flee":1.0,"escape":0.95,"evade":0.85,"avoid":0.8,"withdraw":0.85,
                     "うしろ":0.85,"にげろ":1.0,
                     "后退":0.9,},
    "nociception":  {"위험":1.0,"위험해":1.0,"고통":0.9,"아파":0.9,"아프다":0.9,
                     "살려":1.0,"살려줘":1.0,"공격":0.9,"적":0.8,"조심":0.6,
                     "뜨거워":0.8,"차가워":0.7,"따가워":0.8,"찌릿":0.7,
                     "전기":0.7,"충격":0.85,"맞아":0.8,
                     "danger":1.0,"pain":0.9,"hurt":0.8,"ouch":0.9,"ow":0.8,
                     "attack":1.0,"threat":0.9,"hot":0.7,"burn":0.8,
                     "shock":0.85,"sting":0.8,"bite":0.9,"hit":0.8,
                     "enemy":0.9,"predator":1.0,"help":0.7,
                     "death":1.0,"dead":0.9,"die":0.9,"kill":1.0,"lethal":1.0,
                     "죽어":1.0,"죽음":0.9,
                     "たいへん":0.9,"いたい":0.9,
                     "危险":1.0,"痛":0.9,},
    "chemical":     {"먹이":0.9,"먹이냄새":1.0,"음식":0.85,"밥":0.85,"먹어":0.9,
                     "배고파":0.85,"배고":0.8,"냄새":0.7,"달콤":0.8,"맛있":0.85,
                     "과일":0.7,"설탕":0.8,"꿀":0.85,"먹자":0.9,"식사":0.8,
                     "간식":0.75,"영양":0.7,
                     "food":0.9,"eat":0.85,"hungry":0.85,"smell":0.7,"taste":0.8,
                     "sweet":0.8,"sugar":0.8,"feed":0.9,"meal":0.8,"snack":0.75,
                     "nutrient":0.7,"yummy":0.9,"delicious":0.85,
                     "えさ":0.9,"におい":0.7,
                     "食物":0.9,"气味":0.7,},
    "olfactory":    {"빛":0.7,"향기":0.7,"좋아":0.4,"궁금":0.4,"탐색":0.6,
                     "살펴":0.5,"보자":0.4,"여기":0.3,"저기":0.3,
                     "light":0.7,"glow":0.6,"explore":0.6,"search":0.5,
                     "curious":0.5,"look":0.4,"find":0.5,"discover":0.6,},
}

_STOP_KW = {
    "stop","rest","quiet","sleep","freeze","halt","pause","still","idle",
    "멈춰","멈춰라","멈추","정지","쉬어","잠잠","조용","잠","그만",
    "止まれ","止まる","やめ","休め","静止",
    "停","停止","休息","安静",
}

def _kmap_parse(text: str) -> dict:
    t = text.lower()
    if any(kw in t for kw in _STOP_KW):
        return {}          # explicit stop → zero stimulation
    stim = {}
    for kw, ch_num in KMAP.items():
        if kw in t:
            ch = _CH[ch_num]
            val = _STRENGTH.get(ch, {}).get(kw, 0.7)
            stim[ch] = max(stim.get(ch, 0), val)
    return stim if stim else {"olfactory": 0.15}


# ──────────────────────────────────────────────────────
# Qwen 입력 파서
# ──────────────────────────────────────────────────────
_PARSE_SYSTEM = """\
You are a sensory decoder for a digital insect brain.
Given any text input, output ONLY a JSON object mapping active sensory channels to intensity (0.0-1.0).
Channels: touch_front, touch_back, nociception, chemical, olfactory, visual
Rules:
- Movement/forward intent → touch_front
- Backward/retreat intent → touch_back
- Pain/danger/threat → nociception
- Food/smell/taste/hunger → chemical
- Light/curiosity/greeting → olfactory
- Stop/rest/quiet/sleep/identity questions → {} (empty, no stimulation)
- Output ONLY valid JSON, no explanation.
Examples:
"run forward fast" → {"touch_front": 1.0}
"danger attack" → {"nociception": 1.0, "touch_back": 0.7}
"food smell" → {"chemical": 0.9, "olfactory": 0.5}
"배고파" → {"chemical": 0.85}
"위험해 도망쳐" → {"nociception": 1.0, "touch_back": 0.8}
"날아봐" → {"touch_front": 0.95}
"뒤로 물러" → {"touch_back": 0.9}
"안녕" → {"olfactory": 0.3}
"jump" → {"touch_front": 0.9}
"뛰어" → {"touch_front": 0.9}
"roll" → {"touch_front": 0.8}
"굴러" → {"touch_front": 0.8}
"death" → {"nociception": 1.0}
"dead" → {"nociception": 0.9}
"die" → {"nociception": 0.9}
"bye" → {}
"goodbye" → {}
"잘있어" → {}
"""

def make_parser(llm) -> "callable":
    """Voice.llm을 받아 Qwen 기반 파서 반환. llm=None이면 kmap 폴백."""
    import json as _json

    def parse(text: str) -> dict:
        if llm is None:
            return _kmap_parse(text)
        try:
            out = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": _PARSE_SYSTEM},
                    {"role": "user",   "content": text},
                ],
                max_tokens=60,
                temperature=0.1,          # 결정론적으로
                stop=["\n", "```"],
            )
            raw = out["choices"][0]["message"]["content"].strip()
            # JSON 파싱
            result = _json.loads(raw)
            # 유효한 채널만, 0~1 범위로 클리핑
            channels = {"touch_front","touch_back","nociception","chemical","olfactory","visual"}
            stim = {k: float(min(max(v, 0.0), 1.0))
                    for k, v in result.items() if k in channels and v > 0}
            return stim if stim else _kmap_parse(text)
        except Exception:
            return _kmap_parse(text)

    return parse


def parse(text: str) -> dict:
    """전역 파서. main()에서 Qwen 로드 후 make_parser()로 교체됨."""
    return _kmap_parse(text)


# ──────────────────────────────────────────────────────
# 액추에이터 제어
# ──────────────────────────────────────────────────────
def apply_controls(data, sig: dict, t: float):
    fwd     = sig.get("forward",    0.0)
    back    = sig.get("backward",   0.0)
    wing    = sig.get("wing",       0.0)
    tL      = sig.get("turn_left",  0.0)
    tR      = sig.get("turn_right", 0.0)
    curl    = sig.get("curl",       0.0)
    eat     = sig.get("eat",        0.0)
    tremble = sig.get("tremble",    0.0)
    active  = sig.get("active",     0.0)
    net     = fwd - back

    # ── [0-3] 날개 ────────────────────────────────────
    wamp = wing * 0.9
    data.ctrl[0] = -wamp * math.sin(t * 25.0)
    data.ctrl[1] =  wamp * math.sin(t * 25.0)
    data.ctrl[2] = -wamp * math.sin(t * 25.0 + 0.3)
    data.ctrl[3] =  wamp * math.sin(t * 25.0 + 0.3)

    # ── [4-9] 다리 ────────────────────────────────────
    PHASE = [0.0, math.pi, math.pi/3, 4*math.pi/3, 2*math.pi/3, 5*math.pi/3]
    if tremble > 0.2:
        # 떨기: 빠른 진동, 작은 진폭
        for i, phi in enumerate(PHASE):
            data.ctrl[4+i] = tremble * 0.4 * math.sin(t * 15.0 + phi)
    elif curl > 0.5:
        # 웅크리기: 다리 접기
        for i in range(6):
            data.ctrl[4+i] = np.clip(-curl * 0.7, -1, 1)
    elif active < 0.01:
        # 완전 정지: 모든 다리 고정
        for i in range(6):
            data.ctrl[4+i] = 0.0
    else:
        # 일반 보행
        for i, phi in enumerate(PHASE):
            amp  = abs(net) + 0.25
            bias = (tL - tR) * 0.4 if i % 2 == 0 else (tR - tL) * 0.4
            val  = amp * math.sin(t * 4.0 + phi) * (1.0 if net >= 0 else -1.0) + bias
            data.ctrl[4+i] = np.clip(val, -1, 1)

    # ── [10] 복부 굽힘 ────────────────────────────────
    if curl > 0.3:
        # 위험 시 복부를 들어올림
        data.ctrl[10] = np.clip(curl * 0.8, -1, 1)
    elif eat > 0.3:
        # 먹을 때 복부 앞쪽으로 굽힘
        data.ctrl[10] = np.clip(-eat * 0.5 * math.sin(t * 3.0), -1, 1)
    elif back > fwd + 0.1:
        # 후진 시 복부 약간 들림
        data.ctrl[10] = np.clip(back * 0.3, -1, 1)
    else:
        data.ctrl[10] = 0.0

    # ── [11] 머리 방향 ────────────────────────────────
    if eat > 0.3:
        # 먹을 때 머리 좌우 스캔
        data.ctrl[11] = eat * 0.6 * math.sin(t * 2.0)
    elif tL > tR + 0.1:
        data.ctrl[11] = np.clip(tL * 0.5, -1, 1)
    elif tR > tL + 0.1:
        data.ctrl[11] = np.clip(-tR * 0.5, -1, 1)
    else:
        # 탐색 중 느린 좌우 움직임
        active = sig.get("active", 0)
        data.ctrl[11] = active * 0.2 * math.sin(t * 1.5)


# ──────────────────────────────────────────────────────
# 공유 상태
# ──────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.sig    = {"forward":0,"backward":0,"wing":0.3,"active":0.3,
                       "turn_left":0,"turn_right":0,
                       "curl":0,"eat":0,"tremble":0,
                       "_total_spikes":0}
        self.stim   = {"olfactory":0.15}
        self.speech = ""
        self.lock   = threading.Lock()

state = State()


# ──────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────
def main():
    # macOS: MuJoCo viewer requires mjpython
    import platform
    if platform.system() == "Darwin":
        # 이미 mjpython으로 실행 중인지 환경변수로 체크
        if not os.environ.get("CHIMERA_MJPYTHON"):
            import shutil, subprocess, glob
            mjpython = shutil.which("mjpython")
            if not mjpython:
                candidates = (
                    glob.glob(os.path.expanduser("~/Library/Python/*/bin/mjpython")) +
                    glob.glob("/usr/local/bin/mjpython") +
                    glob.glob("/opt/homebrew/bin/mjpython")
                )
                mjpython = candidates[0] if candidates else None
            if mjpython:
                env = os.environ.copy()
                env["CHIMERA_MJPYTHON"] = "1"
                ret = subprocess.run([mjpython] + sys.argv, env=env)
                sys.exit(ret.returncode)
            else:
                print("⚠️  mjpython not found. Try:")
                print("   pip3 install mujoco --upgrade")
                print("   mjpython chimera_app.py")
                sys.exit(1)

    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        sys.exit("❌  pip install mujoco")

    # 커넥톰 선택
    ensure_fallback()
    if os.path.exists(CONNECTOME_NPY) and os.path.exists(CONNECTOME_JSON):
        npy, jsn = CONNECTOME_NPY, CONNECTOME_JSON
        src = "Drosophila larva (Winding 2023)"
    else:
        npy, jsn = FALLBACK_NPY, FALLBACK_JSON
        src = "C. elegans fallback"

    print("=" * 55)
    print("  CHIMERA — Digital Lifeform")
    print(f"  Brain: {src}")
    print("=" * 55)

    brain = Brain(npy, jsn)
    voice = Voice(MODEL_GGUF)

    # Qwen이 로드됐으면 입력 파서를 Qwen으로 교체
    global parse
    parse = make_parser(voice.llm)
    if voice.llm:
        print("🔍 Input parser: Qwen (natural language → sensory channels)")
    else:
        print("🔍 Input parser: keyword matching (fallback)")

    # MCP 커맨드 파일 경로
    _MCP_CMD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_cmd.txt")

    def _process_text(text):
        """Parse text and update shared state. Returns speech string."""
        stim = parse(text)
        sig  = brain.run(stim)
        word = voice.speak(stim, sig, input_text=text)
        with state.lock:
            state.stim   = stim
            state.sig    = sig
            state.speech = word
        print(f"\n  [MCP] {text!r} → spikes={sig['_total_spikes']}  "
              f"fwd={sig['forward']:.2f}  back={sig['backward']:.2f}")
        print(f"  💬 \"{word}\"\n")
        return word

    def _poll_mcp_cmd():
        """Background thread: check mcp_cmd.txt every 0.2s and execute."""
        while True:
            try:
                if os.path.exists(_MCP_CMD_FILE):
                    with open(_MCP_CMD_FILE, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    os.remove(_MCP_CMD_FILE)
                    if text:
                        _process_text(text)
            except Exception:
                pass
            time.sleep(0.2)

    threading.Thread(target=_poll_mcp_cmd, daemon=True).start()

    # 입력 스레드
    def input_loop():
        print("\nInput mode (quit: q)")
        print("e.g.  danger / food smell / fly / back\n")
        while True:
            try:
                text = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text: continue
            if text.lower() in ("q","quit","exit","종료"):
                os._exit(0)
            stim = parse(text)
            sig  = brain.run(stim)
            word = voice.speak(stim, sig, input_text=text)
            with state.lock:
                state.stim  = stim
                state.sig   = sig
                state.speech = word
            print(f"\n  spikes={sig['_total_spikes']}  "
                  f"fwd={sig['forward']:.2f}  back={sig['backward']:.2f}  "
                  f"wing={sig['wing']:.2f}")
            print(f"\n  💬 \"{word}\"\n")

    threading.Thread(target=input_loop, daemon=True).start()

    # MuJoCo
    model = mujoco.MjModel.from_xml_string(CHIMERA_XML)
    data  = mujoco.MjData(model)
    data.qpos[2] = 0.18

    print("\n🪲 Viewer starting...")
    import platform
    is_mac = platform.system() == "Darwin"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance  = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth   = 45
        t, step_n = 0.0, 0
        last_time = time.time()

        _MCP_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_state.json")
        _last_mcp_check = 0.0

        while viewer.is_running():
            # Poll shared file every 100ms — MCP writes here to override state
            _now = time.time()
            if _now - _last_mcp_check > 0.1:
                _last_mcp_check = _now
                if os.path.exists(_MCP_STATE_FILE):
                    try:
                        import json as _json
                        with open(_MCP_STATE_FILE, "r") as _f:
                            _mcp = _json.load(_f)
                        with state.lock:
                            state.sig.update(_mcp)
                    except Exception:
                        pass

            with state.lock:
                sig_now = dict(state.sig)
            apply_controls(data, sig_now, t)
            mujoco.mj_step(model, data)

            # 불안정 감지 기준 완화 (macOS에서 더 민감함)
            qvel_limit = 30.0
            if (not np.all(np.isfinite(data.qpos)) or
                    np.any(np.abs(data.qvel) > qvel_limit)):
                mujoco.mj_resetData(model, data)
                data.qpos[2] = 0.18
                t = 0.0

            t += model.opt.timestep
            step_n += 1
            if step_n % 100 == 0:
                viewer.cam.lookat[0] = data.qpos[0]
                viewer.cam.lookat[1] = data.qpos[1]
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)


if __name__ == "__main__":
    main()
