"""
chimera_load_connectome.py  (수정본)
annotations.csv 실제 컬럼:
  left_id, right_id, celltype, additional_annotations, level_7_cluster
"""
import csv, json, os
import numpy as np
from pathlib import Path

ADJ_PATH  = "chimera/connectome/real_adjacency.csv"
META_PATH = "chimera/connectome/real_neurons.csv"
OUT_DIR   = Path("chimera/connectome")

for p in [ADJ_PATH, META_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"파일 없음: {p}")

# ── 1. 뉴런 메타데이터 ────────────────────────────────
print("📂 뉴런 메타데이터 로드...")
with open(META_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

neurons, celltype_map, annotation_map = [], {}, {}
for i, row in enumerate(rows):
    rid = row.get('right_id', '').strip()
    lid = row.get('left_id', '').strip()
    nid = rid if rid and rid != 'no pair' else lid
    if not nid:
        nid = str(i)
    neurons.append(nid)
    celltype_map[nid]   = row.get('celltype', 'inter').strip()
    annotation_map[nid] = row.get('additional_annotations', '').strip().lower()

N   = len(neurons)
idx = {n: i for i, n in enumerate(neurons)}
print(f"   총 뉴런: {N}")

ct = {}
for v in celltype_map.values():
    ct[v] = ct.get(v, 0) + 1
for t, c in sorted(ct.items(), key=lambda x: -x[1]):
    print(f"   {t:30s}: {c}")

an = {}
for v in annotation_map.values():
    if v: an[v] = an.get(v, 0) + 1
print(f"\n   annotation 상위 10:")
for t, c in sorted(an.items(), key=lambda x: -x[1])[:10]:
    print(f"   {t:35s}: {c}")

# ── 2. 채널 매핑 (실제 annotation 문자열 기반) ────────
# celltype='sensory' 제한 없이 annotation으로만 분류
CHANNEL_MAP = {
    "touch_front":  ["mechano","chordotonal","propriocept","touch"],
    "touch_back":   ["mechano","chordotonal"],
    "nociception":  ["noci","multidendritic","md "],
    "chemical":     ["gustatory","taste","pharyngeal","gut"],
    "olfactory":    ["olfactory","olfact","or42","or85"],
    "visual":       ["visual","photo","optic"],
}

channel_neurons = {ch: [] for ch in CHANNEL_MAP}
seen = set()
for nid in neurons:
    ann = annotation_map[nid]
    ct  = celltype_map[nid]
    if not ann or ann == 'no official annotation':
        continue
    for ch, keywords in CHANNEL_MAP.items():
        if nid in seen and ch != "touch_back":
            continue
        for kw in keywords:
            if kw.lower() in ann:
                channel_neurons[ch].append(nid)
                seen.add(nid)
                break

print(f"\n   채널별 뉴런:")
for ch, ns in channel_neurons.items():
    print(f"   {ch:15s}: {len(ns)}개")

# ── 3. 기능 분류 (실제 celltype명 기반) ──────────────
# DN-VNC, DN-SEZ → 전진 운동 (하행 뉴런)
# pre-DN-VNC, pre-DN-SEZ → 운동 전 단계
# ascending → 출력
MOTOR_TYPES  = {"DN-VNC", "DN-SEZ"}
PREMOTOR_TYPES = {"pre-DN-VNC", "pre-DN-SEZ"}
OUTPUT_TYPES = {"ascending", "MBON"}

func_map = {}
for nid in neurons:
    ct = celltype_map[nid]
    if ct == 'sensory':               func_map[nid] = 'sensory'
    elif ct in MOTOR_TYPES:           func_map[nid] = 'motor'
    elif ct in PREMOTOR_TYPES:        func_map[nid] = 'premotor'
    elif ct in OUTPUT_TYPES:          func_map[nid] = 'output'
    else:                             func_map[nid] = 'inter'

sensory_ids = [n for n,f in func_map.items() if f=='sensory']
motor_ids   = [n for n,f in func_map.items() if f in ('motor','premotor')]
inter_ids   = [n for n,f in func_map.items() if f=='inter']
output_ids  = [n for n,f in func_map.items() if f=='output']
print(f"\n   감각:{len(sensory_ids)}  운동:{len(motor_ids)}  "
      f"중간:{len(inter_ids)}  출력:{len(output_ids)}")

# ── 4. 시냅스 행렬 ────────────────────────────────────
print("\n📂 시냅스 행렬 로드...")
with open(ADJ_PATH, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    first  = next(reader)
    try:
        float(first[0]); adj_rows = [first]
    except ValueError:
        adj_rows = []
    for row in reader:
        adj_rows.append(row)

print(f"   행렬: {len(adj_rows)} × {len(adj_rows[0]) if adj_rows else 0}")

# 행렬이 뉴런 수보다 크면 경고 (양쪽 반구 포함 행렬인 경우)
M = len(adj_rows)
if M != N:
    print(f"   ⚠️  행렬({M}) ≠ 뉴런({N}) → 앞 {N}×{N} 사용")

W = np.zeros((N, N), dtype=np.float32)
nz = 0
for i, row in enumerate(adj_rows[:N]):
    for j, val in enumerate(row[:N]):
        try:
            v = float(val)
            if v:
                W[i, j] = v; nz += 1
        except (ValueError, TypeError):
            pass

nz_vals = W[W != 0]
p99 = float(np.percentile(nz_vals, 99)) if len(nz_vals) else 1.0
W = np.clip(W / p99, 0.0, 1.0).astype(np.float32)
print(f"   비제로: {nz:,}  p99기준정규화: {p99:.1f}  (이상치 클리핑)")

# ── 5. 저장 ──────────────────────────────────────────
np.save(OUT_DIR / "real_weight_matrix.npy", W)
with open(OUT_DIR / "real_neurons.json", 'w') as f:
    json.dump({
        "source":   "Winding et al. Science 2023",
        "neurons":  neurons, "types": celltype_map,
        "func":     func_map, "index": idx, "count": N,
        "sensory":  sensory_ids, "motor": motor_ids,
        "inter":    inter_ids,   "output": output_ids,
        "channels": channel_neurons,
        "stats":    {"n_neurons": N, "n_synapses": nz},
    }, f, indent=2, ensure_ascii=False)

print("\n✅ 저장 완료 → python chimera_app.py")
