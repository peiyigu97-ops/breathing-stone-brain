"""
CHIMERA — 진짜 파리 유충 커넥톰 다운로더
==========================================
출처: Winding et al. Science 2023
      "The connectome of an insect brain"
      3,016 뉴런 / 548,000 시냅스

실행: python chimera_real_connectome.py

계정 불필요. GitHub에서 직접 다운로드.
"""

import urllib.request
import os
import sys
from pathlib import Path

Path("chimera/connectome").mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────
# 실제 파일 URL (brain-networks/larval-drosophila-connectome)
# ──────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/brain-networks/larval-drosophila-connectome/main"

FILES = {
    # 시냅스 연결 행렬 (3016 x 3016, 가중치)
    "chimera/connectome/real_adjacency.csv": f"{BASE}/data/Aw.csv",
    # 뉴런 메타데이터 (타입, 클래스, 반구 등)
    "chimera/connectome/real_neurons.csv":   f"{BASE}/data/nodeMeta.csv",
}

# 백업 파일명 (repo 구조에 따라 다를 수 있음)
FALLBACK = {
    "chimera/connectome/real_adjacency.csv": [
        f"{BASE}/data/Aw.csv",
        f"{BASE}/data/adjacency_matrix.csv",
        f"{BASE}/Aw.csv",
        f"{BASE}/adjacency.csv",
    ],
    "chimera/connectome/real_neurons.csv": [
        f"{BASE}/data/nodeMeta.csv",
        f"{BASE}/data/neuron_annotations.csv",
        f"{BASE}/nodeMeta.csv",
        f"{BASE}/neurons.csv",
    ],
}


def download(url: str, dest: str) -> bool:
    try:
        print(f"  시도: {url.split('/')[-1]}", end=" ... ", flush=True)
        urllib.request.urlretrieve(url, dest)
        size = os.path.getsize(dest)
        if size < 100:   # 너무 작으면 오류 페이지일 가능성
            os.remove(dest)
            print("❌ (파일 너무 작음)")
            return False
        print(f"✅ ({size/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"❌ ({e})")
        return False


print("=" * 55)
print("  진짜 Drosophila 유충 커넥톰 다운로드")
print("  Winding et al. Science 2023")
print("=" * 55)
print()

success = {}
for dest, urls in FALLBACK.items():
    print(f"📥 {os.path.basename(dest)}")
    for url in urls:
        if download(url, dest):
            success[dest] = True
            break
    else:
        success[dest] = False
    print()

# ──────────────────────────────────────────────────────
# 결과 확인
# ──────────────────────────────────────────────────────
adj_path  = "chimera/connectome/real_adjacency.csv"
meta_path = "chimera/connectome/real_neurons.csv"

if not all(success.values()):
    print("─" * 55)
    print("⚠️  자동 다운로드 실패. 수동 다운로드 방법:")
    print()
    print("  1. 브라우저에서 이 주소 열기:")
    print("     https://github.com/brain-networks/larval-drosophila-connectome")
    print()
    print("  2. 초록 'Code' 버튼 → 'Download ZIP' 클릭")
    print()
    print("  3. 압축 풀기 후 이 파일들을 복사:")
    print(f"     data/Aw.csv        →  {adj_path}")
    print(f"     data/nodeMeta.csv  →  {meta_path}")
    print()
    print("  4. 이 스크립트 다시 실행")
    sys.exit(1)

# ──────────────────────────────────────────────────────
# 파싱 및 검증
# ──────────────────────────────────────────────────────
print("🔬 데이터 검증 중...")

try:
    import numpy as np
    import csv

    # 뉴런 메타데이터
    neurons = []
    with open(meta_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            neurons.append(row)

    print(f"   뉴런 수     : {len(neurons)}")
    if neurons:
        print(f"   컬럼        : {list(neurons[0].keys())}")

    # 타입 분포
    types = {}
    for n in neurons:
        t = n.get('type', n.get('class', n.get('cell_type', 'unknown')))
        types[t] = types.get(t, 0) + 1
    print(f"   뉴런 타입 수: {len(types)}")
    for t, c in sorted(types.items(), key=lambda x: -x[1])[:8]:
        print(f"     {t:30s}: {c}")

    # 시냅스 행렬 미리보기 (첫 5행)
    print()
    print(f"   시냅스 행렬 미리보기:")
    with open(adj_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = [next(reader) for _ in range(3)]
    print(f"     행 길이: {len(rows[0])} 컬럼")
    print(f"     예시: {rows[0][:5]} ...")

except Exception as e:
    print(f"   파싱 오류: {e}")
    print("   → 파일은 있지만 포맷이 다를 수 있음. chimera_load_connectome.py로 진행.")

print()
print("─" * 55)
print("✅ 다운로드 완료!")
print()
print("다음: python chimera_load_connectome.py")
print("       (3,016뉴런 행렬 → CHIMERA 뇌 엔진에 로드)")
