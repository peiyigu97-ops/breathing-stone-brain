import os
import sys
import urllib.request
import zipfile
import platform
from pathlib import Path

GITHUB = "https://raw.githubusercontent.com/caparison1234/chimera/main"
IS_MAC  = platform.system() == "Darwin"
IS_WIN  = platform.system() == "Windows"
PIP     = "pip3" if IS_MAC else "pip"
PYTHON  = "python3" if IS_MAC else "python"

Path("chimera/connectome").mkdir(parents=True, exist_ok=True)
Path("chimera/models").mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("  CHIMERA - Setup")
print(f"  Platform: {platform.system()}")
print("  github.com/caparison1234/chimera")
print("=" * 55)
print()

def download(url, dest, label=""):
    if os.path.exists(dest):
        print(f"  Already exists: {dest}")
        return True
    try:
        print(f"  Downloading {label}...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, 'wb') as f:
            f.write(r.read())
        size = os.path.getsize(dest)
        if size < 100:
            os.remove(dest)
            print("FAILED (file too small)")
            return False
        print(f"OK ({size/1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False

# ── 1. Source files ───────────────────────────────────
print("[1/4] Source files")
for fname in ["chimera_app.py", "chimera_load_connectome.py", "chimera_real_connectome.py"]:
    download(f"{GITHUB}/{fname}", fname, fname)
print()

# ── 2. Packages ───────────────────────────────────────
print("[2/4] Installing packages...")
if IS_MAC:
    # macOS: llama-cpp needs metal support
    ret = os.system(f"{PIP} install mujoco numpy --quiet")
    ret2 = os.system(
        "CMAKE_ARGS='-DGGML_METAL=on' "
        f"{PIP} install llama-cpp-python --quiet"
    )
    if ret2 != 0:
        # Metal 실패시 CPU 폴백
        os.system(f"{PIP} install llama-cpp-python --quiet")
else:
    ret = os.system(f"{PIP} install mujoco numpy llama-cpp-python --quiet")
if ret != 0:
    print("  [WARN] Some packages may have failed. Continue anyway.")
print()

# ── 3. Connectome (Winding et al. Science 2023) ───────
print("[3/4] Drosophila larva connectome (Winding 2023)")

ADJ_PATH  = "chimera/connectome/real_adjacency.csv"
META_PATH = "chimera/connectome/real_neurons.csv"

if os.path.exists(ADJ_PATH) and os.path.exists(META_PATH):
    print("  Already exists.")
else:
    ZIP_URL  = ("https://github.com/brain-networks/larval-drosophila-connectome"
                "/raw/main/Supplementary-Data-S1.zip")
    ZIP_PATH = "chimera/connectome/connectome.zip"

    ok = download(ZIP_URL, ZIP_PATH, "connectome ZIP (~10MB)")
    if not ok:
        print()
        print("  Auto-download failed. Manual steps:")
        print("  1. https://github.com/brain-networks/larval-drosophila-connectome")
        print("  2. Download Supplementary-Data-S1.zip")
        print("  3. Extract: ad_connectivity_matrix.csv -> chimera/connectome/real_adjacency.csv")
        print("              annotations.csv            -> chimera/connectome/real_neurons.csv")
        print("  4. Run setup.py again")
        sys.exit(1)

    print("  Extracting...", end=" ", flush=True)
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            names = z.namelist()
            # __MACOSX 메타 파일 제외
            clean = [n for n in names if "__MACOSX" not in n]
            aw_name   = next((n for n in clean if "ad_connectivity_matrix" in n), None)
            meta_name = next((n for n in clean if "annotations" in n and n.endswith(".csv")), None)
            # 폴백
            if not aw_name:
                aw_name = next((n for n in clean if "connectivity_matrix" in n and n.endswith(".csv")), None)
            if not meta_name:
                meta_name = next((n for n in clean if ("meta" in n.lower() or "node" in n.lower()) and n.endswith(".csv")), None)
            if aw_name:
                with z.open(aw_name) as src, open(ADJ_PATH, 'wb') as dst:
                    dst.write(src.read())
            if meta_name:
                with z.open(meta_name) as src, open(META_PATH, 'wb') as dst:
                    dst.write(src.read())
        os.remove(ZIP_PATH)
        print("OK")
    except Exception as e:
        print(f"FAILED ({e})")
        sys.exit(1)

    if not os.path.exists(ADJ_PATH) or not os.path.exists(META_PATH):
        print("  Could not locate required CSV files in ZIP")
        sys.exit(1)

print("  Parsing connectome...")
ret = os.system(f"{PYTHON} chimera_load_connectome.py")
if ret != 0:
    print("  Parse failed.")
    sys.exit(1)
print()

# ── 4. Qwen 0.5B ─────────────────────────────────────
print("[4/4] Qwen2.5 0.5B LLM (~400MB)")
GGUF_PATH = "chimera/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"
GGUF_URL  = ("https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF"
             "/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf")

if not download(GGUF_URL, GGUF_PATH, "Qwen2.5 0.5B Q4_K_M"):
    print()
    print("  Manual: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF")
    print(f"  Place at: {GGUF_PATH}")
    print("  (Without LLM, CHIMERA still runs in rule-based mode)")

print()
print("=" * 55)
print("  Setup complete!")
print(f"  Run: {PYTHON} chimera_app.py")
print("=" * 55)
