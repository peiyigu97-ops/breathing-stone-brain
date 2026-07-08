# chimera.spec  (최종 빌드용)
import sys, os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── 데이터 파일 ───────────────────────────────────────
datas = []

# 커넥톰 (실제 경로: chimera/connectome/)
for src, dst in [
    ("chimera/connectome/real_weight_matrix.npy", "chimera/connectome"),
    ("chimera/connectome/real_neurons.json",       "chimera/connectome"),
]:
    if os.path.exists(src):
        datas.append((src, dst))
        print(f"[spec] 포함: {src}")
    else:
        print(f"[spec] WARN 없음: {src}")

# Qwen LLM (실제 경로: chimera/models/)
gguf = "chimera/models/Qwen2.5-0.5B-Instruct.Q4_K_M.gguf"
if os.path.exists(gguf):
    datas.append((gguf, "chimera/models"))
    print(f"[spec] LLM 포함: {gguf}")
else:
    print(f"[spec] WARN LLM 없음 → 규칙 기반 모드")

# MuJoCo 데이터
try:
    datas += collect_data_files("mujoco")
except Exception as e:
    print(f"[spec] mujoco collect_data_files: {e}")

# ── 바이너리 ─────────────────────────────────────────
binaries = []
try:
    binaries += collect_dynamic_libs("mujoco")
except Exception as e:
    print(f"[spec] mujoco collect_dynamic_libs: {e}")

# llama_cpp .dll/.so
try:
    binaries += collect_dynamic_libs("llama_cpp")
except Exception:
    pass

# ── 숨겨진 임포트 ────────────────────────────────────
hiddenimports = [
    "mujoco", "mujoco.viewer",
    "numpy", "numpy.core._multiarray_umath",
    "json", "threading", "math",
]
try:
    import llama_cpp
    hiddenimports += ["llama_cpp", "llama_cpp.llama"]
except ImportError:
    pass
try:
    import glfw
    hiddenimports.append("glfw")
except ImportError:
    pass

# ── 분석 ────────────────────────────────────────────
a = Analysis(
    ["chimera_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "matplotlib", "scipy", "pandas", "PIL", "cv2",
        "tkinter", "wx", "PyQt5", "PySide2",
        "IPython", "jupyter", "notebook",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="CHIMERA",
    debug=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name="CHIMERA",
)
