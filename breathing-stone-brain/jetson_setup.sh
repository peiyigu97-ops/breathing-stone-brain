#!/bin/bash
# jetson_setup.sh
# ================
# Deploy Breathing Stone + HumanBrainDT on Jetson Nano 4GB
# JetPack 4.6+ (Ubuntu 18.04, Python 3.8)
#
# Usage:
#   chmod +x jetson_setup.sh
#   ./jetson_setup.sh

set -e
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}=== Breathing Stone · Jetson Nano Setup ===${NC}"

# ── 1. System deps ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[1/7] System packages${NC}"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libopenblas-dev libatlas-base-dev \
    libhdf5-serial-dev hdf5-tools \
    git curl screen

# ── 2. Python deps ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[2/7] Python packages${NC}"

# Numpy — use pre-built wheel for Jetson (aarch64)
pip3 install --upgrade pip
pip3 install numpy --extra-index-url https://pypi.ngc.nvidia.com

# Core
pip3 install fastapi uvicorn[standard] websockets pyserial bleak

# TGAM / bridge
pip3 install scipy  # optional: for signal processing extensions

# ── 3. Clone / copy chimera ──────────────────────────────────────────────────
echo -e "\n${YELLOW}[3/7] Chimera source${NC}"
CHIMERA_DIR="$HOME/chimera"

if [ -d "$CHIMERA_DIR/.git" ]; then
    echo "  Repo exists, pulling latest..."
    cd "$CHIMERA_DIR" && git pull
else
    echo "  Cloning from GitHub..."
    git clone https://github.com/caparison1234/chimera.git "$CHIMERA_DIR"
fi

cd "$CHIMERA_DIR"

# ── 4. Connectome data (if not present) ──────────────────────────────────────
echo -e "\n${YELLOW}[4/7] Connectome data${NC}"
NPY="$CHIMERA_DIR/chimera/connectome/real_weight_matrix.npy"
if [ ! -f "$NPY" ]; then
    echo "  Downloading connectome..."
    python3 chimera_real_connectome.py || python3 chimera_load_connectome.py
else
    echo "  Connectome already present."
fi

# ── 5. Skip MuJoCo on Jetson (no display / GPU mismatch) ─────────────────────
echo -e "\n${YELLOW}[5/7] MuJoCo skip (headless Jetson)${NC}"
echo "  NOTE: chimera_app.py (MuJoCo viewer) will NOT run on Jetson."
echo "  Use stone_bridge.py + HumanBrainDT viewer instead."
echo "  The viewer runs in browser via FastAPI WebSocket."

# ── 6. Viewer port ────────────────────────────────────────────────────────────
echo -e "\n${YELLOW}[6/7] Viewer setup${NC}"
VIEWER="$CHIMERA_DIR/HumanBrainDT/viewer/server.py"
echo "  Viewer entry: $VIEWER"
echo "  Start with:   python3 $VIEWER"
echo "  Then open:    http://<JETSON_IP>:7860"

# ── 7. Systemd service (optional auto-start) ─────────────────────────────────
echo -e "\n${YELLOW}[7/7] Systemd service (optional)${NC}"
SERVICE_FILE="/etc/systemd/system/stone-bridge.service"
if [ ! -f "$SERVICE_FILE" ]; then
cat << EOF | sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=Breathing Stone Bridge
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$CHIMERA_DIR
ExecStart=/usr/bin/python3 $CHIMERA_DIR/stone_bridge.py --serial /dev/ttyUSB0 --out /tmp/stone_session.csv
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    echo "  Service installed. Enable with:"
    echo "    sudo systemctl enable stone-bridge"
    echo "    sudo systemctl start  stone-bridge"
else
    echo "  Service already exists."
fi

echo -e "\n${GREEN}=== Setup complete ===${NC}"
echo ""
echo "  Quick start (demo mode):"
echo "    cd $CHIMERA_DIR"
echo "    python3 stone_bridge.py --demo --steps 60 --fast"
echo ""
echo "  Live ESP32 serial:"
echo "    python3 stone_bridge.py --serial /dev/ttyUSB0 --out session.csv"
echo ""
echo "  3D Brain Viewer:"
echo "    python3 HumanBrainDT/viewer/server.py"
echo "    → http://$(hostname -I | awk '{print $1}'):7860"
echo ""
