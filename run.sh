#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  —  Complete OOP Detection pipeline (Path A: learned head)
#
# Run from the solution/ directory:
#   chmod +x run.sh
#   ./run.sh
#
# Each phase prints its status.  You can re-run individual steps by
# calling the Python scripts or cmake/make directly.
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on first error
cd "$(dirname "$0")"

BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; RESET="\033[0m"
step() { echo -e "\n${BOLD}${CYAN}══ $1 ══${RESET}"; }
ok()   { echo -e "${GREEN}[OK] $1${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 0 — Python dependencies"
pip install -r requirements.txt --break-system-packages -q
ok "Dependencies installed"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Export YOLOv8 model to ONNX
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 1 — Export model"
python tools/0_export_model.py
ok "models/yolov8m-pose.onnx ready"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Dashboard ROI calibration (MANUAL STEP)
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 2 — Dashboard ROI calibration  (MANUAL)"
echo ""
echo "  This step REQUIRES your input:"
echo "  1. Run:  python tools/1_roi_picker.py"
echo "  2. Click 4 corners of the dashboard in the window"
echo "  3. Copy the printed C++ code into include/config.hpp (Section 2)"
echo "  4. Run:  python tools/2_verify_roi.py  to confirm the ROI looks right"
echo ""
read -rp "  Press ENTER after you have updated config.hpp with your ROI coords: "
ok "ROI calibration acknowledged"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Path A training chain
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 3 — Feature extraction (runs YOLOv8 on all videos)"
echo "  This takes 20-40 minutes.  Watch GPU with: watch -n 1 nvidia-smi"
python tools/3_extract_features.py
ok "data/features.csv written"

step "PHASE 3 — Auto-labeling"
python tools/4_auto_label.py
ok "data/labeled.csv written"

step "PHASE 3 — Training classifier head"
python tools/5_train_head.py
ok "models/head.pkl written"

step "PHASE 3 — Leave-one-profile-out validation"
python tools/6_validate_head.py

step "PHASE 3 — Export weights to C++ constants"
python tools/7_export_head.py
echo ""
echo "  *** MANUAL ACTION REQUIRED ***"
echo "  Open include/config.hpp and replace Section 7"
echo "  with the block printed above (or copy from models/head_cpp.txt)."
echo "  Make sure USE_LEARNED_HEAD = true"
echo ""
read -rp "  Press ENTER after you have updated config.hpp with the head weights: "
ok "Learned head weights acknowledged"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — C++ build
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 4 — C++ build"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DORT_ROOT=/usr/local/onnxruntime 2>&1 | tail -5
make -j"$(nproc)"
cd ..
ok "build/oop_detect compiled"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — Run inference on all videos
# ─────────────────────────────────────────────────────────────────────────────
step "PHASE 5 — Run OOP detection on all videos"
mkdir -p ../output
./build/oop_detect ../input ../output
ok "All output videos written to ../output/"

echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Pipeline complete.  Output in ../output/${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════${RESET}"
