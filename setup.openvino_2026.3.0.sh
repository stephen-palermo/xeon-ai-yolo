#!/bin/bash
# Setup script: OpenVINO 2026.2.1 runtime environment for YOLO26 benchmarking
# Creates a dedicated virtual environment and installs OpenVINO + Ultralytics
# (for YOLO26 export) + headless OpenCV.

set -e  # Exit on any error

rm -rf models
rm -rf venv_openvino_2026.3.0

VENV_DIR="venv_openvino_2026.3.0"
# torch/NNCF (TorchScript) don't support Python 3.14+ and segfault during INT8
# calibration, so pick a supported interpreter (3.10-3.12) if available.
PYTHON_CMD=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        case "$ver" in
            3.10|3.11|3.12) PYTHON_CMD="$candidate"; break ;;
        esac
    fi
done

# No system interpreter in the supported range: provision one with uv, which
# downloads a standalone CPython without touching the system Python.
if [ -z "$PYTHON_CMD" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "No supported system Python (3.10-3.12) found; bootstrapping uv to" >&2
        echo "provision a standalone Python 3.12 (torch/NNCF INT8 segfaults on 3.14)." >&2
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # uv installs to ~/.local/bin (or $XDG_BIN_HOME); make it visible now.
        export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
        [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    fi
    if command -v uv >/dev/null 2>&1; then
        echo "Provisioning standalone Python 3.12 via uv"
        uv python install 3.12
        PYTHON_CMD="$(uv python find 3.12)"
    fi
fi

if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
    echo "ERROR: No supported Python (3.10-3.12) found and uv bootstrap failed." >&2
    echo "       torch/NNCF INT8 export segfaults on Python 3.14. Install e.g.:" >&2
    echo "         sudo apt-get install -y python3.12 python3.12-venv" >&2
    exit 1
fi
echo "Using interpreter: ${PYTHON_CMD} ($(${PYTHON_CMD} --version 2>&1))"

echo "Creating virtual environment: ${VENV_DIR}"
"$PYTHON_CMD" -m venv "$VENV_DIR"

echo "Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing OpenVINO 2026.3.0"
pip install openvino==2026.3.0

echo "Installing CPU-only PyTorch (avoids ~2.5 GB of unused CUDA packages)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing Ultralytics (YOLO26), headless OpenCV, and NNCF (INT8 export)"
pip install ultralytics opencv-python-headless nncf

echo "Exporting INT8 YOLO26 models for 640x360 and 1280x768 streams to OpenVINO IR"
python ./src/download_yolo26.py --int8 --imgsz 640x360 640x480 640x640 720p 1280x768

echo ""
echo "Intel GPU (Core Ultra iGPU) inference note:"
echo "  Running with --gpu (OpenVINO 'intel:gpu') needs the Intel GPU compute"
echo "  runtime + Level-Zero on the host. On Ubuntu, install with:"
echo "    sudo apt-get update && sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero"
echo "  Verify the GPU is visible to OpenVINO:"
echo "    python -c \"import openvino as ov; print(ov.Core().available_devices)\"   # expect 'GPU' in the list"

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
