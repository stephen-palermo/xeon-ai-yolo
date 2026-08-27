#!/bin/bash
# Setup script: OpenVINO 2026.2.1 runtime environment for YOLO26 benchmarking
# Creates a dedicated virtual environment and installs OpenVINO + Ultralytics
# (for YOLO26 export) + headless OpenCV.

set -e  # Exit on any error

rm -rf models
rm -rf venv_openvino_2026.3.0

VENV_DIR="venv_openvino_2026.3.0"
PYTHON_CMD="python3"

echo "Creating virtual environment: ${VENV_DIR}"
$PYTHON_CMD -m venv "$VENV_DIR"

echo "Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing OpenVINO 2026.3.0"
pip install openvino==2026.3.0

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
