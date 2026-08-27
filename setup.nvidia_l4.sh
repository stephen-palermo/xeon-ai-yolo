#!/bin/bash
# Setup script: NVIDIA GPU (e.g. L4) runtime environment for YOLO26 benchmarking.
# Creates a dedicated virtual environment and installs Ultralytics + the
# TensorRT export toolchain (onnx/onnxslim/onnxruntime-gpu/tensorrt) up front,
# so exporting a .engine model doesn't trigger Ultralytics AutoUpdate mid-run.

set -e  # Exit on any error

rm -rf models
rm -rf venv_nvidia_l4

VENV_DIR="venv_nvidia_l4"
PYTHON_CMD="python3"

echo "Creating virtual environment: ${VENV_DIR}"
$PYTHON_CMD -m venv "$VENV_DIR"

echo "Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing Ultralytics (YOLO26) and headless OpenCV"
pip install ultralytics opencv-python-headless

echo "Installing the TensorRT export toolchain (ONNX -> TensorRT .engine)"
pip install onnx onnxslim onnxruntime-gpu "tensorrt-cu13>=7.0.0,!=10.2.0"

echo "Exporting INT8 YOLO26n engine for a 640x360 stream"
python ./src/download_yolo26_tensorrt.py --int8 --imgsz 640x360

echo ""
echo "NVIDIA GPU inference note:"
echo "  Requires the NVIDIA driver + a CUDA-enabled PyTorch install visible to"
echo "  this venv. Verify with:"
echo "    nvidia-smi"
echo "    python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
