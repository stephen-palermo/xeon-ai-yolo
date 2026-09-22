#!/bin/bash
# Setup script: NVIDIA GPU runtime environment for YOLO26 benchmarking.
# Works on Ada Lovelace (e.g. L4, sm_89) through Blackwell (e.g. B200/RTX 50xx,
# sm_100/sm_120). Creates a dedicated virtual environment and installs a
# Blackwell-capable CUDA PyTorch, Ultralytics, and the TensorRT export toolchain
# (onnx/onnxslim/onnxruntime-gpu/tensorrt) up front, so exporting a .engine model
# doesn't trigger Ultralytics AutoUpdate mid-run.

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

# cu128 fat-binary wheels bundle kernels for Ada (sm_89, L4) through Blackwell (sm_100/sm_120).
echo "Installing Blackwell/L4-capable CUDA PyTorch (cu128)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "Installing Ultralytics (YOLO26) and headless OpenCV"
pip install ultralytics opencv-python-headless

echo "Installing the TensorRT export toolchain (ONNX -> TensorRT .engine)"
pip install onnx onnxslim onnxruntime-gpu "tensorrt-cu13>=7.0.0,!=10.2.0"

# Fail fast with an actionable message instead of a cryptic export traceback if
# PyTorch cannot see a GPU (missing/old driver, or container without GPU passthrough).
echo "Verifying PyTorch can see a CUDA GPU"
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
	echo "ERROR: torch.cuda.is_available() is False — no usable CUDA GPU detected." >&2
	echo "  Check the following on this host:" >&2
	echo "    1. 'nvidia-smi' works and shows an R570+ driver (required for cu128)." >&2
	echo "    2. If in a container, it was started with GPU passthrough (docker --gpus all" >&2
	echo "       or --runtime=nvidia, with nvidia-container-toolkit installed)." >&2
	echo "    3. If the driver is older than cu128 needs, either upgrade it or reinstall" >&2
	echo "       torch for your driver's CUDA (e.g. --index-url .../whl/cu126)." >&2
	nvidia-smi || echo "  ('nvidia-smi' not found or failed — driver/GPU not visible.)" >&2
	exit 1
fi

echo "Exporting INT8 YOLO26n engine for a 640x360 stream"
python ./src/download_yolo26_tensorrt.py --int8 --imgsz 640x360

echo ""
echo "NVIDIA GPU inference note:"
echo "  Requires the NVIDIA R570+ driver (minimum for Blackwell; also supports L4)."
echo "  CUDA PyTorch is installed above (cu128). Verify with:"
echo "    nvidia-smi"
echo "    python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
echo "  Note: TensorRT .engine files are architecture-specific and are (re)built"
echo "  here on the target host, so no cross-GPU engine portability is needed."

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
