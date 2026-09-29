#!/bin/bash
# Setup script: OpenVINO 2026.4 runtime environment for YOLO26 benchmarking on
# Intel Nova Lake. Same as setup.openvino_2026.4.sh, but caps oneDNN to the
# AVX-512 VNNI ISA during the INT8 export so pre-production Nova Lake silicon
# (which advertises AVX-512 FP16/BF16 kernels it doesn't fully implement) does
# not segfault during NNCF calibration. Creates a dedicated virtual environment
# and installs OpenVINO + Ultralytics (for YOLO26 export) + headless OpenCV.

set -e  # Exit on any error

rm -rf models
rm -rf venv_openvino_2026.4.nova-lake

VENV_DIR="venv_openvino_2026.4.nova-lake"
# torch/NNCF (TorchScript) don't support Python 3.14+ and segfault during INT8
# calibration, so pick a supported interpreter (3.10-3.13) if available.
PYTHON_CMD=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    # A pyenv shim can resolve but still fail to run (version not set global),
    # so tolerate a failing probe and just move on to the next candidate.
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || continue
    case "$ver" in
        # Store the resolved path so the later `-x` check works ($candidate
        # is a PATH name; `-x` only tests the current directory otherwise).
        3.10|3.11|3.12|3.13) PYTHON_CMD="$(command -v "$candidate")"; break ;;
    esac
done

# pyenv often has a supported interpreter installed but not active. Point at the
# real binary under $(pyenv root)/versions to bypass the shim entirely.
if [ -z "$PYTHON_CMD" ] && command -v pyenv >/dev/null 2>&1; then
    pyroot="$(pyenv root 2>/dev/null)"
    for v in $(pyenv versions --bare --skip-aliases 2>/dev/null | grep -v '/envs/' | sort -rV); do
        case "$v" in
            3.10.*|3.11.*|3.12.*|3.13.*)
                if [ -x "$pyroot/versions/$v/bin/python" ]; then
                    PYTHON_CMD="$pyroot/versions/$v/bin/python"
                    break
                fi ;;
        esac
    done
fi

# No system interpreter in the supported range: provision one with uv, which
# downloads a standalone CPython without touching the system Python.
if [ -z "$PYTHON_CMD" ]; then
    # Make uv visible whether it was pre-installed or lands in ~/.local/bin.
    export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    if ! command -v uv >/dev/null 2>&1; then
        echo "No supported system Python (3.10-3.13) found; bootstrapping uv to" >&2
        echo "provision a standalone Python 3.12 (torch/NNCF INT8 segfaults on 3.14)." >&2
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
        [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    fi
    # Resolve uv even if PATH still hasn't picked it up in this shell.
    UV_BIN="$(command -v uv || true)"
    [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ] && UV_BIN="$HOME/.local/bin/uv"
    if [ -n "$UV_BIN" ]; then
        echo "Provisioning standalone Python 3.12 via uv"
        "$UV_BIN" python install 3.12
        PYTHON_CMD="$("$UV_BIN" python find 3.12 2>/dev/null)"
    fi
fi

if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
    echo "ERROR: No supported Python (3.10-3.13) found and uv bootstrap failed." >&2
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

echo "Installing OpenVINO 2026.4"
pip install openvino==2026.4.0

echo "Installing CPU-only PyTorch (avoids ~2.5 GB of unused CUDA packages)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "Installing Ultralytics (YOLO26), headless OpenCV, and NNCF (INT8 export)"
pip install ultralytics opencv-python-headless nncf

# Nova Lake workaround: cap oneDNN (used by both torch and OpenVINO's CPU
# plugin) to AVX2 for the one-time INT8 calibration. Pre-production silicon
# (reports as "Genuine Intel 0000") advertises AVX-512 kernels that segfault
# when oneDNN JITs them during NNCF statistics collection -- capping to
# AVX512_CORE_VNNI was not enough, so avoid AVX-512 entirely here. Calibration
# runs on 4 images, so the slower ISA is irrelevant, and the cap only affects
# the kernels run during calibration -- the exported INT8 IR is unchanged.
# ONEDNN_ and legacy DNNL_ names are both set so either oneDNN build honors it.
echo "Exporting INT8 YOLO26 models (oneDNN capped to AVX2 for Nova Lake)"
ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2 \
    python ./src/download_yolo26.py --int8 --imgsz 640x360 640x480 640x640 720p 1280x768

echo ""
echo "Intel GPU (Core Ultra iGPU) inference note:"
echo "  Running with --gpu (OpenVINO 'intel:gpu') needs the Intel GPU compute"
echo "  runtime + Level-Zero on the host. On Ubuntu, install with:"
echo "    sudo apt-get update && sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero"
echo "  Verify the GPU is visible to OpenVINO:"
echo "    python -c \"import openvino as ov; print(ov.Core().available_devices)\"   # expect 'GPU' in the list"

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
