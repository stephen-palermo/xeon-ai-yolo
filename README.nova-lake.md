# Nova Lake — Edge AI Computer-Vision Benchmark

## Welcome

This repository is a **foundational Edge AI computer-vision benchmark** built to
optimize **Intel architecture**. It runs a real-time YOLO26 object-detection
pipeline over live and recorded RTSP camera streams and measures throughput
(FPS), latency, and CPU/GPU utilization under controlled core and camera counts.

The goal goes beyond a single score. The benchmark is used to **profile Intel
silicon** — CPU vector/AI ISA extensions and the integrated GPU — on a
representative vision workload, and to turn those profiles into **silicon
manufacturing feedback**: identifying where instruction-set accelerators,
memory bandwidth, and the iGPU help (or bottleneck) YOLO inference so that
**future generations of Intel CPUs and GPUs** can be tuned for Edge AI. Running
it on pre-production parts like **Nova Lake** produces early, workload-grounded
data that feeds directly back into architecture and manufacturing decisions.

## Author

**Created by Stephen Palermo** — Tech Lead, Xeon Edge AI.

## Clone the repository

```bash
git clone https://github.com/stephen-palermo/xeon-ai-yolo.git
cd xeon-ai-yolo
```

## Nova Lake CPU accelerators for AI computer vision

Nova Lake exposes a broad set of on-CPU instruction-set accelerators. The table
below lists the ones most relevant to YOLO / computer-vision inference and how
each can help the workload. (Capabilities are read from `lscpu` `Flags`; this
pre-production part reports **no AMX** tile unit, so INT8 acceleration comes from
the AVX-512 VNNI / AVX-VNNI dot-product path rather than AMX.)

| Accelerator (CPU flag) | What it is | How it helps YOLO / vision inference |
|------------------------|------------|--------------------------------------|
| **AVX-512 Foundation** (`avx512f`, `avx512dq`, `avx512bw`, `avx512vl`, `avx512cd`) | 512-bit wide SIMD across int/float, masked and per-lane | Wide vectorization of convolution, GEMM, and element-wise layers — the backbone of FP32/FP16 inference |
| **AVX-512 VNNI** (`avx512_vnni`) | Vector Neural Network Instructions: fused INT8 multiply-accumulate (`VPDPBUSD`) | Accelerates **INT8** convolutions/GEMM — the primary quantized inference path on this part (no AMX) |
| **AVX-VNNI** (`avx_vnni`) | VNNI dot-product on the 256-bit AVX path | INT8 acceleration for code/threads that stay on AVX2-width kernels; broad client compatibility |
| **AVX-512 BF16** (`avx512_bf16`) | Native bfloat16 dot-product | Fast **BF16** float inference at ~half the memory traffic of FP32 with minimal accuracy loss |
| **AVX-512 FP16** (`avx512_fp16`) | Native IEEE half-precision arithmetic | **FP16** inference (the precision this benchmark exports for Nova Lake) with full-rate half-precision math |
| **AVX-512 IFMA** (`avx512ifma`) | 52-bit integer fused multiply-add | Speeds high-precision integer/fixed-point pre/post-processing and some quantization math |
| **AVX-512 VBMI / VBMI2** (`avx512vbmi`, `avx512_vbmi2`) | Byte-level permute / shift-into | Fast tensor layout changes, packing/unpacking, and NHWC↔NCHW shuffles around conv layers |
| **AVX-512 BITALG / VPOPCNTDQ** (`avx512_bitalg`, `avx512_vpopcntdq`) | Bit manipulation and vector popcount | Helps NMS-style masking, sparsity handling, and bit-packed feature ops |
| **GFNI / VAES / VPCLMULQDQ** (`gfni`, `vaes`, `vpclmulqdq`) | Galois-field, vector AES/carryless-multiply | Accelerates encrypted stream ingest and integrity checks so crypto doesn't steal cycles from inference |
| **SHA-NI** (`sha_ni`) | Hardware SHA hashing | Fast frame/model hashing and secure pipeline attestation with negligible CPU cost |
| **MOVDIRI / MOVDIR64B** (`movdiri`, `movdir64b`) | Direct stores / 64-byte atomic write | Efficient doorbell/descriptor writes for feeding the iGPU and other accelerators |
| **Hardware-Feedback / Thread Director** (`hfi`, `hwp`, `hwp_epp`) | OS hints for per-core performance/efficiency | Steers inference threads to the best cores, improving sustained FPS and perf/watt on hybrid topologies |
| **CET** (`user_shstk`, `ibt`), `serialize` | Control-flow integrity, ordering | Hardened, deterministic execution for an always-on edge service — reliability rather than raw speed |

**Integrated GPU (Nova Lake iGPU).** The same FP16 OpenVINO IR runs on the Intel
iGPU via the `intel:gpu` device, letting you compare the CPU AI ISA path against
the GPU on an identical model — the core apples-to-apples measurement this
benchmark is designed for.

> Note on precision: pre-production Nova Lake segfaults during NNCF **INT8**
> calibration, so the Nova Lake setup exports **FP16** IR (which runs on both the
> CPU and the iGPU). INT8 (AVX-512 VNNI) remains the target quantized path once
> calibration is stable on production silicon.

## Setup and run the benchmarks

### 1. Create the virtual environment

Installs OpenVINO 2026.4, CPU PyTorch, Ultralytics, and exports the FP16 YOLO26
models used by every Nova Lake benchmark:

```bash
./setup.openvino_2026.4.nova-lake.sh
source venv_openvino_2026.4.nova-lake/bin/activate
```

### 2. Start a test RTSP stream (two terminals)

```bash
# Terminal 1 — media server
wget https://github.com/bluenviron/mediamtx/releases/download/v1.19.2/mediamtx_v1.19.2_linux_amd64.tar.gz
tar -xzf mediamtx_v1.19.2_linux_amd64.tar.gz
./mediamtx

# Terminal 2 — loop a video file into the server
sudo apt install -y ffmpeg
ffmpeg -re -stream_loop -1 -i stairway.video.mp4 -c copy -f rtsp rtsp://localhost:8554/stream
```

### 3. Run on the CPU

```bash
./1x1x8.nova-lake.yolo26n.sh
```

### 4. Run on the Nova Lake GPU

The GPU run needs the Intel GPU compute runtime + Level-Zero:

```bash
sudo apt-get update && sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
# Verify the GPU is visible to OpenVINO (expect 'GPU' in the list):
python -c "import openvino as ov; print(ov.Core().available_devices)"

./1x1x8.nova-lake-gpu.yolo26n.sh
```

Both runs serve a live dashboard at `http://<host>:8080`. Because CPU and GPU use
the **same FP16 model** at `640x640` over the same stream, their FPS / latency /
utilization numbers are directly comparable.
