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

## How OpenVINO exposes the CPU accelerators

This benchmark pins to **OpenVINO 2026.4** (`openvino==2026.4.0`), installed by
`setup.openvino_2026.4.nova-lake.sh`. OpenVINO is what turns the raw CPU
instruction-set accelerators listed above into real YOLO26 speedups — the
Ultralytics model is exported to an **OpenVINO IR** (`.xml` + `.bin`) and
executed by the OpenVINO CPU plugin, which maps each network layer onto the best
available ISA kernel at runtime.

Technically, the CPU plugin builds on Intel's **oneDNN** library, which ships
multiple hand-tuned kernel implementations (JIT micro-kernels) for every
primitive — convolution, GEMM, pooling, element-wise. At model-load time the
plugin runs **CPU dispatch**: it reads the processor's capability flags (the same
`lscpu` `Flags` surfaced earlier) and selects the widest kernel the silicon
supports. On Nova Lake that means:

- **FP16/FP32 convolution and GEMM** are vectorized through the **AVX-512**
  foundation and **AVX-512 FP16/BF16** units, so the dense math at the heart of
  YOLO26 runs at full SIMD width.
- **INT8** paths (once calibration is stable) dispatch to the **AVX-512 VNNI /
  AVX-VNNI** `VPDPBUSD` dot-product kernels rather than AMX, since this
  pre-production part reports no tile unit.
- OpenVINO's **graph optimizations** — operator fusion (conv + bias + activation),
  constant folding, and memory-layout conversion to the oneDNN-preferred
  blocked/NHWC format — keep those vector units fed and cut memory traffic, which
  is exactly where the byte-permute (**VBMI**) and bit-manipulation accelerators
  help with packing/unpacking around conv layers.

The net effect: the FP16 IR exported here is hardware-agnostic, and the OpenVINO
runtime decides per-layer whether to execute on the CPU's AI ISA kernels or — via
the `intel:gpu` device — on the Nova Lake iGPU, which is what makes the
CPU-vs-GPU comparison in this benchmark an apples-to-apples one.

> **Future updates will be necessary.** Nova Lake is pre-production silicon, and
> OpenVINO 2026.4 predates its official launch, so the CPU plugin currently
> dispatches Nova Lake using the closest known microarchitecture profile rather
> than a Nova Lake-specific one. Once **Intel officially releases Nova Lake** and
> ships **official OpenVINO support** for it, this benchmark should be re-pinned
> to that OpenVINO release to pick up Nova Lake-tuned oneDNN kernels, the correct
> CPU dispatch, and (pending stable calibration) the INT8/VNNI quantized path.

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
chmod +x stream.ffmpeg.sh
./stream.ffmpeg.sh
```

`stream.ffmpeg.sh` installs ffmpeg if needed and loops `stairway.video.mp4` into
`rtsp://localhost:8554/stream`. Override the defaults with
`./stream.ffmpeg.sh <video-file> <rtsp-url>`.

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

### 5. Foundation computer-Vision AI Workload: 8x8x8

The 8x8x8 workload scales the benchmark to **8 concurrent camera streams across 8
CPU cores** (and the iGPU), driving the same FP16 YOLO26 model on all feeds at
once — a foundational, sustained computer-vision AI load rather than a single
stream. As with the `1x1x8` runs, the CPU and GPU use the **same FP16 model** at
`640x640` over identical streams, so their FPS / latency / utilization numbers
are directly comparable.

Start the RTSP stream first (Terminals 1 and 2 from step 2), then run:

```bash
# CPU — 8 streams pinned to cores 0-7
./recorded.8x8x8.nova-lake.yolo26n.sh
```

```bash
# GPU — 8 streams on the Nova Lake iGPU
./recorded.8x8x8.nova-lake-gpu.yolo26n.sh
```

Both serve the live dashboard at `http://<host>:8080`.

### 6. Uncapped throughput: CPU vs GPU

The `1x1x8` scripts throttle input to `--input-fps 8`, so `FPS out` plateaus near
8 on both devices and only latency separates them. To measure true maximum
throughput, drop the input-FPS cap and let each device run as fast as it can.
Start the RTSP stream first (Terminals 1 and 2 from step 2), then run:

```bash
# CPU — uncapped, single stream pinned to core 0
taskset -c 0 python src/yolo-cv-benchmark.py \
        --model models/yolo26n_640x640_openvino_model \
        --rtsp-frame-size 1080x608 \
        --imgsz 640x640 \
        --rtsp-urls "rtsp://localhost:8554/stream" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7
```

```bash
# GPU — uncapped, single stream on the Nova Lake iGPU
python src/yolo-cv-benchmark.py \
        --gpu \
        --model models/yolo26n_640x640_openvino_model \
        --rtsp-frame-size 1080x608 \
        --imgsz 640x640 \
        --rtsp-urls "rtsp://localhost:8554/stream" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7
```

Both use the **same FP16 IR** at `640x640` over the same stream, so the resulting
`FPS out` and `Latency` are directly comparable.

#### Results (single stream, YOLO26n @ 640×640, 30 FPS input)

| Metric | CPU (`intel:cpu`, 1 core) | GPU (`intel:gpu`, GPU.0) |
|---|---|---|
| Compute precision | FP32 | FP16 (native) |
| Inference hardware | 1 CPU core (`taskset -c 0`) | iGPU (+ CPU threads for pre/post) |
| **Steady-state FPS out** | **~22.8** | **~27.0** |
| Latency | steady ~43 ms | ~7–10 ms (min), bursty to ~45 ms |
| CPU utilization | 100% (1 core saturated) | ~80% (pre/post + feeding GPU) |

**Takeaways**

- The iGPU delivers ~18% higher throughput (~27 vs ~22.8 FPS) than a single CPU
  core, while leaving CPU headroom, and its raw inference latency (~7–10 ms) is
  several times lower than the CPU's ~43 ms.
- Both sit below the 30 FPS input: the CPU is compute-bound (1 core at 100%); the
  GPU has CPU headroom, so it's limited by the single-stream pipeline, not the CPU.

> Caveats for a fair read: the CPU run is pinned to **one core** (`taskset -c 0`),
> so this is *1 CPU core* vs *the whole iGPU* — a full-socket CPU run scales much
> higher. Precision also differs (CPU FP32 for stability on this non-AMX silicon
> vs native GPU FP16). For CPU-socket vs iGPU, drop `taskset -c 0`; for scaling,
> use the `8x8x8` scripts.

![Nova Lake CPU vs GPU](docs/nova_lake_cpu_vs_gpu.png)
