# yolo26

## Benchmark runner scripts

All runners share one static-shape model, `models/yolo26n_640x640_int8_openvino_model`
(`--imgsz 640x640`, exported via `python src/download_yolo26.py --variant n --imgsz 640x640 --int8`).
Each script keeps `--rtsp-frame-size` equal to its actual stream resolution, so the
pre-inference resize is a no-op and Ultralytics performs exactly one aspect-correct
letterbox to 640×640 per frame.

| Script | Cores | Source res | `--rtsp-frame-size` | Stage-1 resize | Stage-2 (Ultralytics) | Resizes/frame | Wasted work |
|--------|-------|-----------|---------------------|----------------|-----------------------|:-------------:|:-----------:|
| `1-core.1-cam.yolo26n.sh` | 0 | 640×360 (LOW) | 640×360 | no-op | letterbox → 640×640 | 1 | none |
| `1-core.1-cam.1280x720.yolo26n.sh` | 0 | 1280×720 (MED) | 1280×720 | no-op | letterbox → 640×640 | 1 | none |
| `8-core.8-cam.yolo26n.sh` | 0–7 | 640×360 ×8 | 640×360 | no-op | letterbox → 640×640 | 1 | none |
| `8-core.8-cam.1280x720.yolo26n.sh` | 0–7 | 1280×720 ×8 | 1280×720 | no-op | letterbox → 640×640 | 1 | none |
| `recorded.1-core.1-cam.1080x608.yolo26n.sh` | 0 | 1080×608 | 1080×608 | no-op | letterbox → 640×640 | 1 | none |
| `recorded.8-core.8-cam.1080x608.yolo26n.sh` | 0–7 | 1080×608 ×8 | 1080×608 | no-op | letterbox → 640×640 | 1 | none |

**Rule to keep it optimal:** always set `--rtsp-frame-size` to the actual stream
resolution — that keeps Stage-1 a no-op and lets Ultralytics do the single letterbox.
Annotation and dashboard compositing are skipped entirely when run headless (no viewer),
so they add no overhead to the throughput measurement.

## Results summary

Measured on Intel Xeon **4514Y** (Turbo ON, hyperthreading on) with
OpenVINO **2026.3.0**, model `yolo26n_640x640_int8_openvino_model` (INT8),
1 thread/model, 8 FPS target per camera. Each camera is pinned to one
virtual/SMT core (`*`).

| Test | Stream type | Virtual cores | Cameras | Source res | FPS in | FPS out / cam | Aggregate FPS out | Latency | CPU utilization |
|------|-------------|:-------------:|:-------:|-----------|:------:|:-------------:|:-----------------:|:-------:|:---------------:|
| T1 | Live | 1 | 1 | 640×360 | 8.0 | **7.5** | **7.5** | **43.0 ms** | **43%** |
| T2 | Live | 1 | 1 | 1280×720 | 8.0 | **7.8** | **7.8** | **45.6 ms** | **46%** |
| T3 | Live | 8 | 8 | 640×360 | 8.0 | **7.9** | **63.2** | **38.6 ms** | **84%** |
| T4 | Live | 8 | 8 | 1280×720 | 8.0 | **7.6** | **60.8** | **39.8 ms** | **94%** |
| T5 | Recorded | 1 | 1 | 1080×608 | 8.0 | **6.6** | **6.6** | **45.0 ms** | **48%** |
| T6 | Recorded | 8 | 8 | 1080×608 | 8.0 | **7.3** | **58.4** | **58.5 ms** | **92%** |

**Highlights**

- **FPS throughput:** single-core live streams hold **7.5–7.8 FPS out** against
  the 8 FPS target; scaling to 8 cores/8 cameras delivers **~60–63 aggregate FPS**
  (T3: 63.2, T4: 60.8) while keeping **per-camera FPS at 7.6–7.9**.
- **Latency:** stays low across the board — **38.6–45.6 ms** for live streams,
  with the heaviest recorded 8-camera case (T6) at **58.5 ms**.
- **Virtual cores:** each camera maps to exactly **one virtual/SMT core**;
  linear fan-out from **1 → 8 cores** adds cameras without collapsing per-stream FPS.
- **CPU utilization:** single-core tests run at **43–48%**, leaving headroom;
  8-core tests push to **84–94%**, showing near-full use of the assigned cores at
  higher resolutions (T4 1280×720 at 94%, T6 1080×608 at 92%).

