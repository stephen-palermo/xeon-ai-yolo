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

