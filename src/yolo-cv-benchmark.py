#!/usr/bin/env python3
"""FPS-focused YOLO/OpenVINO benchmark with a lightweight web preview.

Design goals (differ from ``preview_openvino_web.py``):

  1. Same core functionality: read RTSP stream(s), run Ultralytics YOLO through
     OpenVINO on CPU, and show the annotated detections in a browser.
  2. The same header/system information is shown in the browser, but the static
     parts are collected once at startup (before the FPS benchmark runs).
  3. Header info (memory, per-core frequencies, average FPS) refreshes ONLY when
     the user manually reloads the page. There is no auto-refresh and no
     background polling thread.
  4. FPS throughput is prioritized above everything else. The ``--amx`` toggle is
     preserved so AMX ON vs OFF can be benchmarked separately.
  5. Self-contained: uses only ``yolo`` (new,
      loop-free) and
     ``amx_detect`` (the AMX authority). No legacy benchmark/preview modules are
     imported, so none of their runtime logic can add CPU overhead.
  6. No unnecessary runtime work: there is no server-side dashboard compositor,
     and frames are annotated/JPEG-encoded only while a browser is actually
     watching that camera. With no viewer, each worker spends CPU purely on
     inference (the throughput measurement).
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from http import server
from html import escape
from pathlib import Path

import yolo_cv_benchmark_browser_support as support
from amx_detect import AmxStatus, detect_amx


MAX_AI_CORES = 32
DEFAULT_BENCHMARK_MODELS = ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x")


def _prefetch_openvino_models(models_dir: str, imgsz: int | tuple[int, int]) -> None:
    """Ensure the standard benchmark model set exists under ``models_dir``."""
    root = Path(models_dir)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring benchmark models in {root.resolve()} ...")

    for base_name in DEFAULT_BENCHMARK_MODELS:
        target = root / f"{base_name}_int8_openvino_model"
        print(f"Preparing {target} ...")
        support.resolve_model_source(str(target), imgsz)


class FrameStore:
    """Thread-safe storage for the latest encoded preview frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._jpeg: bytes | None = None

    def update(self, jpeg: bytes) -> None:
        with self._condition:
            self._sequence += 1
            self._jpeg = jpeg
            self._condition.notify_all()

    def wait_for_next(self, last_sequence: int, timeout: float = 1.0):
        with self._condition:
            if self._sequence <= last_sequence:
                self._condition.wait(timeout=timeout)
            return self._sequence, self._jpeg


class StreamState:
    """Per-camera shared state.

    ``fps`` is updated every processed frame (a cheap float write). The latest
    annotated frame is stored for the composite dashboard, which is built only
    while a browser is watching.
    """

    def __init__(self, stream_id: int, source: str) -> None:
        self.stream_id = stream_id
        self.source = source
        self._lock = threading.Lock()
        self._fps = 0.0
        self._latency_ms = 0.0
        self._annotated = None

    def set_fps(self, value: float) -> None:
        with self._lock:
            self._fps = value

    def get_fps(self) -> float:
        with self._lock:
            return self._fps

    def set_latency(self, value: float) -> None:
        with self._lock:
            self._latency_ms = value

    def get_latency(self) -> float:
        with self._lock:
            return self._latency_ms

    def set_annotated(self, frame) -> None:
        with self._lock:
            self._annotated = frame

    def get_annotated(self):
        with self._lock:
            return self._annotated


class DashboardState:
    """Shared state for the single composite MJPEG stream.

    Serving one multiplexed stream (instead of one MJPEG connection per camera)
    avoids the browser's per-host connection limit (~6), which otherwise drops
    higher camera indices on large fan-in tests. Compositing and per-frame annotation are gated on
    ``viewers`` so they run only while the dashboard is actually being viewed.
    """

    def __init__(self) -> None:
        self.frame_store = FrameStore()
        self._lock = threading.Lock()
        self._viewers = 0

    def add_viewer(self) -> None:
        with self._lock:
            self._viewers += 1

    def remove_viewer(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    def has_viewers(self) -> bool:
        with self._lock:
            return self._viewers > 0


def _average_fps(streams: list[StreamState]) -> float:
    values = [stream.get_fps() for stream in streams]
    return sum(values) / len(values) if values else 0.0


def _average_latency(streams: list[StreamState]) -> float:
    values = [stream.get_latency() for stream in streams]
    return sum(values) / len(values) if values else 0.0


def _pin_thread_affinity(cores: list[int] | None) -> None:
    """Pin the calling thread (and threads it spawns) to ``cores`` on Linux.

    Confining each workflow to its own core subset gives many concurrent YOLO
    pipelines realistic, non-oversubscribed isolation. OpenVINO/TBB worker
    threads created afterward inherit this mask. No-op on non-Linux hosts.
    """
    if not cores:
        return
    try:
        os.sched_setaffinity(0, set(cores))
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------------------
# Inference worker (the FPS workload)
# ---------------------------------------------------------------------------
class _LatestFrameReader:
    """Background RTSP reader that always holds only the newest decoded frame.

    Inference is slower than the camera frame rate, so reading inline lets the
    FFmpeg/socket buffer fill during each inference call and then return
    ever-older frames -- the preview drifts further behind real time over time.
    Draining continuously on a separate thread and keeping just the latest frame
    keeps the preview close to real time. Reconnects when the stream stalls.
    """

    def __init__(self, source, open_fn, stop_event, stream_id, stall_timeout=5.0):
        self._source = source
        self._open_fn = open_fn
        self._stop_event = stop_event
        self._stream_id = stream_id
        self._stall_timeout = stall_timeout
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._local_stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._local_stop.set()

    def _stopped(self) -> bool:
        return self._local_stop.is_set() or self._stop_event.is_set()

    def _run(self) -> None:
        capture = self._open_fn(self._source)
        last_ok = time.perf_counter()
        while not self._stopped():
            ok, frame = capture.read()
            if not ok or frame is None:
                if time.perf_counter() - last_ok > self._stall_timeout:
                    print(
                        f"Camera {self._stream_id}: stream stalled, reconnecting...",
                        flush=True,
                    )
                    capture.release()
                    time.sleep(0.5)
                    capture = self._open_fn(self._source)
                    last_ok = time.perf_counter()
                else:
                    time.sleep(0.01)
                continue
            last_ok = time.perf_counter()
            with self._lock:
                self._frame = frame
                self._seq += 1
        capture.release()

    def read_latest(self):
        """Return (sequence, frame) for the most recent decoded frame."""
        with self._lock:
            return self._seq, self._frame


def _run_inference_loop(
    args: argparse.Namespace,
    stream: StreamState,
    dashboard: DashboardState,
    model_source: str,
    cores: list[int] | None,
    stop_event: threading.Event,
) -> None:
    """Read RTSP frames and run YOLO; annotate only while the dashboard is watched."""
    import cv2  # noqa: PLC0415

    # Pin this workflow to its allotted cores before the model (and its OpenVINO
    # worker threads) are created, so the whole pipeline stays on those cores.
    _pin_thread_affinity(cores)

    # Safety net for deployments that may carry older helper code: resolve the
    # model path again in-worker so Ultralytics always receives a concrete
    # OpenVINO artifact path when possible.
    resolved_model_source = support.resolve_model_source(str(model_source), args.imgsz)
    if resolved_model_source != str(model_source):
        print(
            f"Camera {stream.stream_id}: resolved model source "
            f"{model_source} -> {resolved_model_source}",
            flush=True,
        )
    model = support.build_model(resolved_model_source)

    # Decode RTSP on a background thread that keeps only the newest frame, so a
    # slow inference call can't let the buffer accumulate and lag the preview.
    reader = _LatestFrameReader(
        stream.source, support.open_rtsp_capture, stop_event, stream.stream_id
    ).start()

    class_ids = support.parse_classes(args.classes)
    rtsp_frame_size = support.parse_rtsp_frame_size(args.rtsp_frame_size)
    infer_height, infer_width = support.imgsz_dimensions(args.imgsz)
    amx_on = getattr(args, "amx_effective", args.amx == "on")
    model_label = os.path.splitext(os.path.basename(str(args.model)))[0]

    # Optional input-FPS throttle: drop frames so inference runs at most this
    # many times per second, even when the camera delivers more (e.g. 30 FPS).
    target_interval = 1.0 / args.input_fps if args.input_fps > 0 else 0.0
    last_processed = 0.0
    last_seq = 0

    frame_count = 0
    started_at = time.perf_counter()

    try:
        while not stop_event.is_set():
            seq, frame = reader.read_latest()
            # Wait for a genuinely new frame; never re-run inference on a stale
            # one, which would add latency without adding information.
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue

            # Throttle to --input-fps, but always take the most recent frame so
            # the preview tracks real time instead of a growing backlog.
            if target_interval > 0.0:
                now = time.perf_counter()
                if now - last_processed < target_interval:
                    time.sleep(0.002)
                    continue
                last_processed = now

            last_seq = seq

            if rtsp_frame_size is None:
                if frame.shape[0] != infer_height or frame.shape[1] != infer_width:
                    frame = cv2.resize(
                        frame,
                        (infer_width, infer_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
            else:
                target_w, target_h = rtsp_frame_size
                if frame.shape[1] != target_w or frame.shape[0] != target_h:
                    frame = cv2.resize(
                        frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )

            infer_start = time.perf_counter()
            results = model.predict(
                [frame],
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                classes=class_ids,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
            )
            latency_ms = (time.perf_counter() - infer_start) * 1000.0

            frame_count += 1
            elapsed = time.perf_counter() - started_at
            preview_fps = frame_count / elapsed if elapsed > 0 else 0.0
            stream.set_fps(preview_fps)
            stream.set_latency(latency_ms)

            # Prioritize throughput: skip annotation entirely unless the
            # composite dashboard is currently being viewed.
            if not dashboard.has_viewers():
                continue

            annotated = results[0].plot() if results else frame
            overlay_text = (
                f"cam {stream.stream_id}  E2E FPS {preview_fps:.1f}  "
                f"{latency_ms:.1f} ms"
            )
            amx_text = f"AMX {'ON' if amx_on else 'OFF'}  {model_label}"
            amx_color = (0, 150, 0) if amx_on else (0, 0, 200)

            frame_height = annotated.shape[0]
            box_top = frame_height - 92
            box_bottom = frame_height - 8
            cv2.rectangle(
                annotated,
                (8, box_top),
                (annotated.shape[1] - 8, box_bottom),
                (235, 235, 235),
                -1,
            )
            cv2.putText(
                annotated,
                overlay_text,
                (16, box_top + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                amx_text,
                (16, box_top + 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                amx_color,
                3,
                cv2.LINE_AA,
            )

            stream.set_annotated(annotated)
    finally:
        reader.stop()


def _run_dashboard_loop(
    args: argparse.Namespace,
    streams: list[StreamState],
    dashboard: DashboardState,
    stop_event: threading.Event,
) -> None:
    """Compose the latest per-camera frames into one MJPEG stream.

    Runs only while the dashboard has a viewer; otherwise it idles so no CPU is
    spent compositing when nobody is watching.
    """
    import math  # noqa: PLC0415

    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    cols = support.grid_columns(len(streams))
    rows = math.ceil(len(streams) / cols)
    # Keep the composite tiles modest so the (unpinned) dashboard thread's
    # resize + JPEG-encode work does not steal CPU from the pinned per-camera
    # inference pipelines. Raising this was found to cut throughput while the
    # dashboard is being viewed. Compositing only runs when a viewer is present.
    tile_size = max(160, min(support.imgsz_max_dimension(args.imgsz), 480))
    title_bar = 34

    while not stop_event.is_set():
        if not dashboard.has_viewers():
            time.sleep(0.2)
            continue

        tiles = []
        for stream in streams:
            frame = stream.get_annotated()
            if frame is None:
                frame = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"Camera {stream.stream_id}: waiting for frames",
                    (20, max(40, tile_size // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            frame = cv2.resize(
                frame, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
            )
            tile = np.zeros((tile_size + title_bar, tile_size, 3), dtype=np.uint8)
            tile[:title_bar, :, :] = (24, 33, 45)
            tile[title_bar:, :, :] = frame
            cv2.putText(
                tile,
                f"Camera {stream.stream_id}",
                (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 239, 226),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        blank = np.zeros_like(tiles[0])
        while len(tiles) < rows * cols:
            tiles.append(blank.copy())

        row_images = []
        for row_index in range(rows):
            start = row_index * cols
            row_images.append(np.hstack(tiles[start:start + cols]))
        dashboard_img = np.vstack(row_images)

        ok, encoded = cv2.imencode(
            ".jpg",
            dashboard_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        if ok:
            dashboard.frame_store.update(encoded.tobytes())

        time.sleep(0.03)


def _cpu_model_number(cpu_model: str) -> str:
    """Extract just the CPU SKU/model number from a full CPU brand string."""
    match = re.search(r"\b(?:i[3579]|Ultra\s*[3579])[- ]?\d{3,5}[A-Z]*\b", cpu_model)
    if match:
        return match.group(0)
    match = re.search(r"\b\d{4,5}[A-Z+]*\b", cpu_model)
    if match:
        return match.group(0)
    return cpu_model


def _run_metrics_loop(
    streams: list[StreamState],
    stop_event: threading.Event,
    interval: float,
    static_info: dict,
) -> None:
    """Print the grounded system/video/AI info block plus live averages.

    This lets the benchmark be measured with no browser attached, so the numbers
    reflect pure inference throughput rather than display overhead. CPU
    utilization is measured over each interval window from ``/proc/stat``.
    """
    core_ids = static_info.get("assigned_core_ids") or []

    cpu_number = _cpu_model_number(str(static_info.get("cpu_model", "unknown")))
    model_display = os.path.splitext(static_info.get("model_name", ""))[0]
    model_precision = str(static_info.get("inference_precision", "unknown"))
    system_line = (
        f"[System INFO]  CPU: {cpu_number} | MODEL: {model_display} / {model_precision}"
    )
    video_line = (
        f"[Video INFO] RTSP input {static_info.get('rtsp_frame_size', 'unknown')} | "
        f"Model input {static_info.get('model_input_size', 'unknown')}"
    )
    if static_info.get("openvino_enabled"):
        openvino_status = (
            f"{static_info.get('openvino_version', 'unknown')} / "
            f"{static_info.get('model_precision', 'unknown')}"
        )
    else:
        openvino_status = "OFF"
    ai_line = (
        f"[AI INFO] OpenVino: {openvino_status} | "
        f"threads/model: {static_info.get('threads_per_model')} | "
        f"core assignments: {static_info.get('core_assignments')}"
    )

    throttle = static_info.get("input_fps_throttle", 0.0) or 0.0
    camera_input_fps = static_info.get("camera_input_fps") or []
    per_cam_input = [throttle if throttle > 0 else fps for fps in camera_input_fps]
    avg_input_fps = sum(per_cam_input) / len(per_cam_input) if per_cam_input else 0.0

    prev_jiffies = support.read_cpu_stat_jiffies()
    while not stop_event.wait(interval):
        curr_jiffies = support.read_cpu_stat_jiffies()
        utilization = support.cpu_utilization_from_jiffies(
            prev_jiffies, curr_jiffies, core_ids
        )
        prev_jiffies = curr_jiffies
        used = [utilization[cid] for cid in core_ids if cid in utilization]
        avg_util = sum(used) / len(used) if used else 0.0

        perf_line = (
            f"[AI Perf INFO] Averages: RTSP FPS in {avg_input_fps:.1f} | "
            f"FPS out {_average_fps(streams):.1f} | "
            f"Latency {_average_latency(streams):.1f}ms | "
            f"core utilization {avg_util:.0f}%"
        )
        print("[-----------------------------]", flush=True)
        print(system_line, flush=True)
        print(video_line, flush=True)
        print(ai_line, flush=True)
        print(perf_line, flush=True)
        print("[-----------------------------]", flush=True)


# ---------------------------------------------------------------------------
# HTTP serving
# ---------------------------------------------------------------------------
def _serve_mjpeg(handler, viewable: DashboardState, stop_event: threading.Event) -> None:
    """Stream the composite MJPEG output, tracking viewer count for CPU gating."""
    handler.send_response(200)
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Connection", "close")
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.end_headers()

    viewable.add_viewer()
    last_sequence = 0
    try:
        while not stop_event.is_set():
            last_sequence, jpeg = viewable.frame_store.wait_for_next(last_sequence)
            if not jpeg:
                continue
            try:
                handler.wfile.write(b"--frame\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                handler.wfile.write(jpeg)
                handler.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
    finally:
        viewable.remove_viewer()


def _render_index(
    streams: list[StreamState],
    title: str,
    static_info: dict,
    amx_status: AmxStatus,
    amx_effective: bool,
) -> bytes:
    """Build the dashboard HTML.

    Static fields come from the startup snapshot; only the cheap dynamic fields
    (memory, per-core frequency, average FPS) are re-read here, which happens
    solely when the user manually reloads the page.
    """
    if static_info["openvino_enabled"]:
        openvino_status = (
            f"v{escape(static_info['openvino_version'])}, "
            f"{escape(static_info['model_precision'])}"
        )
    else:
        openvino_status = "OFF"

    model_display = os.path.splitext(static_info["model_name"])[0]
    device_str = str(static_info.get("device", "cpu"))
    on_gpu = device_str.lower().startswith(("gpu", "intel:gpu"))
    device_label = "Intel GPU" if on_gpu else "CPU"
    header_line = (
        f"CPU: {escape(static_info['cpu_model'])} &nbsp;&#124;&nbsp; "
        f"Device: {escape(device_label)} &nbsp;&#124;&nbsp; "
        f"AMX: {'ON' if amx_effective else 'OFF'} &nbsp;&#124;&nbsp; "
        f"Model: {escape(model_display)} / {escape(str(static_info.get('inference_precision', 'unknown')))}"
    )
    header_line_io = (
        f"Model input: {escape(str(static_info.get('model_input_size', 'unknown')))} &nbsp;&#124;&nbsp; "
        f"RTSP input: {escape(str(static_info.get('rtsp_frame_size', 'unknown')))}"
    )
    header_line2 = (
        f"OpenVINO: {openvino_status} &nbsp;&#124;&nbsp; "
        f"threads/model: {static_info['threads_per_model']} &nbsp;&#124;&nbsp; "
        f"core assignments: {escape(str(static_info['core_assignments']))}"
    )
    # Per-camera effective input rate only. When throttling is enabled, this is
    # --input-fps; otherwise it is the measured camera input rate.
    # One merged line of live averages across all cameras. Core utilization is
    # sampled over a short window here, so it runs only on manual page reload.
    throttle = static_info.get("input_fps_throttle", 0.0) or 0.0
    camera_input_fps = static_info.get("camera_input_fps") or []
    per_cam_input = [throttle if throttle > 0 else fps for fps in camera_input_fps]
    avg_input_fps = sum(per_cam_input) / len(per_cam_input) if per_cam_input else 0.0

    assigned_core_ids = static_info.get("assigned_core_ids") or []
    prev_jiffies = support.read_cpu_stat_jiffies()
    time.sleep(0.2)
    utilization = support.cpu_utilization_from_jiffies(
        prev_jiffies, support.read_cpu_stat_jiffies(), assigned_core_ids
    )
    avg_core_util = sum(utilization.values()) / len(utilization) if utilization else 0.0

    header_line_averages = (
        f"Averages &#8212; RTSP FPS in: {avg_input_fps:.1f} &nbsp;&#124;&nbsp; "
        f"FPS out: {_average_fps(streams):.1f} &nbsp;&#124;&nbsp; "
        f"Latency: {_average_latency(streams):.1f}ms &nbsp;&#124;&nbsp; "
        f"Core util: {avg_core_util:.0f}%"
    )
    show_cpu_info = static_info.get("show_cpu_info", False)
    header_line_accel = ""
    header_line_freq = ""

    if show_cpu_info:
        accelerators = static_info.get("ai_accelerators") or []
        amx_flags = [
            name
            for name, present in (
                ("amx_tile", amx_status.amx_tile),
                ("amx_bf16", amx_status.amx_bf16),
                ("amx_int8", amx_status.amx_int8),
            )
            if present
        ]
        all_accelerators = amx_flags + accelerators
        accel_text = ", ".join(all_accelerators) if all_accelerators else "none detected"
        header_line_accel = f"CPU AI accelerators: {escape(accel_text)}"

        core_frequencies = support.read_cpu_core_frequencies()
        if core_frequencies:
            # Keep the frequency header readable on many-core systems.
            freq_chunks = [
                core_frequencies[index:index + 8]
                for index in range(0, len(core_frequencies), 8)
            ]
            header_line_freq = "<br>".join(
                "CPU Freq: " + ", ".join(f"C{cid}={mhz:.0f}MHz" for cid, mhz in chunk)
                for chunk in freq_chunks
            )
        else:
            header_line_freq = "CPU Freq: unavailable"

    optional_cpu_header_lines = ""
    if show_cpu_info:
        optional_cpu_header_lines = (
            f"    <h1>{header_line_accel}</h1>\n"
            f"    <h1>{header_line_freq}</h1>\n"
        )

    # The composite packs all cameras into one image, so with more cameras each
    # tile shrinks. Widen the browser window past 4 cameras so they stay legible.
    frame_max_width = 1600 if len(streams) > 4 else 720

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; font-family: "Courier New", Courier, monospace; background: radial-gradient(circle at top, #243b55, #10151c 65%); color: #f5efe2; }}
    main {{ max-width: 1920px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.5rem; letter-spacing: 0.02em; }}
    p {{ margin: 0 0 18px; color: #d9d0c0; }}
    table.cores {{ width: 100%; border-collapse: collapse; margin: 0 0 18px; table-layout: fixed; }}
    table.cores td {{ border: 1px solid rgba(255,255,255,0.16); padding: 10px 12px; text-align: center; font-size: 1.35rem; color: #f5efe2; background: rgba(10,12,16,0.45); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .frame {{ max-width: {frame_max_width}px; margin: 0 auto; border: 1px solid rgba(255,255,255,0.16); border-radius: 18px; overflow: hidden; background: rgba(10,12,16,0.55); box-shadow: 0 24px 80px rgba(0,0,0,0.28); }}
    .frame img {{ display: block; width: 100%; height: auto; image-rendering: auto; background: #0c1117; }}
  </style>
</head>
<body>
  <main>
    <h1>{header_line}</h1>
    <h1>{header_line_io}</h1>
    <h1>{header_line2}</h1>
{optional_cpu_header_lines}    <h1>{header_line_averages}</h1>
    <p>Reload this page to refresh system stats. Each camera appears as its frames become ready.</p>
    <div class="frame"><img src="/dashboard.mjpg" alt="Composite detection dashboard"></div>
  </main>
</body>
</html>""".encode("utf-8")
    return body


def _make_handler(
    streams: list[StreamState],
    dashboard: DashboardState,
    stop_event: threading.Event,
    title: str,
    static_info: dict,
    amx_status: AmxStatus,
    amx_effective: bool,
):
    """Create the HTTP handler bound to the current benchmark state."""

    class BenchmarkHandler(server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                body = _render_index(
                    streams, title, static_info, amx_status, amx_effective
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok\n")
                return

            if self.path == "/metrics":
                import json  # noqa: PLC0415

                payload = {
                    "amx_requested": static_info["amx_requested"],
                    "amx_effective": amx_effective,
                    "device": static_info.get("device", "cpu"),
                    "model": static_info["model_name"],
                    "precision": static_info["model_precision"],
                    "num_cameras": len(streams),
                    "aggregate_fps": round(sum(s.get_fps() for s in streams), 2),
                    "average_fps": round(_average_fps(streams), 2),
                    "streams": [
                        {"id": s.stream_id, "fps": round(s.get_fps(), 2)}
                        for s in streams
                    ],
                }
                body = json.dumps(payload, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/dashboard.mjpg":
                _serve_mjpeg(self, dashboard, stop_event)
                return

            self.send_error(404)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return BenchmarkHandler


# ---------------------------------------------------------------------------
# CPU AI-acceleration reporting
# ---------------------------------------------------------------------------
# AI-relevant x86 ISA extensions beyond AMX. These help deep-learning inference
# on both Xeon (server) and Core/Ultra (client) CPUs. Ordered newest/most
# capable first. amx_* flags are reported separately by amx_detect.
_AI_ACCEL_FLAGS: tuple[tuple[str, str], ...] = (
    ("avx512_fp16", "AVX-512 FP16 — native half-precision vector math"),
    ("avx512_bf16", "AVX-512 BF16 — bfloat16 vector math (AMX-OFF fallback path)"),
    ("avx512_vnni", "AVX-512 VNNI (Intel DL Boost) — fused INT8 dot-product"),
    ("avx_vnni_int8", "AVX-VNNI-INT8 — INT8 dot-product on client (P/E-core) CPUs"),
    ("avx_vnni_int16", "AVX-VNNI-INT16 — INT16 dot-product on client CPUs"),
    ("avx_vnni", "AVX-VNNI — VNNI without AVX-512 (Alder Lake+ client CPUs)"),
    ("avx512f", "AVX-512 Foundation — 512-bit vector math"),
    ("avx2", "AVX2 — 256-bit vector math (broad client/server baseline)"),
    ("f16c", "F16C — half-precision load/store conversion"),
)


def _read_cpu_flags() -> set[str]:
    """Return the set of CPU feature flags from Linux /proc/cpuinfo (or empty)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags") and ":" in line:
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def _active_ai_accelerators() -> list[str]:
    """Return the active (present) non-AMX AI accelerator flag names on this CPU."""
    flags = _read_cpu_flags()
    return [flag for flag, _desc in _AI_ACCEL_FLAGS if flag in flags]


# Low-precision compute paths, mapped to the CPU flags that accelerate them.
# Ordered so the tags read INT8, FP16, BF16 regardless of CPU type.
_PRECISION_CAP_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("INT8", ("amx_int8", "avx512_vnni", "avx_vnni_int8", "avx_vnni")),
    ("FP16", ("avx512_fp16", "f16c")),
    ("BF16", ("amx_bf16", "avx512_bf16")),
)


def _active_precision_tags() -> list[str]:
    """Return accelerated CPU precision tags (INT8/FP16/BF16) present on this CPU."""
    flags = _read_cpu_flags()
    return [tag for tag, needed in _PRECISION_CAP_FLAGS if any(f in flags for f in needed)]


def _report_cpu_ai_accelerators() -> None:
    """Print a single-line summary of CPU AI accelerators to the terminal."""
    amx = detect_amx()
    amx_bits = [name for name, present in (
        ("TILE", amx.amx_tile), ("BF16", amx.amx_bf16), ("INT8", amx.amx_int8)
    ) if present]
    amx_text = f"AMX({'+'.join(amx_bits) if amx_bits else 'none'}, {'usable' if amx.usable else 'not usable'})"
    others = _active_ai_accelerators()
    others_text = ", ".join(others) if others else "none"
    print(f"CPU AI accelerators: {amx_text} | {others_text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="models/yolo26n_640x384_int8_openvino_model/",
        help="YOLO weights or exported OpenVINO model. Default is the INT8 IR so Xeon (AMX-INT8) and client CPUs (AVX-VNNI-INT8) benchmark on a consistent accelerated path.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory used to store downloaded/exported benchmark models.",
    )
    parser.add_argument(
        "--download-all-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download/export YOLO26 n/s/m/l/x OpenVINO models into --models-dir before running. Disabled by default so the benchmark only prepares the requested model.",
    )
    parser.add_argument(
        "--show",
        nargs="*",
        default=[],
        choices=("cpu_info",),
        help="Optional dashboard sections to show. Use '--show cpu_info' to show CPU accelerator/utilization/frequency lines.",
    )
    parser.add_argument("--rtsp-url", default="", help="Primary RTSP stream. Optional when --rtsp-urls is set.")
    parser.add_argument(
        "--rtsp-urls",
        default="",
        help=f"Comma-separated RTSP stream list (max {support.MAX_RTSP_STREAMS}).",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=1,
        help=f"Replicate --rtsp-url this many times (max {support.MAX_RTSP_STREAMS}).",
    )
    parser.add_argument("--rtsp-frame-size", default="640x360", help="Optional pre-inference frame size WIDTHxHEIGHT (e.g. 640x360).")
    parser.add_argument(
        "--imgsz",
        default="640x384",
        help="Inference image size as N or WIDTHxHEIGHT. Must be a multiple of 32; default 640x384 fits a 640x360 stream with minimal padding.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run inference on (e.g. cpu, gpu, 0). Default is 'cpu'.",
    )
    parser.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run inference on the Intel Core Ultra integrated GPU via OpenVINO "
             "('intel:gpu') instead of the CPU. Overrides --device when set.",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=0.0,
        help="Throttle processing to this many frames per second per camera by dropping "
             "excess frames (e.g. 5 to process 5 FPS from a 30 FPS camera). 0 = no limit.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--amx", choices=("on", "off"), default="on", help="Allow or disable AMX via oneDNN ISA cap.")
    parser.add_argument("--classes", default="0", help="Comma-separated YOLO class IDs. Default '0' (people); empty shows all.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for the browser stream (1-100).")
    parser.add_argument(
        "--threads-per-model",
        type=int,
        default=0,
        help="Inference threads per camera workflow. 0 = auto (available cores / #cameras). "
             "Caps oneDNN/OpenVINO threads and pins each workflow to its own cores to avoid "
             "oversubscription across the concurrent pipelines.",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=5.0,
        help="Seconds between headless per-stream/aggregate FPS prints (regardless of AMX "
             "state). 0 disables (use the /metrics endpoint instead).",
    )
    args = parser.parse_args(argv)

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        parser.error("--jpeg-quality must be between 1 and 100.")
    if args.num_streams < 1 or args.num_streams > support.MAX_RTSP_STREAMS:
        parser.error(f"--num-streams must be between 1 and {support.MAX_RTSP_STREAMS}.")
    if args.threads_per_model < 0:
        parser.error("--threads-per-model must be >= 0 (0 = auto).")
    if args.metrics_interval < 0:
        parser.error("--metrics-interval must be >= 0 (0 disables).")
    if args.input_fps < 0:
        parser.error("--input-fps must be >= 0 (0 = no limit).")
    try:
        support.parse_rtsp_frame_size(args.rtsp_frame_size)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        args.imgsz = support.parse_imgsz(str(args.imgsz))
    except ValueError as exc:
        parser.error(str(exc))

    # A dedicated GPU toggle maps to OpenVINO's Intel GPU device. It takes
    # precedence over --device so the flag is unambiguous.
    if args.gpu:
        args.device = "intel:gpu"
    on_gpu = str(args.device).lower().startswith(("gpu", "intel:gpu"))

    # The bundled default targets an INT8 IR, which only benchmarks reliably on
    # CPUs with an accelerated INT8 dot-product path (AMX-INT8/AVX-512
    # VNNI/AVX-VNNI). On a CPU without any of those (or when running on the
    # GPU, where the CPU flag check does not apply), quietly fall back to the
    # FP16 default so a brand-new machine works out of the box instead of
    # falling into a slow/unsupported INT8 path or an on-the-fly NNCF export
    # that needs internet access for calibration data.
    default_model = parser.get_default("model")
    if args.model == default_model:
        int8_model_dir = Path(args.models_dir) / "yolo26n_640x384_int8_openvino_model"
        if on_gpu or "INT8" in _active_precision_tags() or int8_model_dir.exists():
            args.model = str(int8_model_dir)
        else:
            args.model = str(Path(args.models_dir) / "yolo26n_640x384_openvino_model")
            print(
                "No INT8-accelerated CPU path detected (no AMX-INT8/AVX-512 VNNI/"
                "AVX-VNNI); defaulting to the FP16 model instead of INT8. Pass "
                "--model explicitly to override."
            )

    if args.download_all_models:
        _prefetch_openvino_models(args.models_dir, args.imgsz)

    # AMX ISA control (must precede OpenVINO native import).
    support.configure_amx(args.amx == "on")

    # amx_detect.py is the authority for AMX capability. "AMX ON" is only real
    # when requested AND the CPU/OS can actually run it. AMX is a CPU feature,
    # so it never applies when inference runs on the GPU.
    amx_status = detect_amx()
    amx_effective = args.amx == "on" and amx_status.usable and not on_gpu
    args.amx_effective = amx_effective

    # Report AMX status plus any other on-CPU AI accelerators (server & client).
    _report_cpu_ai_accelerators()

    # Resilient, low-latency RTSP capture for the FFmpeg backend. nobuffer +
    # low_delay + reorder_queue_size;0 stop the demuxer from queueing frames,
    # which (with the latest-frame reader) keeps the preview near real time.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|avioflags;direct|"
        "reorder_queue_size;0|stimeout;5000000|timeout;5000000",
    )
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

    rtsp_sources = support.parse_rtsp_sources(
        args.rtsp_url, args.rtsp_urls, args.num_streams
    )

    # Per-workflow thread budget. With N concurrent models on a fixed core set,
    # letting each model grab every core causes oversubscription and wrecks
    # aggregate FPS. Cap oneDNN/OpenVINO threads per model and (below) pin each
    # workflow to its own cores so the pipelines don't fight for the same units.
    cores = support.available_cores()[:MAX_AI_CORES]
    if args.threads_per_model > 0:
        threads_per_model = args.threads_per_model
    else:
        threads_per_model = max(1, len(cores) // len(rtsp_sources))
    # OMP_NUM_THREADS mainly governs the PyTorch comparison path; stock OpenVINO
    # is TBB-based and ignores it. The OpenVINO thread budget is left to the
    # LATENCY performance hint's auto-tuning (reinforced by the per-workflow CPU
    # pinning below). Forcing INFERENCE_NUM_THREADS/NUM_STREAMS here was found to
    # roughly halve aggregate FPS with many cameras.
    os.environ["OMP_NUM_THREADS"] = str(threads_per_model)

    core_assignments = [
        sorted({cores[(i * threads_per_model + k) % len(cores)] for k in range(threads_per_model)})
        for i in range(len(rtsp_sources))
    ]
    assigned_core_ids = sorted({cid for subset in core_assignments for cid in subset})
    print(
        f"Cores available: {len(cores)} | threads/model: {threads_per_model} | "
        f"core assignments: {core_assignments}"
    )

    print(f"Validating {len(rtsp_sources)} RTSP stream(s)...")
    # Validate all streams concurrently; opening 8 RTSP feeds one-by-one (each
    # waiting for its first keyframe) dominated startup time.
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=len(rtsp_sources)) as pool:
        validations = list(pool.map(support.validate_rtsp_stream, rtsp_sources))

    camera_input_fps: list[float] = []
    for index, (is_valid, fps, error_msg) in enumerate(validations, start=1):
        if not is_valid:
            raise RuntimeError(f"camera {index}: {error_msg}")
        camera_input_fps.append(fps)
        print(f"Camera {index} stream is valid. Input FPS: {fps:.2f}")

    model_source = support.resolve_model_source(str(args.model), args.imgsz)
    openvino_enabled = support.is_openvino_model(model_source)
    model_precision = support.detect_model_precision(model_source)

    # Resolution context: the model's static input size and the RTSP frame size
    # each incoming frame is resized to before inference. Printed so a varying
    # input resolution is always traceable to what actually runs.
    model_input_size = support.read_model_input_size(model_source)
    if model_input_size is not None:
        model_input_str = f"{model_input_size[1]}x{model_input_size[0]}"  # WxH
    else:
        model_input_str = "dynamic"
    rtsp_frame_size = support.parse_rtsp_frame_size(args.rtsp_frame_size)
    if rtsp_frame_size is not None:
        rtsp_frame_size_str = f"{rtsp_frame_size[0]}x{rtsp_frame_size[1]}"  # WxH
    else:
        infer_h, infer_w = support.imgsz_dimensions(args.imgsz)
        rtsp_frame_size_str = f"{infer_w}x{infer_h} (imgsz)"
    print(
        f"Model input (static): {model_input_str} | RTSP frame size: {rtsp_frame_size_str}"
    )

    # Speed lever: pin OpenVINO CPU precision. INT8 keeps its quantized ops.
    # For float models, run BF16 only when the CPU has hardware BF16 (AMX-BF16
    # or AVX-512 BF16); on client CPUs without it (e.g. Panther Lake) a bf16
    # hint forces a slow emulated path, so run native f32 for best throughput.
    if model_precision == "INT8":
        os.environ.pop("OV_CPU_INFERENCE_PRECISION_HINT", None)
        inference_precision = "INT8"
    elif on_gpu:
        # The Intel GPU plugin picks its own precision (FP16 native); the CPU
        # precision hint does not apply.
        inference_precision = "FP16"
    elif support.cpu_supports_bf16():
        os.environ["OV_CPU_INFERENCE_PRECISION_HINT"] = "bf16"
        inference_precision = "BF16"
    else:
        os.environ["OV_CPU_INFERENCE_PRECISION_HINT"] = "f32"
        inference_precision = "FP32"

    # Static header snapshot, collected once before the FPS benchmark runs.
    static_info = {
        "cpu_model": support.read_cpu_model(),
        "core_count": support.read_core_count(),
        "gdc_version": support.read_gdc_version(),
        "cores_available": len(cores),
        "threads_per_model": threads_per_model,
        "core_assignments": core_assignments,
        "device": args.device,
        "model_name": os.path.basename(str(args.model)),
        "openvino_enabled": openvino_enabled,
        "openvino_version": support.openvino_version(),
        "model_precision": model_precision,
        "inference_precision": inference_precision,
        "model_input_size": model_input_str,
        "rtsp_frame_size": rtsp_frame_size_str,
        "amx_requested": args.amx,
        "ai_accelerators": _active_ai_accelerators(),
        "camera_input_fps": camera_input_fps,
        "input_fps_throttle": args.input_fps,
        "assigned_core_ids": assigned_core_ids,
        "show_cpu_info": "cpu_info" in args.show,
    }

    streams = [
        StreamState(stream_id=index, source=source)
        for index, source in enumerate(rtsp_sources, start=1)
    ]
    dashboard = DashboardState()

    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_run_inference_loop,
            args=(args, stream, dashboard, model_source, cores_subset, stop_event),
            daemon=True,
        )
        for stream, cores_subset in zip(streams, core_assignments)
    ]
    threads.append(
        threading.Thread(
            target=_run_dashboard_loop,
            args=(args, streams, dashboard, stop_event),
            daemon=True,
        )
    )
    if args.metrics_interval > 0:
        threads.append(
            threading.Thread(
                target=_run_metrics_loop,
                args=(streams, stop_event, args.metrics_interval, static_info),
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()

    title = "YOLO26 OpenVINO FPS Benchmark"
    httpd = server.ThreadingHTTPServer(
        (args.host, args.port),
        _make_handler(streams, dashboard, stop_event, title, static_info, amx_status, amx_effective),
    )

    bind_host = "localhost" if args.host == "0.0.0.0" else args.host
    device_label = "Intel GPU (intel:gpu)" if on_gpu else f"CPU ({args.device})"
    print(f"Inference device: {device_label}")
    print(f"AMX requested: {args.amx} | effective: {'ON' if amx_effective else 'OFF'}")
    print(f"Open Chrome at http://{bind_host}:{args.port}/")
    print(f"Composite dashboard MJPEG endpoint: http://{bind_host}:{args.port}/dashboard.mjpg")
    print(f"Headless FPS metrics (JSON): http://{bind_host}:{args.port}/metrics")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
