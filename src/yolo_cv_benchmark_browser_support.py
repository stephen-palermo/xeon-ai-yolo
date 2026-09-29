#!/usr/bin/env python3
"""Lightweight, self-contained helpers for ``yolo-cv-benchmark.py``.

This module is intentionally independent from the older benchmark/preview
modules. It contains only pure, loop-free helpers (system info, model
resolution, RTSP capture, AMX ISA control) so nothing here spins background
work or consumes CPU after startup. All runtime CPU should be spent on YOLO26
inference (the FPS-throughput goal), not on bookkeeping.

AMX capability reporting defers to :mod:`amx_detect`, which remains the single
source of truth for whether the CPU/OS support AMX-TILE/BF16/INT8.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_RTSP_STREAMS = 32

# oneDNN ISA caps. AMX kernels require AVX512_CORE_AMX; capping at
# AVX512_CORE_BF16 disables AMX while keeping AVX-512 + BF16 paths available.
# This lets the same script demonstrate the AMX speedup by toggling --amx.
_ONEDNN_ISA_AMX_ON = "AVX512_CORE_AMX"
_ONEDNN_ISA_AMX_OFF = "AVX512_CORE_BF16"


def read_cpu_flags() -> set[str]:
    """Return the set of CPU feature flags from Linux /proc/cpuinfo (or empty)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("flags") and ":" in line:
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def cpu_has_amx() -> bool:
    """True when the CPU exposes AMX tiles (server Xeon, not client CPUs)."""
    return "amx_tile" in read_cpu_flags()


def cpu_supports_bf16() -> bool:
    """True when the CPU has hardware BF16 compute (AMX-BF16 or AVX-512 BF16).

    Client CPUs such as Panther Lake have neither, so a bf16 inference hint
    would force a slow emulated path instead of accelerating anything.
    """
    return bool(read_cpu_flags() & {"amx_bf16", "avx512_bf16"})


def configure_amx(enabled: bool) -> None:
    """Cap the oneDNN ISA so AMX is allowed (ON) or disabled (OFF).

    Must be called before OpenVINO imports its native libraries.

    Only meaningful on CPUs that actually have AMX (server Xeon). On client
    CPUs without AVX-512/AMX (e.g. Panther Lake) capping at an AVX-512 tier is
    nonsensical, so the cap is cleared and oneDNN is left to dispatch to the
    best available client ISA (AVX-VNNI / AVX2) on its own.
    """
    import os  # noqa: PLC0415

    if cpu_has_amx():
        os.environ["ONEDNN_MAX_CPU_ISA"] = (
            _ONEDNN_ISA_AMX_ON if enabled else _ONEDNN_ISA_AMX_OFF
        )
    else:
        os.environ.pop("ONEDNN_MAX_CPU_ISA", None)
    os.environ.setdefault("ONEDNN_VERBOSE", "0")


# ---------------------------------------------------------------------------
# CLI value parsing
# ---------------------------------------------------------------------------
def parse_rtsp_sources(
    rtsp_url: str, rtsp_urls: str, num_streams: int
) -> list[str]:
    """Build the RTSP source list from CLI flags, allowing duplicates."""
    if rtsp_urls:
        sources = [item.strip() for item in rtsp_urls.split(",") if item.strip()]
    elif rtsp_url:
        sources = [rtsp_url for _ in range(num_streams)]
    else:
        raise ValueError("Provide --rtsp-url or --rtsp-urls.")

    if not sources:
        raise ValueError("No RTSP stream sources were provided.")
    if len(sources) > MAX_RTSP_STREAMS:
        raise ValueError(
            f"At most {MAX_RTSP_STREAMS} RTSP streams are supported; got {len(sources)}."
        )
    return sources


def parse_classes(value: str) -> list[int] | None:
    """Parse a comma-separated class list. Empty means all classes."""
    value = value.strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_rtsp_frame_size(value: str) -> tuple[int, int] | None:
    """Parse --rtsp-frame-size as WIDTHxHEIGHT (for example, 640x360)."""
    raw = value.strip()
    if not raw:
        return None

    normalized = raw.lower().replace(",", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --rtsp-frame-size '{value}'. Expected WIDTHxHEIGHT (for example, 640x360)."
        )

    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("--rtsp-frame-size width and height must be positive integers.")

    return width, height


def parse_imgsz(value: str) -> int | tuple[int, int]:
    """Parse --imgsz as N or WIDTHxHEIGHT, returning int or (height, width)."""
    raw = value.strip()
    if not raw:
        raise ValueError("--imgsz cannot be empty.")

    if any(separator in raw.lower() for separator in ("x", ",")):
        width, height = parse_rtsp_frame_size(raw)
        return (height, width)

    size = int(raw)
    if size <= 0:
        raise ValueError("--imgsz must be a positive integer or WIDTHxHEIGHT.")
    return size


def imgsz_dimensions(imgsz: int | tuple[int, int]) -> tuple[int, int]:
    """Return inference size as (height, width)."""
    if isinstance(imgsz, tuple):
        return imgsz
    return (imgsz, imgsz)


def imgsz_max_dimension(imgsz: int | tuple[int, int]) -> int:
    """Return the largest dimension from an ``imgsz`` value."""
    height, width = imgsz_dimensions(imgsz)
    return max(height, width)


# ---------------------------------------------------------------------------
# Model resolution / loading
# ---------------------------------------------------------------------------
def resolve_model_source(model_arg: str, imgsz: int | tuple[int, int]) -> str:
    """Resolve a model argument to a path Ultralytics can load via OpenVINO.

    - ``.pt`` weights are exported once to an OpenVINO IR directory.
    - An existing OpenVINO directory or ``.xml`` file is used as-is.
    """
    path = Path(model_arg)

    if model_arg.endswith(".pt"):
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(model_arg)
        ov_path = model.export(
            format="openvino",
            imgsz=imgsz,
            batch=1,
        )
        return str(ov_path)

    if path.suffix.lower() == ".xml" and path.is_file():
        return str(path.resolve())
    if path.is_dir():
        # Resolve to a concrete IR XML path when present.
        preferred = path / "openvino_model.xml"
        if preferred.is_file():
            return str(preferred.resolve())

        xml_candidates = sorted(path.glob("*.xml")) or sorted(path.rglob("*.xml"))
        if xml_candidates:
            return str(xml_candidates[0].resolve())

        return str(path.resolve())

    # Convenience: if a user points to an OpenVINO directory that does not exist
    # yet (for example models/yolo26n_openvino_model/), infer the base .pt model,
    # export it, and place the IR at the requested path.
    if path.suffix == "" and path.name.endswith("_openvino_model"):
        import shutil  # noqa: PLC0415

        from ultralytics import YOLO  # noqa: PLC0415

        requested_dir = path
        requested_dir.parent.mkdir(parents=True, exist_ok=True)

        base_name = path.name[: -len("_openvino_model")]
        export_imgsz = imgsz
        export_int8 = False
        # Honor an optional _int8 suffix (INT8 quantized export) and a trailing
        # _WxH suffix (e.g. yolo26n_640x384_int8_openvino_model): export at that
        # size/precision and strip both to recover the .pt weights name.
        if base_name.endswith("_int8"):
            export_int8 = True
            base_name = base_name[: -len("_int8")]
        match = re.match(r"^(.*)_(\d+)x(\d+)$", base_name)
        if match:
            base_name = match.group(1)
            export_imgsz = [int(match.group(3)), int(match.group(2))]  # [h, w]
        pt_local = requested_dir.parent / f"{base_name}.pt"

        model = YOLO(str(pt_local))
        # export() writes the IR next to the weights and ignores project/name, so
        # relocate the result into the requested directory.
        exported = Path(
            model.export(
                format="openvino",
                imgsz=export_imgsz,
                half=not export_int8,
                int8=export_int8,
                batch=1,
            )
        )
        if exported.resolve() != requested_dir.resolve():
            if requested_dir.exists():
                shutil.rmtree(requested_dir)
            shutil.move(str(exported), str(requested_dir))
        return str(requested_dir)

    return model_arg


def build_model(model_source: str):
    """Load a YOLO26 model through OpenVINO for CPU inference.

    Ultralytics loads an exported ``*_openvino_model`` directory natively, so a
    concrete IR ``.xml`` path is mapped back to its parent directory.
    """
    path = Path(model_source)
    model_dir = path.parent if path.suffix.lower() == ".xml" else path

    from ultralytics import YOLO  # noqa: PLC0415

    return YOLO(str(model_dir), task="detect")


def is_openvino_model(model_source: str) -> bool:
    """Return True if the resolved model artifact is an OpenVINO IR."""
    path = Path(model_source)
    if path.suffix.lower() == ".xml":
        return True
    if path.is_dir():
        return path.name.endswith("_openvino_model") or any(path.glob("*.xml"))
    return False


def _find_ir_xml(model_source: str) -> Path | None:
    """Return the OpenVINO IR ``.xml`` path for the resolved model, if any."""
    path = Path(model_source)
    if path.suffix.lower() == ".xml" and path.is_file():
        return path
    if path.is_dir():
        preferred = path / "openvino_model.xml"
        if preferred.is_file():
            return preferred
        xml_candidates = sorted(path.glob("*.xml")) or sorted(path.rglob("*.xml"))
        if xml_candidates:
            return xml_candidates[0]
    return None


def detect_model_precision(model_source: str) -> str:
    """Infer the model precision (INT8/FP16/FP32) from the model artifact.

    OpenVINO IR is inspected directly; other formats (e.g. TensorRT ``.engine``
    files) fall back to the ``_int8``/``_fp16`` naming convention used by this
    repo's export scripts.
    """
    xml_path = _find_ir_xml(model_source)
    if xml_path is None:
        name = Path(model_source).name.lower()
        if "int8" in name:
            return "INT8"
        if "fp16" in name or "half" in name:
            return "FP16"
        return "unknown"
    try:
        text = xml_path.read_text(errors="ignore")
    except OSError:
        return "unknown"

    if 'FakeQuantize' in text or 'element_type="i8"' in text or 'element_type="u8"' in text:
        return "INT8"
    if 'element_type="f16"' in text:
        return "FP16"
    if 'element_type="f32"' in text:
        return "FP32"
    return "unknown"


def read_model_input_size(model_source: str) -> tuple[int, int] | None:
    """Return the model's static input size as (height, width), or None.

    Reads the OpenVINO IR and inspects the first input's partial shape. Returns
    None when the model is missing or its spatial dims are dynamic.
    """
    xml_path = _find_ir_xml(model_source)
    if xml_path is None:
        return None
    try:
        import openvino as ov  # noqa: PLC0415

        model = ov.Core().read_model(str(xml_path))
        shape = model.inputs[0].get_partial_shape()
        if shape.rank.is_static and shape.rank.get_length() == 4:
            h_dim, w_dim = shape[2], shape[3]
            if h_dim.is_static and w_dim.is_static:
                return (h_dim.get_length(), w_dim.get_length())
    except Exception:  # noqa: BLE001
        return None
    return None


def openvino_version() -> str:
    """Return the installed OpenVINO version string (short form)."""
    try:
        import openvino  # noqa: PLC0415
    except ImportError:
        return "unknown"

    version = getattr(openvino, "__version__", "") or ""
    if not version:
        try:
            version = openvino.get_version()
        except Exception:  # noqa: BLE001
            version = "unknown"
    return version.split("-", 1)[0].strip() or "unknown"


# ---------------------------------------------------------------------------
# RTSP capture
# ---------------------------------------------------------------------------
# OpenCV's bundled FFmpeg demuxer segfaults (SIGSEGV) opening RTSP on some
# Linux/OpenVINO wheels, so RTSP capture is done by piping raw bgr24 frames from
# an ffmpeg subprocess instead of cv2.VideoCapture. cv2 is still used elsewhere
# for image resize/annotation; only the crashing capture backend is replaced.
def _find_executable(name: str) -> str | None:
    """Return the full path to an executable, or None if not on PATH."""
    import shutil  # noqa: PLC0415

    return shutil.which(name)


def _read_exact(stream, num_bytes: int) -> bytes:
    """Read exactly num_bytes from a pipe, or fewer only at EOF."""
    chunks: list[bytes] = []
    remaining = num_bytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _ffprobe_stream_info(source: str) -> tuple[int, int, float] | None:
    """Return (width, height, fps) for an RTSP video stream via ffprobe."""
    import json  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    ffprobe = _find_executable("ffprobe")
    if ffprobe is None:
        return None
    cmd = [
        ffprobe, "-v", "error", "-rtsp_transport", "tcp",
        "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of", "json", source,
    ]
    try:
        out = subprocess.check_output(cmd, timeout=15.0, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        streams = json.loads(out).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    info = streams[0]
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = str(info.get(key) or "0/0")
        num, _, den = rate.partition("/")
        try:
            numf, denf = float(num), float(den or 1)
        except ValueError:
            continue
        if denf > 0 and numf > 0:
            fps = numf / denf
            break
    if fps <= 0:
        fps = 30.0
    return width, height, fps


class FFmpegCapture:
    """Minimal RTSP capture backed by an ffmpeg subprocess (bgr24 pipe).

    Exposes the small subset of the cv2.VideoCapture API the benchmark uses:
    ``read() -> (ok, frame)`` and ``release()``.
    """

    def __init__(self, source: str, width: int, height: int, fps: float):
        import subprocess  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        self._np = np
        self._width = width
        self._height = height
        self.fps = fps
        self._frame_bytes = width * height * 3
        ffmpeg = _find_executable("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg executable not found on PATH")
        cmd = [
            ffmpeg, "-nostdin", "-loglevel", "quiet",
            "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
            "-i", source, "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
        )

    def read(self):
        if self._proc.stdout is None:
            return False, None
        raw = _read_exact(self._proc.stdout, self._frame_bytes)
        if len(raw) < self._frame_bytes:
            return False, None
        # Copy out of the immutable pipe buffer so downstream annotation can
        # draw on the frame in place (np.frombuffer views are read-only).
        frame = self._np.frombuffer(raw, dtype=self._np.uint8).reshape(
            self._height, self._width, 3
        ).copy()
        return True, frame

    def release(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            proc.kill()


def open_rtsp_capture(source: str):
    """Open an RTSP stream via an ffmpeg subprocess (low-latency bgr24 pipe)."""
    import time  # noqa: PLC0415

    # Retry the probe briefly so a transient hiccup during reconnect doesn't
    # kill the reader thread.
    info = None
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        info = _ffprobe_stream_info(source)
        if info is not None:
            break
        time.sleep(0.5)
    if info is None:
        raise RuntimeError(f"Unable to probe RTSP stream: {source}")
    width, height, fps = info
    return FFmpegCapture(source, width, height, fps)


def _parse_rtsp_endpoint(source: str) -> tuple[str | None, int]:
    """Return (host, port) for an ``rtsp://`` URL, or (None, 0) if not RTSP."""
    from urllib.parse import urlparse  # noqa: PLC0415

    parsed = urlparse(source)
    if parsed.scheme != "rtsp" or not parsed.hostname:
        return None, 0
    return parsed.hostname, parsed.port or 554


def _wait_for_tcp(host: str, port: int, timeout: float) -> bool:
    """Return True once a TCP connection to host:port succeeds within timeout."""
    import socket  # noqa: PLC0415
    import time  # noqa: PLC0415

    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def validate_rtsp_stream(source: str) -> tuple[bool, float, str]:
    """Validate an RTSP stream and read its advertised FPS.

    Returns (is_valid, fps, error_message). Runs once at startup only. Uses an
    ffmpeg subprocess (not cv2.VideoCapture) so a bad OpenCV FFmpeg build can't
    segfault the benchmark.
    """
    import time  # noqa: PLC0415

    if _find_executable("ffmpeg") is None or _find_executable("ffprobe") is None:
        return False, 0.0, "ffmpeg/ffprobe not found on PATH (apt install ffmpeg)"

    # Probe TCP reachability first (cheap), so we only spawn ffmpeg once the
    # port actually accepts connections.
    host, port = _parse_rtsp_endpoint(source)
    if host is not None and not _wait_for_tcp(host, port, timeout=15.0):
        return False, 0.0, f"Unable to open RTSP stream: {source}"

    info = _ffprobe_stream_info(source)
    if info is None:
        return False, 0.0, f"Unable to open RTSP stream: {source}"

    capture = FFmpegCapture(source, *info)
    try:
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                return True, capture.fps, ""
        return False, 0.0, f"Unable to read frames from RTSP stream: {source}"
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# System information (all cheap; only called at startup or on manual refresh)
# ---------------------------------------------------------------------------
def display_source_label(source: str) -> str:
    """Return a compact label for one RTSP source."""
    match = re.search(r"://[^/]+/(.+)$", source)
    if match:
        tail = match.group(1)
        return tail[-48:] if len(tail) > 48 else tail
    return source[-48:] if len(source) > 48 else source


def read_cpu_model() -> str:
    """Return the CPU model name via lscpu (Ubuntu server). Startup only."""
    import subprocess  # noqa: PLC0415

    try:
        output = subprocess.check_output(["lscpu"], text=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    for line in output.splitlines():
        if line.strip().startswith("Model name:"):
            return line.split(":", 1)[1].strip().replace("(R)", "")
    return "unknown"


def read_cpu_turbo() -> str:
    """Return CPU turbo/boost state as 'ON', 'OFF', or 'unknown' (Linux).

    Reads the Intel P-state driver first (``no_turbo``: 0 = turbo on, 1 = off),
    then the generic cpufreq ``boost`` flag (1 = on, 0 = off).
    """
    try:
        with open("/sys/devices/system/cpu/intel_pstate/no_turbo", "r", encoding="ascii") as handle:
            return "OFF" if handle.read().strip() == "1" else "ON"
    except OSError:
        pass
    try:
        with open("/sys/devices/system/cpu/cpufreq/boost", "r", encoding="ascii") as handle:
            return "ON" if handle.read().strip() == "1" else "OFF"
    except OSError:
        pass
    return "unknown"


def read_core_count() -> int:
    """Return the number of CPU cores this process may run on (CPU affinity)."""
    import os  # noqa: PLC0415

    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 0


def available_cores() -> list[int]:
    """Return the sorted list of CPU core IDs this process may run on.

    Honors an external ``taskset``/cgroup affinity mask on Linux; falls back to
    ``range(cpu_count())`` on platforms without ``sched_getaffinity``.
    """
    import os  # noqa: PLC0415

    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def smt_sibling_cores() -> set[int]:
    """Return logical CPU IDs backed by a hyperthreaded (SMT) physical core.

    When hyperthreading is ON, a physical core is split into two logical CPUs
    that share the same execution units, so *each* of those logical CPUs is a
    "virtual" core rather than a dedicated physical one. We therefore flag every
    logical CPU whose physical core exposes more than one thread sibling. Reads
    Linux topology; returns an empty set when unavailable (non-Linux or SMT off).
    """
    import glob  # noqa: PLC0415

    virtual: set[int] = set()
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
        try:
            with open(path, "r", encoding="ascii") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        siblings: list[int] = []
        for token in raw.replace(",", " ").split():
            if "-" in token:
                start, _, end = token.partition("-")
                siblings.extend(range(int(start), int(end) + 1))
            elif token:
                siblings.append(int(token))
        if len(siblings) > 1:
            # SMT active on this physical core: every sibling is a virtual thread.
            virtual.update(siblings)
    return virtual


def format_core_assignments(
    core_assignments: list[list[int]], virtual_cores: set[int]
) -> str:
    """Render per-model core assignments, tagging SMT/virtual cores with '*'.

    Appends a '(*=virtual/SMT core, hyperthreading on)' legend when any assigned
    core is a hyperthread thread so the terminal/dashboard clearly reports that a
    virtual core (not a dedicated physical core) is in use.
    """
    groups: list[str] = []
    any_virtual = False
    for subset in core_assignments:
        labels: list[str] = []
        for cid in subset:
            if cid in virtual_cores:
                any_virtual = True
                labels.append(f"{cid}*")
            else:
                labels.append(str(cid))
        groups.append("[" + ", ".join(labels) + "]")
    rendered = "[" + ", ".join(groups) + "]"
    if any_virtual:
        rendered += " (*=virtual/SMT core, hyperthreading on)"
    return rendered


def read_gdc_version() -> str:
    """Return the first node's kubelet version via kubectl. Startup only."""
    import subprocess  # noqa: PLC0415

    try:
        output = subprocess.check_output(
            [
                "kubectl",
                "get",
                "nodes",
                "-o",
                "jsonpath={.items[0].status.nodeInfo.kubeletVersion}",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return output or "unknown"


def read_memory_used() -> str:
    """Return current memory usage as 'used GB / total GB' from /proc/meminfo."""
    try:
        info: dict[str, str] = {}
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key.strip()] = rest.strip()
        total_kb = int(info["MemTotal"].split()[0])
        available_kb = int(info.get("MemAvailable", info["MemFree"]).split()[0])
        used_gb = (total_kb - available_kb) / (1024 * 1024)
        total_gb = total_kb / (1024 * 1024)
        return f"{used_gb:.1f} GB / {total_gb:.1f} GB"
    except (OSError, KeyError, ValueError, IndexError):
        return "unknown"


def read_cpu_core_frequencies() -> list[tuple[int, float]]:
    """Return per-logical-CPU current operating frequency in MHz (Ubuntu/Xeon)."""
    cores: list[tuple[int, float]] = []
    try:
        current_id: int | None = None
        with open("/proc/cpuinfo", "r", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                key = key.strip()
                if key == "processor":
                    try:
                        current_id = int(rest.strip())
                    except ValueError:
                        current_id = None
                elif key == "cpu MHz" and current_id is not None:
                    try:
                        cores.append((current_id, float(rest.strip())))
                    except ValueError:
                        pass
                    current_id = None
    except OSError:
        return []
    return sorted(cores, key=lambda item: item[0])


def grid_columns(count: int) -> int:
    """Return the number of columns for the browser camera grid (max 4)."""
    return min(4, max(1, count))


def read_cpu_stat_jiffies() -> dict[int, tuple[int, int]]:
    """Return ``{cpu_id: (idle_jiffies, total_jiffies)}`` from ``/proc/stat``.

    ``idle`` folds in ``iowait``; ``total`` sums every field. Compare two
    snapshots taken over an interval to derive per-core utilization. Linux only;
    returns an empty dict elsewhere.
    """
    result: dict[int, tuple[int, int]] = {}
    try:
        with open("/proc/stat", "r", encoding="ascii") as handle:
            for line in handle:
                if not line.startswith("cpu"):
                    continue
                fields = line.split()
                label = fields[0]
                if label == "cpu":
                    continue  # aggregate line; we want per-core entries only
                try:
                    cpu_id = int(label[3:])
                except ValueError:
                    continue
                values = [int(value) for value in fields[1:]]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                result[cpu_id] = (idle, sum(values))
    except (OSError, ValueError, IndexError):
        return {}
    return result


def cpu_utilization_from_jiffies(
    prev: dict[int, tuple[int, int]],
    curr: dict[int, tuple[int, int]],
    core_ids: list[int],
) -> dict[int, float]:
    """Return ``{cpu_id: percent_busy}`` for ``core_ids`` between two snapshots."""
    percents: dict[int, float] = {}
    for cpu_id in core_ids:
        if cpu_id not in prev or cpu_id not in curr:
            continue
        idle0, total0 = prev[cpu_id]
        idle1, total1 = curr[cpu_id]
        delta_total = total1 - total0
        delta_idle = idle1 - idle0
        if delta_total <= 0:
            percents[cpu_id] = 0.0
        else:
            percents[cpu_id] = max(0.0, min(100.0, (1.0 - delta_idle / delta_total) * 100.0))
    return percents
