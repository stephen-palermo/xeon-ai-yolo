#!/usr/bin/env python3
"""
Export YOLO26 object-detection models to a TensorRT engine via Ultralytics, for
benchmarking on an installed NVIDIA GPU (e.g. an L4).

This mirrors ``download_yolo26.py`` (which targets OpenVINO/CPU) so the same
YOLO26 variant/resolution can be compared apples-to-apples between an Intel
CPU (OpenVINO IR) and an NVIDIA GPU (TensorRT engine) run of
``yolo-cv-benchmark.py``.

Each requested resolution is exported to its own ``.engine`` file named by the
(stride-32 rounded) input size, e.g.:

    models/yolo26n_640x384_int8.engine
    models/yolo26n_640x384_fp16.engine

Building a TensorRT engine requires the NVIDIA driver + TensorRT to be
installed and an NVIDIA GPU visible to the process (``nvidia-smi``). INT8
export additionally needs a calibration dataset (Ultralytics defaults to
COCO8 when none is supplied).

Usage:
    python download_yolo26_tensorrt.py                       # yolo26n @ 640x384, FP16
    python download_yolo26_tensorrt.py --imgsz 1280x768
    python download_yolo26_tensorrt.py --variant s --int8
"""

import argparse
import math
from pathlib import Path

VARIANTS = ("n", "s", "m", "l", "x")
STRIDE = 32

# Friendly aliases -> (width, height) before stride rounding.
RES_PRESETS = {
    "640x360": (640, 360),
    "640x480": (640, 480),
    "640x640": (640, 640),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1280x768": (1280, 768),
}


def _round_stride(value: int) -> int:
    """Round up to the nearest multiple of the model stride (32)."""
    return max(STRIDE, math.ceil(value / STRIDE) * STRIDE)


def parse_resolution(spec: str) -> tuple[int, int]:
    """Return a stride-valid (width, height) for a preset, WxH, or N spec."""
    raw = spec.strip().lower()
    if raw in RES_PRESETS:
        width, height = RES_PRESETS[raw]
    elif "x" in raw:
        width, height = (int(part) for part in raw.split("x", 1))
    else:
        width = height = int(raw)
    return _round_stride(width), _round_stride(height)


def export_model(
    variant: str, models_dir: Path, width: int, height: int, int8: bool, device: str
) -> Path:
    """Export one YOLO26 ``variant`` at ``width``x``height`` to a TensorRT engine."""
    import shutil

    from ultralytics import YOLO

    name = f"yolo26{variant}"
    suffix = "int8" if int8 else "fp16"
    target = models_dir / f"{name}_{width}x{height}_{suffix}.engine"
    if target.exists():
        print(f"Skipping {target} (already present)")
        return target

    print(f"Exporting {name} @ {width}x{height} -> {target} ({'INT8' if int8 else 'FP16'}) ...")
    # YOLO(f"{name}.pt") auto-downloads the pretrained weights when missing.
    model = YOLO(f"{name}.pt")
    exported = Path(
        model.export(
            format="engine",
            imgsz=[height, width],
            half=not int8,
            int8=int8,
            device=device,
            batch=1,
        )
    )
    if exported.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(target))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="n",
        help="Which YOLO26 size to export (default: n).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every variant (n/s/m/l/x) instead of a single one.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        default=["640x384"],
        help="One or more resolutions to export (preset, WxH, or N). Rounded up to x32.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store the exported TensorRT engine (default: models).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export an INT8 quantized engine (needs a calibration dataset).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index to build/export the engine on (default: 0).",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS if args.all else (args.variant,)
    resolutions = [parse_resolution(spec) for spec in args.imgsz]

    exported = [
        export_model(variant, models_dir, width, height, args.int8, args.device)
        for variant in variants
        for width, height in resolutions
    ]

    print("\n" + "=" * 60)
    print(f"YOLO26 TensorRT ENGINE READY ({'INT8' if args.int8 else 'FP16'})")
    for target in exported:
        print(f"  Model: {target}")
    print("=" * 60)
    print("\nRun the benchmark (match --imgsz to the model size; set")
    print("--rtsp-frame-size to your actual stream resolution):")
    for target in exported:
        size = target.stem.split("_")[1]
        print(
            f"  python src/yolo-cv-benchmark.py --model {target} --imgsz {size} \\\n"
            f"      --nvidia-gpu --rtsp-frame-size <stream-WxH> --rtsp-urls \"rtsp://...\" "
            f"--host 0.0.0.0 --port 8080 --classes 0"
        )


if __name__ == "__main__":
    main()
