#!/usr/bin/env python3
"""
Export YOLO26 object-detection models to OpenVINO IR via Ultralytics.

YOLO26 (Ultralytics, January 2026) is a native end-to-end / NMS-free detector.
Each requested resolution is exported to its own OpenVINO IR directory named by
the (stride-32 rounded) input size, so multiple resolutions coexist under one
models directory and ``src/yolo-cv-benchmark.py`` can load any of them:

    models/yolo26n_640x384_openvino_model
    models/yolo26n_640x480_openvino_model
    models/yolo26n_1280x768_openvino_model   (etc.)

Inference dimensions must be multiples of 32, so sizes are rounded up
automatically (for example 640x360 -> 640x384, 720p -> 1280x736).

FP16 weights by default (``half=True``). Pass ``--int8`` for an INT8 quantized
export (needs ``nncf`` and a calibration dataset; slower to prepare).

Resolution presets: 640x360, 640x480, 640x640, 720p, 1280x768 (or any WxH / N).

Usage:
    python download_yolo26.py                              # yolo26n @ 640x384
    python download_yolo26.py --imgsz 1280x768
    python download_yolo26.py --imgsz 640x360 1280x768     # build several
    python download_yolo26.py --variant s --imgsz 720p
    python download_yolo26.py --all --imgsz 640x640        # all n/s/m/l/x
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


def export_model(variant: str, models_dir: Path, width: int, height: int, int8: bool) -> Path:
    """Export one YOLO26 ``variant`` at ``width``x``height`` to an OpenVINO IR dir."""
    import shutil

    from ultralytics import YOLO

    name = f"yolo26{variant}"
    suffix = "_int8" if int8 else ""
    target = models_dir / f"{name}_{width}x{height}{suffix}_openvino_model"
    if (target / f"{name}.xml").exists():
        print(f"Skipping {target} (already present)")
        return target

    print(f"Exporting {name} @ {width}x{height} -> {target} ({'INT8' if int8 else 'FP16'}) ...")
    # YOLO(f"{name}.pt") auto-downloads the pretrained weights when missing.
    model = YOLO(f"{name}.pt")
    # export() writes the IR next to the weights and ignores project/name, so
    # relocate the result into the resolution-named target directory.
    exported = Path(
        model.export(
            format="openvino",
            imgsz=[height, width],
            half=not int8,
            int8=int8,
            batch=1,
        )
    )
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
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
        help="Directory to store the exported OpenVINO IR (default: models).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export an INT8 quantized model (needs nncf + calibration data).",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    variants = VARIANTS if args.all else (args.variant,)
    resolutions = [parse_resolution(spec) for spec in args.imgsz]

    exported = [
        export_model(variant, models_dir, width, height, args.int8)
        for variant in variants
        for width, height in resolutions
    ]

    print("\n" + "=" * 60)
    print(f"YOLO26 OpenVINO IR READY ({'INT8' if args.int8 else 'FP16'})")
    for target in exported:
        print(f"  Model: {target}")
    print("=" * 60)
    print("\nRun the benchmark (match --imgsz to the model size; set")
    print("--rtsp-frame-size to your actual stream resolution):")
    for target in exported:
        size = target.name.split("_")[1]
        print(
            f"  python src/yolo-cv-benchmark.py --model {target} --imgsz {size} \\\n"
            f"      --rtsp-frame-size <stream-WxH> --rtsp-urls \"rtsp://...\" "
            f"--host 0.0.0.0 --port 8080 --classes 0"
        )


if __name__ == "__main__":
    main()
