#!/usr/bin/env python3
"""Isolate whether OpenVINO can execute the YOLO26 IR on this machine.

Runs several device/precision configurations, each in its own subprocess, so a
native segfault in one trial is reported (exit code) instead of killing the
harness. Use to pinpoint OpenVINO/silicon crashes independently of Ultralytics
and the benchmark pipeline.

  python src/ov_cpu_smoketest.py --model models/yolo26n_640x640_openvino_model
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_xml(model_dir: str) -> str:
    path = Path(model_dir)
    if path.suffix.lower() == ".xml":
        return str(path)
    candidates = sorted(path.glob("*.xml")) or sorted(path.rglob("*.xml"))
    if not candidates:
        raise SystemExit(f"No .xml IR found under {model_dir}")
    return str(candidates[0])


def _run_trial(xml: str, device: str, precision: str, isa: str) -> int:
    """One isolated trial: compile + one inference. Returns process exit code."""
    env = dict(os.environ)
    if isa:
        env["ONEDNN_MAX_CPU_ISA"] = isa
    else:
        env.pop("ONEDNN_MAX_CPU_ISA", None)
    code = (
        "import numpy as np, openvino as ov\n"
        f"xml = r'''{xml}'''\n"
        f"device = '{device}'\n"
        f"precision = '{precision}'\n"
        "core = ov.Core()\n"
        "print('available:', core.available_devices, flush=True)\n"
        "model = core.read_model(xml)\n"
        "config = {'PERFORMANCE_HINT': 'LATENCY'}\n"
        "if precision:\n"
        "    config['INFERENCE_PRECISION_HINT'] = getattr(ov.Type, precision)\n"
        "compiled = core.compile_model(model, device, config)\n"
        "print('compiled on', compiled.get_property('EXECUTION_DEVICES'), flush=True)\n"
        "port = compiled.input()\n"
        "shape = [d if isinstance(d, int) and d > 0 else 1 for d in port.get_partial_shape().get_max_shape()] if not port.get_partial_shape().is_dynamic else [1,3,640,640]\n"
        "x = np.random.rand(*shape).astype(np.float32)\n"
        "out = compiled(x)\n"
        "print('inference OK, outputs:', len(out), flush=True)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo26n_640x640_openvino_model")
    args = parser.parse_args(argv)

    xml = _find_xml(args.model)
    print(f"Model IR: {xml}\n")

    # (label, device, precision, oneDNN ISA cap)
    trials = [
        ("CPU default (reduced precision)", "CPU", "", ""),
        ("CPU f32 hint", "CPU", "f32", ""),
        ("CPU f32 hint + ISA AVX512_CORE", "CPU", "f32", "AVX512_CORE"),
        ("CPU f32 hint + ISA AVX2", "CPU", "f32", "AVX2"),
        ("GPU", "GPU", "", ""),
    ]

    results = []
    for label, device, precision, isa in trials:
        print(f"=== Trial: {label} ===", flush=True)
        code = _run_trial(xml, device, precision, isa)
        status = "OK" if code == 0 else (
            f"SIGSEGV (exit {code})" if code < 0 else f"FAILED (exit {code})"
        )
        print(f"--- {label}: {status}\n", flush=True)
        results.append((label, status))

    print("================ SUMMARY ================")
    for label, status in results:
        print(f"  {status:<22} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
