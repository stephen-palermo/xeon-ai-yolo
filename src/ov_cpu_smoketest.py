#!/usr/bin/env python3
"""Isolate what makes YOLO26 inference segfault on this machine.

Each trial runs in its own subprocess, so a native segfault is reported as an
exit code instead of killing the harness. Trials progressively add the layers
the benchmark uses (raw OpenVINO -> torch+OpenVINO -> Ultralytics -> Ultralytics
in a background thread) to pinpoint which layer crashes.

  python src/ov_cpu_smoketest.py --model models/yolo26n_640x640_openvino_model
"""

from __future__ import annotations

import argparse
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


# --- Trial bodies (run as `python -c <code>` subprocesses) -------------------
_RAW_OV = """
import numpy as np, openvino as ov
core = ov.Core()
model = core.read_model(r'''{xml}''')
compiled = core.compile_model(model, 'CPU', {{'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': ov.Type.f32}})
x = np.random.rand(1, 3, 640, 640).astype(np.float32)
print('raw OV outputs:', len(compiled(x)), flush=True)
"""

_TORCH_THEN_OV = """
import torch  # imported first, like Ultralytics
import numpy as np, openvino as ov
print('torch', torch.__version__, 'threads', torch.get_num_threads(), flush=True)
core = ov.Core()
model = core.read_model(r'''{xml}''')
compiled = core.compile_model(model, 'CPU', {{'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': ov.Type.f32}})
x = np.random.rand(1, 3, 640, 640).astype(np.float32)
print('torch+OV outputs:', len(compiled(x)), flush=True)
"""

_ULTRALYTICS = """
import numpy as np
from ultralytics import YOLO
model = YOLO(r'''{model_dir}''', task='detect')
frame = (np.random.rand(608, 1080, 3) * 255).astype('uint8')
r = model.predict([frame], imgsz=640, device='intel:cpu', verbose=False)
print('ultralytics predict OK, results:', len(r), flush=True)
"""

_ULTRALYTICS_THREAD = """
import threading, numpy as np
from ultralytics import YOLO
err = {{}}
def run():
    try:
        model = YOLO(r'''{model_dir}''', task='detect')
        frame = (np.random.rand(608, 1080, 3) * 255).astype('uint8')
        r = model.predict([frame], imgsz=640, device='intel:cpu', verbose=False)
        print('ultralytics(thread) predict OK, results:', len(r), flush=True)
    except BaseException as e:  # noqa: BLE001
        err['e'] = repr(e)
t = threading.Thread(target=run)
t.start(); t.join()
if err:
    print('thread error:', err['e'], flush=True); raise SystemExit(1)
"""

_ULTRALYTICS_F32 = """
import openvino as ov
_orig = ov.Core.compile_model
def _cm(self, model, device_name=None, config=None, *a, **k):
    if str(device_name or '').upper() in ('', 'CPU', 'AUTO'):
        config = dict(config or {{}})
        config.setdefault('INFERENCE_PRECISION_HINT', ov.Type.f32)
    return _orig(self, model, device_name, config, *a, **k)
ov.Core.compile_model = _cm
import numpy as np
from ultralytics import YOLO
model = YOLO(r'''{model_dir}''', task='detect')
frame = (np.random.rand(608, 1080, 3) * 255).astype('uint8')
r = model.predict([frame], imgsz=640, device='intel:cpu', verbose=False)
print('ultralytics f32 predict OK, results:', len(r), flush=True)
"""

_TORCHVISION_NMS = """
import torch
print('torch', torch.__version__, flush=True)
import torchvision
print('torchvision', torchvision.__version__, flush=True)
import torchvision.ops as ops
boxes = torch.rand(100, 4)
boxes[:, 2:] += boxes[:, :2]
scores = torch.rand(100)
keep = ops.nms(boxes, scores, 0.5)
print('torchvision nms OK, kept', len(keep), flush=True)
"""


def _run(code: str) -> int:
    return subprocess.run([sys.executable, "-c", code]).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo26n_640x640_openvino_model")
    args = parser.parse_args(argv)

    xml = _find_xml(args.model)
    model_dir = str(Path(xml).parent)
    print(f"Model IR: {xml}\n")

    trials = [
        ("raw OpenVINO (CPU f32)", _RAW_OV.format(xml=xml)),
        ("torch then OpenVINO (CPU f32)", _TORCH_THEN_OV.format(xml=xml)),
        ("torch/torchvision NMS", _TORCHVISION_NMS),
        ("Ultralytics predict (CPU)", _ULTRALYTICS.format(model_dir=model_dir)),
        ("Ultralytics predict f32 hint (CPU)", _ULTRALYTICS_F32.format(model_dir=model_dir)),
        ("Ultralytics predict in thread (CPU)", _ULTRALYTICS_THREAD.format(model_dir=model_dir)),
    ]

    results = []
    for label, code in trials:
        print(f"=== Trial: {label} ===", flush=True)
        rc = _run(code)
        status = "OK" if rc == 0 else (
            f"SIGSEGV (exit {rc})" if rc < 0 else f"FAILED (exit {rc})"
        )
        print(f"--- {label}: {status}\n", flush=True)
        results.append((label, status))

    print("================ SUMMARY ================")
    for label, status in results:
        print(f"  {status:<22} {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
