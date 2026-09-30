#!/usr/bin/env python3
"""Render the Nova Lake CPU vs GPU benchmark comparison plot.

Data is the steady-state / time-series from the uncapped single-stream runs
(YOLO26n @ 640x640, 30 FPS input). Regenerate the PNG with:

  python src/plot_nova_lake_results.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Cumulative-average FPS out reported each interval (ramp to steady state).
CPU_FPS = [
    0.5, 12.7, 16.4, 18.2, 19.2, 19.9, 20.4, 20.7, 21.0, 21.2, 21.4, 21.6,
    21.7, 21.8, 21.9, 22.0, 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8,
]
GPU_FPS = [
    6.9, 17.1, 20.8, 22.6, 23.4, 24.2, 24.7, 25.2, 25.6, 25.7, 26.0, 26.2,
    26.3, 26.4, 26.5, 26.7, 26.8, 26.9, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0,
]

CPU_STEADY_FPS, GPU_STEADY_FPS = 22.8, 27.0
CPU_LATENCY_MS, GPU_LATENCY_MS = 43.0, 8.5  # GPU = representative min inference


def main() -> None:
    fig, (ax_fps, ax_bar) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Nova Lake — YOLO26n @ 640x640, single stream (uncapped, 30 FPS input)",
        fontsize=12, fontweight="bold",
    )

    cpu_color, gpu_color = "#0071c5", "#76b900"

    # Left: FPS-out convergence over reporting intervals.
    x = range(1, len(CPU_FPS) + 1)
    ax_fps.plot(x, CPU_FPS, color=cpu_color, marker="o", ms=3, label="CPU (1 core, FP32)")
    ax_fps.plot(x, GPU_FPS, color=gpu_color, marker="s", ms=3, label="iGPU (FP16)")
    ax_fps.axhline(30, color="gray", ls="--", lw=1, label="input cap (30 FPS)")
    ax_fps.set_xlabel("reporting interval")
    ax_fps.set_ylabel("FPS out (cumulative avg)")
    ax_fps.set_title("Throughput ramp")
    ax_fps.set_ylim(0, 32)
    ax_fps.legend(loc="lower right", fontsize=9)
    ax_fps.grid(True, alpha=0.3)

    # Right: steady-state FPS and latency side by side (twin axis).
    labels = ["CPU\n(1 core)", "iGPU"]
    fps_vals = [CPU_STEADY_FPS, GPU_STEADY_FPS]
    lat_vals = [CPU_LATENCY_MS, GPU_LATENCY_MS]
    xpos = range(len(labels))
    width = 0.35

    bars_fps = ax_bar.bar(
        [p - width / 2 for p in xpos], fps_vals, width,
        color=[cpu_color, gpu_color], label="FPS out",
    )
    ax_bar.set_ylabel("FPS out")
    ax_bar.set_ylim(0, 32)
    ax_bar.set_xticks(list(xpos))
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title("Steady-state throughput vs latency")
    for bar, val in zip(bars_fps, fps_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}",
                    ha="center", fontsize=9)

    ax_lat = ax_bar.twinx()
    bars_lat = ax_lat.bar(
        [p + width / 2 for p in xpos], lat_vals, width,
        color="none", edgecolor="black", hatch="//", label="latency (ms)",
    )
    ax_lat.set_ylabel("latency (ms)")
    ax_lat.set_ylim(0, 55)
    for bar, val in zip(bars_lat, lat_vals):
        ax_lat.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.0f} ms",
                    ha="center", fontsize=9)

    lines = [bars_fps, bars_lat]
    ax_bar.legend(lines, [b.get_label() for b in lines], loc="upper center", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(__file__).resolve().parent.parent / "docs" / "nova_lake_cpu_vs_gpu.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
