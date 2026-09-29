#!/usr/bin/env bash
# Push a video file into MediaMTX on a loop so the benchmark has an RTSP feed.
# Run this in its own terminal AFTER ./mediamtx is already running.
#
# Usage:
#   ./stream.ffmpeg.sh [VIDEO_FILE] [RTSP_URL]
#
# Defaults match 1x1x8.nova-lake.yolo26n.sh (native 1080x608, ~16:9).
set -euo pipefail

VIDEO_FILE="${1:-stairway.video.mp4}"
RTSP_URL="${2:-rtsp://localhost:8554/stream}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found; installing..."
    apt install ffmpeg -y
fi

if [[ ! -f "$VIDEO_FILE" ]]; then
    echo "Error: video file not found: $VIDEO_FILE" >&2
    exit 1
fi

echo "Streaming '$VIDEO_FILE' -> '$RTSP_URL' (loop). Press Ctrl-C to stop."
exec ffmpeg -re -stream_loop -1 -i "$VIDEO_FILE" -c copy -f rtsp "$RTSP_URL"
