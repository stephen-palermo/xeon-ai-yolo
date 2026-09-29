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
# Re-encode to baseline-friendly H.264 (Main profile, yuv420p) over RTSP/TCP.
# Passing the source through with `-c copy` can yield HEVC/10-bit/4:2:2 streams
# that OpenCV's bundled FFmpeg decoder segfaults on; re-encoding guarantees a
# stream the benchmark's capture backend can decode. Audio is dropped (-an).
exec ffmpeg -re -stream_loop -1 -i "$VIDEO_FILE" \
    -an -c:v libx264 -profile:v main -pix_fmt yuv420p \
    -preset veryfast -tune zerolatency -g 30 \
    -rtsp_transport tcp -f rtsp "$RTSP_URL"
