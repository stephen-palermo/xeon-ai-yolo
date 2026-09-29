echo "TERMINAL#1 One-time setup — download MediaMTX"
echo "wget https://github.com/bluenviron/mediamtx/releases/download/v1.19.2/mediamtx_v1.19.2_linux_amd64.tar.gz"
echo "tar -xzf mediamtx_v1.19.2_linux_amd64.tar.gz"
echo "./mediamtx"

echo "TERMINAL#2 Stream your video file into it on a loop (native 1080x608, ~16:9)"
echo "apt install ffmpeg -y"
echo "ffmpeg -re -stream_loop -1 -i stairway.video.mp4 -c copy -f rtsp rtsp://localhost:8554/stream"


python src/download_yolo26.py --variant n --imgsz 640x640

# Intel Nova Lake integrated GPU run (OpenVINO 'intel:gpu'), 8 cameras.
# Requires the Intel GPU compute runtime + Level-Zero on the host:
#   sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
# Verify the GPU is visible:
#   python -c "import openvino as ov; print(ov.Core().available_devices)"   # expect 'GPU'
#
# Same FP16 IR as the CPU run (recorded.8x8x8.nova-lake.yolo26n.sh) so CPU vs GPU
# is apples-to-apples.
# --rtsp-frame-size 1080x608 matches the recorded stream so it passes through untouched; the model letterboxes it to 640x640 internally.
python src/yolo-cv-benchmark.py \
        --gpu \
        --model models/yolo26n_640x640_openvino_model \
        --input-fps 8 \
        --rtsp-frame-size 1080x608 \
        --imgsz 640x640 \
        --rtsp-urls "rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream,rtsp://localhost:8554/stream" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7
