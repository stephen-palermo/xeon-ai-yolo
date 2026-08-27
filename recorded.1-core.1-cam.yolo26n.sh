echo "TERMINAL#1 One-time setup — download MediaMTX"
echo "wget https://github.com/bluenviron/mediamtx/releases/download/v1.19.2/mediamtx_v1.19.2_linux_amd64.tar.gz"
echo "tar -xzf mediamtx_v1.19.2_linux_amd64.tar.gz"
echo "./mediamtx"

echo "TERMINAL#2 Stream your video file into it on a loop"
echo "apt install ffmpeg -y"
echo "ffmpeg -re -stream_loop -1 -i stair.video.mp4 -c copy -f rtsp rtsp://localhost:8554/stream"


python src/download_yolo26.py --variant n --imgsz 640x360 --int8

# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# INT8 model: runs on AMX-INT8 (Xeon) and AVX-VNNI-INT8 (client CPUs like Panther Lake) for a consistent accelerated comparison.
taskset -c 0 python src/yolo-cv-benchmark.py \
        --model models/yolo26n_640x384_int8_openvino_model \
        --input-fps 8 \
        --rtsp-frame-size 640x360 \
        --imgsz 640x384 \
        --rtsp-urls "rtsp://localhost:8554/stream" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7

