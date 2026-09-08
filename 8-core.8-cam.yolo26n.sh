python src/download_yolo26.py --variant n --imgsz 640x640 --int8

# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# INT8 model: runs on AMX-INT8 (Xeon) and AVX-VNNI-INT8 (client CPUs like Panther Lake) for a consistent accelerated comparison.
taskset -c 0-7 python src/yolo-cv-benchmark.py \
        --model  models/yolo26n_640x640_int8_openvino_model \
        --input-fps 8 \
        --rtsp-frame-size 640x360 \
        --imgsz 640x640 \
        --rtsp-urls \
        "rtsp://192.168.0.1:7447/Ak1tLOubDssSW84k,rtsp://192.168.0.1:7447/8cmoizs1IAEZzW1o,rtsp://192.168.0.1:7447/4ch65PdnVbl3QuVc,rtsp://192.168.0.1:7447/Tr6j02BNH2I3yN2D,rtsp://192.168.0.1:7447/IpXWPzT93olpaWHn,rtsp://192.168.0.1:7447/DcLe5ROyvyvnKpKc,rtsp://192.168.0.1:7447/Y2SUx4WGQkFPpnvJ,rtsp://192.168.0.1:7447/m0ZwcWE4CNDQ6KwG" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7
