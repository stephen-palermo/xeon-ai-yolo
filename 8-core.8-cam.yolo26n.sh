python src/download_yolo26.py --variant n --imgsz 640x360 --int8

# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# INT8 model: runs on AMX-INT8 (Xeon) and AVX-VNNI-INT8 (client CPUs like Panther Lake) for a consistent accelerated comparison.
taskset -c 0-7 python src/yolo-cv-benchmark.py \
        --model  models/yolo26n_640x384_int8_openvino_model \
        --input-fps 8 \
        --rtsp-frame-size 640x360 \
        --imgsz 640x384 \
        --rtsp-urls \
        "rtsp://192.168.0.1:7447/vAyzQCHWhkGoMJGP,rtsp://192.168.0.1:7447/qiVm3ApQrrGY2eh0,rtsp://192.168.0.1:7447/Lg3lNKMB9eHNNG9d,rtsp://192.168.0.1:7447/LsSYA17huveMeVrp,rtsp://192.168.0.1:7447/0w5jXmmQk5vz5JJk,rtsp://192.168.0.1:7447/GWTdfOGtbtHoIAis,rtsp://192.168.0.1:7447/BkJGjlcDf79me9RG,rtsp://192.168.0.1:7447/7vkkkaiViONajygE" \
        --host 0.0.0.0 --port 8080 --classes 0,2,3,5,7
