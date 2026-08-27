python src/download_yolo26.py --variant n --imgsz 640x360 --int8

# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# INT8 model: runs on AMX-INT8 (Xeon) and AVX-VNNI-INT8 (client CPUs like Panther Lake) for a consistent accelerated comparison.
taskset -c 0 python src/yolo-cv-benchmark.py \
	--model models/yolo26n_640x384_int8_openvino_model \
	--input-fps 8 \
	--rtsp-frame-size 640x360 \
	--imgsz 640x384 \
	--rtsp-urls "rtsp://192.168.0.1:7447/Ak1tLOubDssSW84k" \
	--host 0.0.0.0 --port 8080 --classes 0,2,3,5,7


# SMALL MODEL
# python src/download_yolo26.py --variant s --imgsz 640x360 --int8
# --model models/yolo26s_640x384_int8_openvino_model
