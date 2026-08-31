python src/download_yolo26.py --variant n --imgsz 640x360 --int8

# Intel Core Ultra integrated GPU run (OpenVINO 'intel:gpu').
# Requires the Intel GPU compute runtime on the host (see setup.ov_2026.3.0.sh):
#   sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
# Verify the GPU is visible:
#   python -c "import openvino as ov; print(ov.Core().available_devices)"   # expect 'GPU'
#
# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# The same INT8 IR runs on the iGPU (Xe/Arc INT8 via DP4a/XMX); no separate GPU model is needed.
python src/yolo-cv-benchmark.py \
	--gpu \
	--model models/yolo26n_640x384_int8_openvino_model \
	--input-fps 8 \
	--rtsp-frame-size 640x360 \
	--imgsz 640x384 \
	--rtsp-urls "rtsp://192.168.0.1:7447/Ak1tLOubDssSW84k" \
	--host 0.0.0.0 --port 8080 --classes 0,2,3,5,7


# SMALL MODEL
# python src/download_yolo26.py --variant s --imgsz 640x360 --int8
# --model models/yolo26s_640x384_int8_openvino_model
