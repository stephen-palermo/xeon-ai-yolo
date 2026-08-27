python src/download_yolo26_tensorrt.py --variant n --imgsz 640x360 --int8

# NVIDIA GPU run (e.g. an L4 already installed in the server) via TensorRT.
# Requires the NVIDIA driver + TensorRT to be installed and the GPU visible:
#   nvidia-smi
#   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#
# Rule of thumb: keep --imgsz equal to the model's exported size. Pick the resolution at export time (that's what changes performance); use --imgsz just to match it.
# This is the NVIDIA-GPU counterpart to gpu.1-cam.yolo26n.sh (Intel iGPU/OpenVINO)
# and 1-core.1-cam.yolo26n.sh (Intel CPU/OpenVINO), so the same model/resolution
# and RTSP stream can be benchmarked across all three for a direct comparison.
python src/yolo-cv-benchmark.py \
	--nvidia-gpu \
	--model models/yolo26n_640x384_int8.engine \
	--input-fps 8 \
	--rtsp-frame-size 640x360 \
	--imgsz 640x384 \
	--rtsp-urls "rtsp://192.168.0.1:7447/Ak1tLOubDssSW84k" \
	--host 0.0.0.0 --port 8080 --classes 0,2,3,5,7


# SMALL MODEL
# python src/download_yolo26_tensorrt.py --variant s --imgsz 640x360 --int8
# --model models/yolo26s_640x384_int8.engine
